# flootyWebUi.jl — xterm.js terminal UI on top of the PTY layer.
#
# Transport: Webviews.jl v1.2.0 (native WKWebView + NSApplication event loop).
# Replaces WebUI entirely. Why (observed during live runs):
#   * webui's macOS WebView window would not activate: no Dock icon, stayed
#     behind every other window; webui_wait parked forever.
#   * any polling of webui_is_shown/webui_wait_async spams the webui HTTP
#     server (`[User] webui_is_shown([1])` floods); there is no clean
#     event-driven lifecycle on that path.
# Webviews.jl instead gives us:
#   * a real NSApplication loop — run(webview) parks in the OS event loop and
#     returns when the app terminates (last window closed / stop:).
#   * proper Dock icon + activation: its Apple backend sets
#     NSApplicationActivationPolicyRegular and activateIgnoringOtherApps,
#     then makeKeyAndOrderFront (platforms/apple/utils.jl).
#   * promise-style JS↔Julia bridge: bind(wv, name => f) injects a global
#     async JS function; Julia→JS pushes go through dispatch()+eval!.
#
# Contract with assets/index.html (kept deliberately close to the old one):
#   JS → Julia: bound globals terminal_create/terminal_begin_live/
#               terminal_kill/flootyCloseWindow/flootyMinimizeWindow take ONE
#               JSON string argument and RETURN a JSON string ({success, ...}).
#               The flooty* names MUST NOT collide with index.html's own function
#               declarations: the page script parses AFTER bind()'s document-start
#               shim and would overwrite a same-named binding (silent no-op close,
#               flakily restored by bind()'s racing eval!).
#   Julia → JS: dispatch/eval! runs window.handleTerminalOutput(id, text) and
#               window.handleTerminalExit(id, code) on the main thread.
#
# The glue stays THIN: every operation is a plain function over a
# TerminalEntry, so the handler layer runs without any window at all
# (see test/webUiSmoke.jl).
using UUIDs
using JSON3
import Webviews
import HTTP

# ---- state -------------------------------------------------------------------

"""
One live UI terminal: its PTY handle, a rolling collected-output log (for
headless testing), and the pump task forwarding PTY output to the frontend.
"""
mutable struct TerminalEntry
    id::String
    pty::PtyHandle
    collected::Vector{String}
    lock::ReentrantLock
    exited::Bool
    exitCode::Union{Nothing,Int}
    pumpTask::Union{Nothing,Task}
    # Bind-handshake buffering: display-safe chunks accumulate here until the
    # page PULLS them via terminal_begin_live (bindings resolve long after
    # creation; live pushes fired earlier would hit an unloaded page and be
    # DROPPED — the intermittent blank tab). After the pull, chunks flow live.
    livePush::Bool
    pendingDisplay::Vector{String}
    # Live-push coalescing (see CHUNK_FLUSH_*): display text accumulated since
    # the last flush, plus the armed flush timer (nothing when none is pending).
    pendingLive::String
    liveFlushTimer::Union{Nothing,Timer}
    # When the last live push went out. The live push is RATE LIMITED (not
    # delayed by a fixed timer): a keystroke echo must leave immediately, and
    # only a burst waits out the interval so it stays batched.
    lastLiveFlush::Float64
    # Server-side shadow MODEL of this terminal, fed the same PTY bytes the
    # page receives. Source of truth for agent snapshots and headless tests;
    # the page xterm remains the renderer of record.
    shadow::ShadowTerminal.ShadowTerm
    # The model is fed through a QUEUE on its own task: parsing the PTY
    # stream into the model costs ~3 MiB/s (measured), and doing that inline
    # in the output tap throttled the LIVE PUSH to the page to the same rate.
    # The queue keeps the display path independent of the model; readers
    # (shadowLines/shadowText/...) drain it first, so they still see every
    # byte that was enqueued.
    shadowQueue::Channel{String}
    shadowEnqueued::Int   # chunks handed to the queue
    shadowFed::Int        # chunks actually parsed into the model
    shadowTask::Union{Nothing,Task}   # the worker feeding the model
    # The model's own lock. The worker parses 16 KiB at a time (~5 ms), and
    # holding the ENTRY's lock for that would stall the live push (the tap
    # needs the same lock) — i.e. the shadow model would slow down typing.
    shadowLock::ReentrantLock
end

const COLLECTED_MAX_CHUNKS = 4096

const ACTIVE_TERMINALS = Dict{String,TerminalEntry}()
const terminalRegistryLock = ReentrantLock()

function defaultTerminalCmd()
    if Sys.iswindows()
        for name in ("pwsh", "powershell")
            p = Sys.which(name)
            p !== nothing && return `$p -NoLogo`
        end
        return `cmd.exe`
    else
        for name in ("zsh", "bash", "sh")
            p = Sys.which(name)
            p !== nothing && return `$p -i`
        end
        return `/bin/sh -i`
    end
end
"`Base.@kwdef` configuration for the terminal UI window."
Base.@kwdef mutable struct TerminalUiConfig
    cmd::Cmd = defaultTerminalCmd()
    width::Int = 1100
    height::Int = 750
    # CHAT-ONLY boot: no terminal tab on load — the conversation takes the window,
    # and the first assistant prompt (or the agent's newTerminal action) spawns the
    # terminal session. Off by default so probes still get a session to adopt.
    chatOnly::Bool = false
end

const terminalUiConfig = Ref{TerminalUiConfig}(TerminalUiConfig())
const terminalUiWebview = Ref{Union{Nothing,Webviews.Webview}}(nothing)
const terminalUiRunning = Ref(false)
const terminalUiShutdownLock = ReentrantLock()
const terminalUiAssetServer = Ref{Any}(nothing)  # HTTP.serve! server for assets
const terminalUiAssetPort = Ref(0)

# ---- GPU renderer state -------------------------------------------------------
#
# The page pushes which xterm renderer is actually painting ({"webgl2":bool,
# "terminals":[{id,state,reason}]}) every time it creates, adopts or closes a
# terminal. The webview has no devtools, so this push is the only way to SEE
# whether the GPU (WebGL2) path is live — the app logs it, probes assert on it.
const GPU_RENDERER_REPORT = Ref{Any}(nothing)

"""
Last renderer report pushed by the UI page (`nothing` until the page reports).

Shape: `{webgl2::Bool, terminals::Vector}` where each terminal entry carries
`id`, `state` (`"webgl"` | `"dom"`) and `reason` (why it fell back).
"""
gpuRendererReport() = GPU_RENDERER_REPORT[]


# ---- registry helpers ---------------------------------------------------------

getEntry(id::String) =
    lock(() -> get(ACTIVE_TERMINALS, id, nothing), terminalRegistryLock)

"Number of currently active UI terminals."
activeTerminalCount() = lock(() -> length(ACTIVE_TERMINALS), terminalRegistryLock)

"All active terminal ids (snapshot)."
activeTerminalIds() = lock(() -> collect(keys(ACTIVE_TERMINALS)), terminalRegistryLock)

"Collected output text for one terminal (concatenated pushed chunks)."
collectedText(entry::TerminalEntry) = lock(() -> join(entry.collected), entry.lock)

"""
    shadowLines(entry) :: Vector{String}

The terminal screen as the SERVER sees it: one string per row, trailing
blanks stripped. This is the shadow model, not the page DOM, so it is
available with no window attached and cannot lag behind the bytes the
agent just produced.
"""
shadowLines(entry::TerminalEntry) = (drainShadow!(entry);
    lock(() -> ShadowTerminal.shadowLines(entry.shadow), entry.shadowLock))

"The shadow screen as one newline-joined string."
shadowText(entry::TerminalEntry) = (drainShadow!(entry);
    lock(() -> ShadowTerminal.shadowText(entry.shadow), entry.shadowLock))

"Cursor position (0-based row/col) and visibility from the shadow model."
shadowCursor(entry::TerminalEntry) = (drainShadow!(entry);
    lock(() -> ShadowTerminal.shadowCursor(entry.shadow), entry.shadowLock))

"Resize the shadow model to match a PTY winsize change."
shadowResize!(entry::TerminalEntry, cols::Integer, rows::Integer) =
    lock(() -> ShadowTerminal.shadowResize!(entry.shadow, cols, rows), entry.shadowLock)

# ---- shadow feed queue --------------------------------------------------------
#
# The shadow model is a SERVER-SIDE copy of the terminal: every PTY byte is
# parsed into it so agent snapshots and headless tests can read the screen
# without the browser. That parse costs ~3 MiB/s (measured), so doing it
# inline in the output tap throttled the LIVE PUSH to the page to the same
# rate — a `cat` of a big file crawled in. The model now runs on its own
# task behind a queue, and the live push is never delayed by it. Readers
# drain the queue first, so they still observe every enqueued byte.

"Start the task that feeds queued PTY bytes into the shadow model."
function startShadowWorker!(entry::TerminalEntry)
    entry.shadowTask = @async begin
        for text in entry.shadowQueue
            try
                # The MODEL's lock, not the entry's: the entry lock is what the
                # live push takes, and the model parse must not delay it.
                lock(entry.shadowLock) do
                    ShadowTerminal.shadowFeed!(entry.shadow, text)
                end
            catch
            end
            lock(entry.lock) do
                entry.shadowFed += 1
            end
        end
    end
    return entry.shadowTask
end

"Hand PTY text to the shadow model's worker (never blocks the caller)."
function enqueueShadow!(entry::TerminalEntry, text::AbstractString)
    isempty(text) && return nothing
    lock(entry.lock) do
        entry.shadowEnqueued += 1
    end
    try
        put!(entry.shadowQueue, String(text))
    catch
        # Queue closed (terminal killed): the model stops here, the page does
        # not care — the tail still holds the raw bytes.
    end
    return nothing
end

"""
Wait (bounded) until the shadow model has parsed everything enqueued.

Readers of the model call this first, so a snapshot taken right after output
arrived still sees it. Returns true when the model caught up.
"""
function drainShadow!(entry::TerminalEntry; timeoutS = 2.0)
    deadline = time() + timeoutS
    while true
        caughtUp = lock(entry.lock) do
            entry.shadowFed >= entry.shadowEnqueued
        end
        caughtUp && return true
        time() >= deadline && return false
        sleep(0.001)
    end
end

"Stop the shadow worker (idempotent)."
function stopShadowWorker!(entry::TerminalEntry)
    try
        close(entry.shadowQueue)
    catch
    end
    return nothing
end

# ---- chunk encoding -----------------------------------------------------------

"""
Decode raw PTY bytes defensively: valid UTF-8 passes through untouched and any
invalid byte becomes U+FFFD, so downstream JS embedding never sees malformed
text.
"""
function decodeChunk(bytes::Vector{UInt8})::String
    isvalid(String, bytes) && return String(copy(bytes))
    out = IOBuffer()
    i = 1
    n = length(bytes)
    replacement = UInt8[0xEF, 0xBF, 0xBD]   # UTF-8 encoding of U+FFFD
    while i <= n
        ok = false
        for j in min(i + 3, n):-1:i
            candidate = bytes[i:j]
            if isvalid(String, candidate)
                write(out, String(candidate))
                i = j + 1
                ok = true
                break
            end
        end
        if !ok
            write(out, replacement)
            i += 1
        end
    end
    return String(take!(out))
end

# ---- JS string building -------------------------------------------------------

"""Escape a string for safe embedding inside a double-quoted JS literal."""
function escapeJsString(s::AbstractString)
    return replace(s,
                   '\\' => "\\\\", '"' => "\\\"", '\n' => "\\n",
                   '\r' => "\\r", '\t' => "\\t",
                   '<' => "\\u003c", '>' => "\\u003e", '&' => "\\u0026")
end

jsonOk() = "{\"success\":true}"

jsonCreated(id::String) = "{\"success\":true,\"terminal_id\":\"$id\"}"

errorJson(msg::AbstractString) =
    "{\"success\":false,\"error\":\"" * escapeJsString(String(msg)) * "\"}"

# ---- data-plane channel (binary WebSocket) ----------------------------------
#
# The CONTROL plane (the bindings: takeover, prompts, snapshots, popups) stays
# on the Webviews RPC. The DATA plane — the PTY bytes — moves over ONE binary
# WebSocket per terminal: raw bytes, no JSON, no per-chunk eval!, no
# per-keystroke RPC round trip. This is the transport terax-ai uses
# (src-tauri/src/modules/pty/protocol.rs) and the one FlootyTerminal ships.
#
# The frame layout is the same as FlootyTerminal's:
#   [type:1][flags:1][termId:2 LE][len:4 LE][payload]
const CH_FRAME_OUTPUT  = 0x01
const CH_FRAME_ACK     = 0x02
const CH_FRAME_INPUT   = 0x03
const CH_FRAME_RESIZE  = 0x04
const CH_FRAME_OPEN    = 0x05
const CH_FRAME_OPENED  = 0x06
const CH_FRAME_CLOSE   = 0x07
const CH_FRAME_EXIT    = 0x08
const CH_FRAME_CLASS   = 0x09
# The channel's readiness handshake: the server sends HELLO, the page answers
# READY, and ONLY THEN do the bytes travel over the socket. A channel that
# never completes the round trip is never used, so a broken socket cannot
# swallow the terminal's output.
const CH_FRAME_HELLO   = 0x0c
const CH_FRAME_READY   = 0x0d
const CH_HEADER_BYTES  = 8

const CHANNEL_LOCK = ReentrantLock()
# session id -> the socket that carries its bytes (plus the byte offsets, so
# the page can see a gap instead of silently losing bytes)
mutable struct ChannelBinding
    ws::HTTP.WebSockets.WebSocket
    offset::UInt64   # cumulative bytes sent
    acked::UInt64    # cumulative bytes the page confirmed
    boundAt::Float64 # when the page opened it
    ready::Bool      # the page answered the HELLO: the socket carries bytes
    sentAt::Float64  # when the last byte went out (the staleness clock)
end
const CHANNEL_SESSIONS = Dict{String,ChannelBinding}()
const CHANNEL_PORT = Ref(0)
# A channel that carried bytes but has not had them acknowledged for this
# long is treated as dead (the page's socket died) and the bytes go back over
# the eval path. A channel that never carried anything is NOT stale: it is
# simply waiting for its first output.
const CHANNEL_STALE_S = 3.0

"Encode one channel frame."
function encode_channel_frame(type::UInt8, termId::UInt16, payload::Vector{UInt8})
    out = Vector{UInt8}(undef, CH_HEADER_BYTES + length(payload))
    out[1] = type
    out[2] = 0x00
    out[3] = termId % UInt8
    out[4] = (termId >> 8) % UInt8
    n = UInt32(length(payload))
    out[5] = n % UInt8
    out[6] = (n >> 8) % UInt8
    out[7] = (n >> 16) % UInt8
    out[8] = (n >> 24) % UInt8
    length(payload) > 0 && copyto!(out, CH_HEADER_BYTES + 1, payload, 1, length(payload))
    return out
end

"Decode one channel frame; nothing on a short read."
function decode_channel_frame(buf::Vector{UInt8})
    length(buf) < CH_HEADER_BYTES && return nothing
    type = buf[1]
    termId = UInt16(buf[3]) | (UInt16(buf[4]) << 8)
    n = UInt32(buf[5]) | (UInt32(buf[6]) << 8) | (UInt32(buf[7]) << 16) | (UInt32(buf[8]) << 24)
    length(buf) < CH_HEADER_BYTES + n && return nothing
    payload = Vector{UInt8}(undef, Int(n))
    n > 0 && copyto!(payload, 1, buf, CH_HEADER_BYTES + 1, Int(n))
    return (type, termId, payload)
end

"The channel binding for a session, if any."
function channel_binding(id::AbstractString)
    lock(CHANNEL_LOCK) do
        get(CHANNEL_SESSIONS, String(id), nothing)
    end
end

"Bind a session's id to a socket (the page opened its channel)."
function bind_channel!(id::AbstractString, ws)
    lock(CHANNEL_LOCK) do
        CHANNEL_SESSIONS[String(id)] = ChannelBinding(ws, UInt64(0), UInt64(0), time(), false, time())
    end
    return nothing
end

"Drop a session's channel."
function unbind_channel!(id::AbstractString)
    lock(CHANNEL_LOCK) do
        delete!(CHANNEL_SESSIONS, String(id))
    end
    return nothing
end

"""
Serve the data-plane channel: one WebSocket per terminal, raw bytes.

The first frame opens a terminal (its id, when the page already has one);
after that the socket carries the PTY's output and the user's keystrokes.
"""
function start_channel_server!()
    server = HTTP.WebSockets.listen!("127.0.0.1", 0) do ws
        try
            handle_channel_connection!(ws)
        catch e
            @debug "flootyWebUi: channel connection failed" exception = e
        end
    end
    hostBound, portBound = try
        Sockets.getsockname(server.listener.server::Base.IOServer)
    catch err
        error("start_channel_server!: could not read the bound socket: ",
              sprint(showerror, err))
    end
    CHANNEL_PORT[] = Int(portBound)
    return CHANNEL_PORT[]
end

function handle_channel_connection!(ws::HTTP.WebSockets.WebSocket)
    boundId = ""
    @debug "[channel] connection opened"
    try
        for msg in ws
            raw = msg isa String ? Vector{UInt8}(codeunits(msg)) : msg
            frame = decode_channel_frame(raw)
            frame === nothing && continue
            type, termId, payload = frame
            if type == CH_FRAME_OPEN
                # The page names the session it already created (the control
                # plane made it); an empty id means "make me one".
                id = String(copy(payload))
                if isempty(id)
                    id = create_terminal_for_channel!()
                end
                isempty(id) && continue
                boundId = id
                @debug string("[channel] open id=", id)
                bind_channel!(id, ws)
                try
                    # HELLO: the page answers READY, and only then does this
                    # socket carry the terminal's bytes. (The term id is zero:
                    # the app's session ids are UUIDs and the page's channel
                    # is per terminal, so it does not need one.)
                    HTTP.WebSockets.send(ws, encode_channel_frame(CH_FRAME_HELLO, UInt16(0), Vector{UInt8}()))
                    @debug string("[channel] HELLO sent id=", id)
                catch e
                    # A send that fails must not be silent: it takes the whole
                    # data plane down and the page falls back to the eval path.
                    @debug string("[channel] HELLO send failed exception=", e)
                end
            elseif type == CH_FRAME_INPUT
                isempty(boundId) && continue
                entry = getEntry(boundId)
                entry === nothing && continue
                isempty(payload) && continue
                try
                    writeToPty(entry.pty, payload; source = :user)
                catch e
                    @debug "flootyWebUi: channel input write failed" id exception = e
                end
            elseif type == CH_FRAME_RESIZE
                length(payload) >= 4 || continue
                isempty(boundId) && continue
                cols = UInt16(payload[1]) | (UInt16(payload[2]) << 8)
                rows = UInt16(payload[3]) | (UInt16(payload[4]) << 8)
                handleTerminalResize(boundId, Int(rows), Int(cols))
            elseif type == CH_FRAME_CLOSE
                isempty(boundId) || unbind_channel!(boundId)
            elseif type == CH_FRAME_READY
                isempty(boundId) && continue
                lock(CHANNEL_LOCK) do
                    b = get(CHANNEL_SESSIONS, boundId, nothing)
                    b === nothing || (b.ready = true)
                end
            elseif type == CH_FRAME_ACK
                # The page confirms what it consumed; recorded so the sender
                # can be throttled by credit (like terax's output credit).
                length(payload) >= 8 || continue
                isempty(boundId) && continue
                off = UInt64(0)
                for i in 0:7
                    off |= UInt64(payload[i + 1]) << (8 * i)
                end
                lock(CHANNEL_LOCK) do
                    b = get(CHANNEL_SESSIONS, boundId, nothing)
                    b === nothing || (b.acked = off)
                end
            end
        end
    catch e
        @debug string("[channel] connection ended exception=", e)
    finally
        isempty(boundId) || unbind_channel!(boundId)
    end
    return nothing
end

"Create a terminal for a channel that did not name one."
function create_terminal_for_channel!()
    id = try
        handleTerminalCreate(defaultTerminalCmd())
    catch e
        @debug "flootyWebUi: channel terminal create failed" exception = e
        ""
    end
    return String(id)
end

"Push a session's output over its channel (binary), if it has one."
function pushOutputToChannel(id::AbstractString, text::AbstractString)
    binding = channel_binding(id)
    binding === nothing && return false
    # Only a channel that completed the readiness handshake carries bytes; one
    # that never did (a webview whose handshake stalled) is left to the eval
    # path, so the terminal's output is never swallowed.
    binding.ready || return false
    # The channel is stale only if it carried bytes the page never
    # acknowledged. A channel that has not carried anything yet is waiting
    # for its first output, not dead.
    if binding.offset > 0 && binding.acked == 0 && time() - binding.sentAt > CHANNEL_STALE_S
        @debug "flootyWebUi: channel went silent, falling back" id
        unbind_channel!(id)
        return false
    end
    bytes = Vector{UInt8}(codeunits(String(text)))
    payload = Vector{UInt8}(undef, 8 + length(bytes))
    offset = lock(CHANNEL_LOCK) do
        b = get(CHANNEL_SESSIONS, String(id), nothing)
        b === nothing && return UInt64(0)
        off = b.offset
        b.offset = off + UInt64(length(bytes))
        off
    end
    for i in 0:7
        payload[i + 1] = (offset >> (8 * i)) % UInt8
    end
    length(bytes) > 0 && copyto!(payload, 9, bytes, 1, length(bytes))
    try
        HTTP.WebSockets.send(binding.ws,
            encode_channel_frame(CH_FRAME_OUTPUT, UInt16(0), payload))
        lock(CHANNEL_LOCK) do
            b = get(CHANNEL_SESSIONS, String(id), nothing)
            b === nothing || (b.sentAt = time())
        end
        return true
    catch e
        @debug "flootyWebUi: channel send failed" id exception = e
        unbind_channel!(id)
        return false
    end
end

"""
Push a session's display text to the page.

The page's binary channel is the ONLY data plane: the bytes go over the
socket (see pushOutputToChannel). There is no eval fallback any more — a
page whose channel never opened shows a terminal that never updates, and
the page says so itself (the channel's close handler).
"""
function pushOutputToWindow(wv, id::String, text::AbstractString)
    wv === nothing && return nothing
    pushOutputToChannel(id, text)
    return nothing
end

# ---- agent feedback popups ---------------------------------------------------

"""
    pushAgentFeedbackToWindow(wv, title, body; kind = "info", sessionId = "")

Run JS calling `window.handleAgentFeedback(sessionId, title, body, kind)` on
the page. The page renders it as a transient glass popup in the lower-right
stack (Warp ambient-agent style) — ONLY while the session's tab is active —
and acknowledges via the `flootyAgentPopupShown` binding. `sessionId` tags
the popup with the OWNING tab's PTY id so concurrent per-session agents
never cross-pollinate each other's tabs; "" keeps the legacy broadcast.
No-op when no window; failures never propagate (feedback must not crash the
pump).
"""
function pushAgentFeedbackToWindow(wv, title::AbstractString,
                                   body::AbstractString,
                                   kind::AbstractString = "info";
                                   sessionId::AbstractString = "")
    wv === nothing && return nothing
    # Julia-side trace of EVERY popup pushed to the window (render ack and
    # fade/dismiss come back as [agent-popup-shown] / [ui-debug] lines).
    @debug string("[agent-popup-push] sid=", isempty(sessionId) ? "*" :
            first(sessionId, 8), " (", kind, ") ", title, ": ", body)
    script = "if (typeof window.handleAgentFeedback === 'function') { " *
             "window.handleAgentFeedback(\"" * escapeJsString(String(sessionId)) *
             "\", \"" * escapeJsString(String(title)) *
             "\", \"" * escapeJsString(String(body)) * "\", \"" *
             escapeJsString(String(kind)) * "\"); }"
    try
        Webviews.dispatch(wv) do
            try
                Webviews.eval!(wv, script)
            catch e2
                @debug "flootyWebUi: eval handleAgentFeedback failed" exception = e2
            end
        end
    catch e
        @debug "flootyWebUi: pushAgentFeedbackToWindow failed" exception = e
    end
    return nothing
end

"""Run JS calling window.handleTerminalExit(id, code) (no-op when no window).

`exitCode` is interpolated into the JS payload as a numeric literal, not a
string — the page reads it via `handleTerminalExit(id, code)` as an integer.
"""
function pushExitToWindow(wv, id::String, exitCode::Int)
    wv === nothing && return nothing
    script = "if (typeof window.handleTerminalExit === 'function') { " *
             "window.handleTerminalExit(\"" * escapeJsString(id) *
             "\", $exitCode); }"
    try
        Webviews.dispatch(wv) do
            try
                Webviews.eval!(wv, script)
            catch e2
                @debug "flootyWebUi: eval handleTerminalExit failed" exception = e2
            end
        end
    catch e
        @debug "flootyWebUi: pushExitToWindow failed" exception = e
    end
    return nothing
end

"""
    attachSessionTerminalToWindow(wv, id, name = "julia";
                                  replayText = "", view = "user")

Tell the PAGE to open an xterm tab bound to backend terminal `id` (previously
created by `adoptSessionTerminal!`). `replayText` seeds the pane with existing
history so the tab is never blank; `view` ("agent" | "user") marks the tab's
role so focusing it reports `flootyViewFocused`. The same session may be
attached TWICE (a user tab + an agent tab of ONE PTY) — the page fans output
to every tab bound to the id. Same dispatch+eval! push channel as the popups;
no-op without a window. A missing page-side function is reported via the
flootyDebug channel instead of failing silently.
"""
function attachSessionTerminalToWindow(wv, id::AbstractString,
                                       name::AbstractString = "julia";
                                       replayText::AbstractString = "",
                                       view::AbstractString = "user",
                                       parent_session::AbstractString = "")
    wv === nothing && return nothing
    script = "if (typeof adoptTerminal === 'function') { " *
             "adoptTerminal(\"" * escapeJsString(String(id)) * "\", \"" *
             escapeJsString(String(name)) * "\", \"" *
             escapeJsString(String(replayText)) * "\", \"" *
             escapeJsString(String(view)) * "\", \"" *
             escapeJsString(String(parent_session)) * "\"); } " *
             "else { window.flootyDebug && " *
             "window.flootyDebug('adoptTerminal MISSING on page'); }"
    # The window's script may not be running yet (a session adopted while the UI
    # is still booting). Evaluating then is a SILENT no-op — the tab never
    # opened and the session's chat had nowhere to go. Poll the page through the
    # report binding on the RUN LOOP (a plain @async task does not get scheduled
    # while the native loop runs), then attach ONCE.
    probe = "if (typeof adoptTerminal === 'function') { " *
            "window.flootyTerminalBufferReport(JSON.stringify(" *
            "{terminal_id: \"__adopt_ready__\", text: \"1\"})); }"
    attempts = Ref(0)
    clearKey = () -> lock(() -> (delete!(TERMINAL_BUFFER_CACHE, "__adopt_ready__");
                                 nothing), TERMINAL_BUFFER_LOCK)
    attach = function ()
        try
            Webviews.dispatch(wv) do
                try
                    Webviews.eval!(wv, script)
                catch e2
                    @debug "flootyWebUi: eval adoptTerminal failed" exception = e2
                end
            end
        catch e
            @debug "flootyWebUi: attachSessionTerminalToWindow failed" exception = e
        end
        return nothing
    end
    settle = function ()
        ready = lock(() -> haskey(TERMINAL_BUFFER_CACHE, "__adopt_ready__"),
                     TERMINAL_BUFFER_LOCK)
        clearKey()
        ready && return attach()
        if attempts[] >= 40
            @warn "flootyWebUi: page never became ready — adopted tab not opened" id
            return nothing
        end
        Webviews.set_timeout(step, wv, 0.25)
        return nothing
    end
    step = function ()
        attempts[] += 1
        clearKey()
        try
            Webviews.dispatch(wv) do
                try
                    Webviews.eval!(wv, probe)
                catch e2
                    @debug "flootyWebUi: adopt readiness probe failed" exception = e2
                end
            end
        catch e
            @debug "flootyWebUi: adopt readiness dispatch failed" exception = e
        end
        Webviews.set_timeout(settle, wv, 0.1)
        return nothing
    end
    Webviews.set_timeout(step, wv, 0.05)
    return nothing
end

# ---- ownership arbitration bridge (Warp TuiInputTarget-style) -------------------

# Per-session ownership arbitration (ONE AGENT PER SESSION): every adopted
# ReplSession is registered under its terminal id, and the page reports tab
# focus so bindings act on whichever session's TAB IS ACTIVE — not on one
# global session. Plain spawned terminals have no agent contract.
const OWNERSHIP_SESSIONS = Dict{String,ReplSession}()
# Reverse index for `sessionIdFor`: the old linear scan made every agent line
# O(adopted sessions); an IdDict keeps the lookup O(1).
const OWNERSHIP_IDS = IdDict{ReplSession,String}()
const OWNERSHIP_SESSIONS_LOCK = ReentrantLock()
const ACTIVE_OWNERSHIP = Ref{Union{Nothing,ReplSession}}(nothing)
const ACTIVE_OWNERSHIP_ID = Ref{String}("")

"Route ownership actions to the session bound to terminal id (tab focus)."
function setActiveOwnershipSession(id::AbstractString)
    sess = lock(() -> get(OWNERSHIP_SESSIONS, String(id), nothing),
                OWNERSHIP_SESSIONS_LOCK)
    ACTIVE_OWNERSHIP[] = sess
    ACTIVE_OWNERSHIP_ID[] = sess === nothing ? "" : String(id)
    return sess
end

# Hooks fired with the ReplSession whenever a NEW terminal gets one —
# including plain "+ New Terminal" tabs created from the PAGE. Higher layers
# (Flooty) register here to attach their per-session machinery
# (standing agent, ambient monitor) so EVERY new tab carries its own agent.
const SESSION_TERMINAL_HOOKS = Vector{Function}()

"Register `f(sess)` to run whenever a new terminal-backed session appears."
addSessionTerminalHook!(f::Function) = push!(SESSION_TERMINAL_HOOKS, f)

# Re-parent hook: the app's session tree must follow a session that moved
# under a different parent (the tree navigation's "make this my parent").
const SESSION_REPARENT_HOOKS = Vector{Function}()
addSessionReparentHook!(f::Function) = push!(SESSION_REPARENT_HOOKS, f)

"""
    notifySessionReparent!(childId, parentId)

Tell the app that `childId` now hangs under `parentId` ("" = a root).
"""
function notifySessionReparent!(childId::AbstractString, parentId::AbstractString)
    for f in copy(SESSION_REPARENT_HOOKS)
        try
            Base.invokelatest(f, String(childId), String(parentId))
        catch err
            @debug "flootyWebUi: session-reparent hook failed" exception = err
        end
    end
    return nothing
end

# Close hook: the app unlinks a session from the tree when its terminal dies.
# Sessions whose close hooks already fired. A terminal can reach recordExit!
# through more than one path (the page kill AND the natural-exit pump); the
# hooks must run ONCE per session, so the id is claimed before they run.
const CLOSE_HOOKED = Set{String}()
const CLOSE_HOOKED_LOCK = ReentrantLock()

const SESSION_TERMINAL_CLOSE_HOOKS = Vector{Function}()
addSessionTerminalCloseHook!(f::Function) = push!(SESSION_TERMINAL_CLOSE_HOOKS, f)


# Reconnect hooks: the app reattaches per-session machinery (ambient
# monitor, standing agent) to the NEW session after a revive, and is told
# the OLD one died so it stops sampling a dead tail.
const SESSION_RECONNECT_HOOKS = Vector{Function}()
addSessionReconnectHook!(f::Function) = push!(SESSION_RECONNECT_HOOKS, f)

function runSessionReconnectHooks!(oldSess, newSess)
    for f in copy(SESSION_RECONNECT_HOOKS)
        try
            Base.invokelatest(f, oldSess, newSess)
        catch err
            @debug "flootyWebUi: session-reconnect hook failed" exception = err
        end
    end
    return nothing
end

function runSessionTerminalCloseHooks!(sess)
    # Run ONCE per terminal: a terminal can reach the exit path twice (natural
    # exit, then the user closes the tab). The id is claimed here so the app
    # unlinks the session from its tree exactly once.
    cid = try
        sessionIdFor(sess)
    catch
        ""
    end
    if !isempty(cid)
        first = lock(CLOSE_HOOKED_LOCK) do
            cid in CLOSE_HOOKED ? false : (push!(CLOSE_HOOKED, cid); true)
        end
        first || return nothing
    end

    for f in copy(SESSION_TERMINAL_CLOSE_HOOKS)
        try
            Base.invokelatest(f, sess)
        catch err
            @debug "flootyWebUi: session-terminal close hook failed" exception = err
        end
    end
    return nothing
end

# ---- agent model picker -------------------------------------------------------
# The page's model selector is driven by the AGENT package (Flooty owns the
# endpoint and the config), but the webview bindings live here. So the agent side
# registers a provider/selector pair at load, exactly like the session hooks above;
# a process that never loads the agent (plain FlootyPTYLayer, probes) simply shows
# no models to pick.
const AGENT_MODEL_PROVIDER = Ref{Union{Nothing,Function}}(nothing)
# Split children: a bounded pool of TEMPORARY second panes. Each child is its
# own ReplSession; the page reports each creation/close. `SPLIT_SESSION_ID` is the
# most recently created child (so plain `target:"split"` keeps its old meaning);
# `SPLIT_CHILDREN` keeps the creation order for `split:1`, `split:2`, …
const SPLIT_SESSION_ID = Ref("")
const SPLIT_SESSION_LOCK = ReentrantLock()
const SPLIT_PARENT = Dict{String,String}()            # child id -> main id
const SPLIT_CHILDREN = Dict{String,Vector{String}}()  # main id -> [child ids]

"Most recently created split child, or nothing."
function splitSession()
    id = lock(SPLIT_SESSION_LOCK) do
        String(SPLIT_SESSION_ID[])
    end
    isempty(id) && return nothing
    return sessionForTerminalId(id)
end

"""
    setSplitSession!(id; parent_id = "", closed = false)

Page callback for one split child. `parent_id` is the main session that owns it
(captured from ACTIVE_OWNERSHIP_ID at creation); `closed=true` removes `id` from
the pool. An empty `id` is the legacy "clear" form.
"""
function setSplitSession!(id::AbstractString; parent_id::AbstractString = "",
                          closed::Bool = false)
    key = String(id)
    lock(SPLIT_SESSION_LOCK) do
        if closed
            parent = pop!(SPLIT_PARENT, key, "")
            isempty(parent) || begin
                list = get(SPLIT_CHILDREN, parent, String[])
                filter!(!=(key), list)
                isempty(list) ? delete!(SPLIT_CHILDREN, parent) :
                                (SPLIT_CHILDREN[parent] = list)
            end
            if SPLIT_SESSION_ID[] == key
                remaining = isempty(parent) ? String[] :
                            get(SPLIT_CHILDREN, parent, String[])
                SPLIT_SESSION_ID[] = isempty(remaining) ? "" : remaining[end]
            end
        elseif isempty(key)
            old = SPLIT_SESSION_ID[]
            SPLIT_SESSION_ID[] = ""
            isempty(old) || begin
                parent = pop!(SPLIT_PARENT, old, "")
                isempty(parent) || begin
                    list = get(SPLIT_CHILDREN, parent, String[])
                    filter!(!=(old), list)
                    isempty(list) ? delete!(SPLIT_CHILDREN, parent) :
                                    (SPLIT_CHILDREN[parent] = list)
                end
            end
        else
            SPLIT_SESSION_ID[] = key
            if !isempty(parent_id)
                # RE-PARENTING: a session that already has a parent moves to
                # the new one — it must leave the OLD parent's child list, or
                # the tree shows it twice.
                old = get(SPLIT_PARENT, key, "")
                if !isempty(old) && old != String(parent_id)
                    list = get(SPLIT_CHILDREN, old, String[])
                    filter!(!=(key), list)
                    isempty(list) ? delete!(SPLIT_CHILDREN, old) :
                                    (SPLIT_CHILDREN[old] = list)
                end
                SPLIT_PARENT[key] = String(parent_id)
                list = get!(SPLIT_CHILDREN, String(parent_id), String[])
                key in list || push!(list, key)
            end
        end
    end
    return nothing
end

"Main session that owns this split child, or nothing."
function splitParentSession(child)
    id = sessionIdFor(child)
    isempty(id) && return nothing
    pid = lock(SPLIT_SESSION_LOCK) do
        get(SPLIT_PARENT, id, "")
    end
    isempty(pid) && return nothing
    return sessionForTerminalId(pid)
end

"All split children of `main`, in creation order."
function splitChildren(main)
    id = sessionIdFor(main)
    isempty(id) && return ReplSession[]
    ids = lock(SPLIT_SESSION_LOCK) do
        copy(get(SPLIT_CHILDREN, id, String[]))
    end
    out = ReplSession[]
    for cid in ids
        sess = sessionForTerminalId(cid)
        sess === nothing || push!(out, sess)
    end
    return out
end

"Split child `n` (1-based) of `main`, or nothing."
function splitChildAt(main, n::Integer)
    n >= 1 || return nothing
    kids = splitChildren(main)
    n <= length(kids) ? kids[Int(n)] : nothing
end

"Number of live split children of `main`."
splitCount(main) = length(splitChildren(main))

"Newest split child of `main` (the snapshot's `split=` line), or nothing."
function splitChildSession(main)
    kids = splitChildren(main)
    isempty(kids) && return nothing
    return kids[end]
end

"Drop a dead split child from the pool (terminal kill)."
function clearSplitChild!(id::AbstractString)
    setSplitSession!(id; closed = true)
    return nothing
end

const AGENT_MODEL_SELECTOR = Ref{Union{Nothing,Function}}(nothing)
# CHAT-ONLY: with no session at all the assist box talks to the agent MODEL, not
# to a PTY. The app installs this handler; without one, prompts queue for the
# next adopted session (legacy behaviour).
const AGENT_CHAT_ONLY_HANDLER = Ref{Union{Nothing,Function}}(nothing)
# The app's ROOT chat session id (the page tags chat-only events with it) and the
# interaction reporter the page calls for typing/pause traces.
const CHAT_SESSION_PROVIDER = Ref{Union{Nothing,Function}}(nothing)
const CHAT_INTERACTION_HANDLER = Ref{Union{Nothing,Function}}(nothing)
# New chat session (child or root) requested from the session-type chooser.
const NEW_CHAT_HANDLER = Ref{Union{Nothing,Function}}(nothing)
const CHAT_CLOSE_HANDLER = Ref{Union{Nothing,Function}}(nothing)
const NEW_BROWSER_HANDLER = Ref{Union{Nothing,Function}}(nothing)
const BROWSER_NAV_HANDLER = Ref{Union{Nothing,Function}}(nothing)
# Separate controller for the AMBIENT MONITOR's model: the monitor is a different
# workload (short, frequent, cheap) and deserves its own choice.
const AGENT_MONITOR_SELECTOR = Ref{Union{Nothing,Function}}(nothing)

"Install the provider of the app's root chat session id."
function addChatSessionProvider!(f::Function)
    CHAT_SESSION_PROVIDER[] = f
    return nothing
end

"Install the chat-interface interaction reporter: `f(session_id, kind, text, idle_s)`."
function addChatInteractionHandler!(f::Function)
    CHAT_INTERACTION_HANDLER[] = f
    return nothing
end

"Install the chat-only responder: `f(wv, text, tab) -> Bool`."
function addChatOnlyHandler!(f::Function)
    AGENT_CHAT_ONLY_HANDLER[] = f
    return nothing
end

"Register `f(parent_id, name, root::Bool)` to create a ChatSession."
function addNewChatHandler!(f::Function)
    NEW_CHAT_HANDLER[] = f
    return nothing
end

"Register `f(session_id) -> closed_ids` to close a ChatSession subtree."
function addChatCloseHandler!(f::Function)
    CHAT_CLOSE_HANDLER[] = f
    return nothing
end

"Register `f(parent_id, url, name)` to create a BrowserSession."
function addNewBrowserHandler!(f::Function)
    NEW_BROWSER_HANDLER[] = f
    return nothing
end

"Register `f(session_id, url)` to sync a user-typed browser URL."
function addBrowserNavigateHandler!(f::Function)
    BROWSER_NAV_HANDLER[] = f
    return nothing
end

"""
    addAgentModelHooks!(provider::Function, selector::Function)

Register the agent's model controller:

* `provider()` → `(; current::String, available::Vector{String})`: the model in
  force and the models the endpoint offers;
* `selector(name::String)` → the same pair AFTER applying the choice.

Both may fail; the UI degrades to "no picker" rather than breaking the window.
"""
function addAgentModelHooks!(provider::Function, selector::Function,
                             monitorSelector::Union{Nothing,Function} = nothing)
    AGENT_MODEL_PROVIDER[] = provider
    AGENT_MODEL_SELECTOR[] = selector
    monitorSelector === nothing || (AGENT_MONITOR_SELECTOR[] = monitorSelector)
    return nothing
end

"""
Current model + available models FOR ONE TAB (empty id = the process default), or
an empty pair when no controller is installed. The provider receives the tab's
terminal id, which is how per-tab models reach the picker.
"""
function agentModelState(session_id::AbstractString = "")
    provider = AGENT_MODEL_PROVIDER[]
    provider === nothing && return (current = "", monitor = "", available = String[])
    try
        state = Base.invokelatest(provider, String(session_id))
        return (current = String(state.current),
                monitor = String(get(state, :monitor, state.current)),
                available = String[String(x) for x in state.available])
    catch err
        @debug "flootyWebUi: agent model provider failed" exception = err
        return (current = "", monitor = "", available = String[])
    end
end

"JSON reply for the picker: the model in force and the ones on offer."
function jsonAgentModelState(state)::String
    options = join(["\"$(escapeJsString(String(name)))\"" for name in state.available], ",")
    monitor = get(state, :monitor, state.current)
    return "{\"success\":true,\"current\":\"" * escapeJsString(String(state.current)) *
           "\",\"monitor\":\"" * escapeJsString(String(monitor)) *
           "\",\"available\":[" * options * "]}"
end

# ---- copy: the app owns it, because WebKit's does nothing here ----------------
# Measured with a real ⌘C key event posted through -[NSApplication sendEvent:]:
# WebKit's editing command collapses the selection and leaves the pasteboard
# UNTOUCHED in this unbundled, in-process WebView — so nothing the user selects,
# in the transcript or in the terminal, ever reached the clipboard. The app
# performs the copy itself: the page reports its live selection (reportSelection
# in index.html) and ⌘C writes that to the pasteboard, swallowing the key so
# WebKit's no-op copy never runs.
const LIVE_SELECTION = Ref("")

"Record the page's live selection (empty string when nothing is selected)."
function setLiveSelection!(text::AbstractString)
    LIVE_SELECTION[] = String(text)
    return nothing
end

const _COPY_HOOK_IMP = Ref{Ptr{Cvoid}}(C_NULL)
const _COPY_HOOK_ORIG = Ref{Ptr{Cvoid}}(C_NULL)

"""
    commandKeyChar(ev) :: String

The character of a ⌘-key key-DOWN carrying no other modifier ("c", "v", …), or ""
for anything else. Matched on the CHARACTER, not the hardware key code:
synthesized events report keyCode 0, and a layout may move a key.
"""
function commandKeyChar(ev::Ptr{Cvoid})::String
    type = ccall(:objc_msgSend, Clong, (Ptr{Cvoid}, Ptr{Cvoid}), ev,
                 ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "type"))
    type == 10 || return ""                        # NSEventTypeKeyDown only
    flags = ccall(:objc_msgSend, Culong, (Ptr{Cvoid}, Ptr{Cvoid}), ev,
                  ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "modifierFlags"))
    (flags & (1 << 20)) != 0 || return ""          # NSEventModifierFlagCommand
    (flags & ((1 << 18) | (1 << 19) | (1 << 17))) == 0 || return ""  # ctrl/opt/shift
    chars = ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), ev,
                  ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                        "charactersIgnoringModifiers"))
    chars == C_NULL && return ""
    cstr = ccall(:objc_msgSend, Cstring, (Ptr{Cvoid}, Ptr{Cvoid}), chars,
                 ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "UTF8String"))
    cstr == C_NULL && return ""
    return lowercase(unsafe_string(cstr))          # a character, never NUL-laden
end

"Clipboard text via `pbpaste` — read on a task, never inside a C callback."
function readPasteboard()::String
    try
        return read(`pbpaste`, String)
    catch err
        @debug "flootyWebUi: readPasteboard failed" exception = err
        return ""
    end
end

"Insert `text` at the caret in the page (assist editor or active terminal)."
function pasteIntoApp!(wv, text::AbstractString)
    isempty(text) && return false
    script = "if (typeof window.flootyPaste === 'function') { window.flootyPaste(\"" *
             escapeJsString(String(text)) * "\"); }"
    try
        Webviews.dispatch(wv) do
            try
                Webviews.eval!(wv, script)
            catch e2
                @debug "flootyWebUi: paste eval failed" exception = e2
            end
        end
        return true
    catch err
        @debug "flootyWebUi: paste dispatch failed" exception = err
        return false
    end
end

"""
    handleEditingKey(ev) :: Bool

Take over the editing shortcuts WebKit cannot perform in this process:

* **⌘C** — the page copies the live selection it reported (its own
  `execCommand('copy')` runs where clipboard writes are granted);
* **⌘V** — the clipboard is read on a TASK (a C callback must not block the run
  loop) and handed to the page, which inserts it at the caret: the assist editor,
  or the active terminal via xterm's own paste.

True when the key was handled and must be swallowed; false forwards it untouched.
"""
function handleEditingKey(ev::Ptr{Cvoid})::Bool
    ev == C_NULL && return false
    key = commandKeyChar(ev)
    isempty(key) && return false
    wv = terminalUiWebview[]
    wv === nothing && return false
    if key == "c"
        # LIVE_SELECTION only covers DOM selections. xterm keeps its own
        # selection and does not fire document.selectionchange, so requiring a
        # non-empty LIVE_SELECTION made terminal copy silently fall through to
        # WebKit's no-op copy. The page resolves DOM first, then terminal.
        try
            Webviews.eval!(wv, "window.flootyCopySelection && window.flootyCopySelection();")
        catch err
            @debug "flootyWebUi: page copy failed" exception = err
            return false
        end
        return true
    elseif key == "v"
        # Paste is ours even with an empty clipboard: WebKit's own paste is a no-op
        # here, so swallowing loses nothing and keeps the two paths symmetric.
        errormonitor(Threads.@spawn pasteIntoApp!(wv, readPasteboard()))
        return true
    end
    return false
end

function _flootyCopyKeyHandler(self::Ptr{Cvoid}, cmd::Ptr{Cvoid}, ev::Ptr{Cvoid})::Cvoid
    handleEditingKey(ev) && return nothing        # ours: WebKit's editing is a no-op here
    orig = _COPY_HOOK_ORIG[]
    orig == C_NULL && return nothing
    ccall(orig, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), self, cmd, ev)
    return nothing
end

"""
    installEditingKeyHandler!()

Swizzle `-[NSApplication sendEvent:]` so ⌘C and ⌘V are performed by the app (see
`handleEditingKey`): in this unbundled, in-process WebView both of WebKit's own
editing commands are inert — copy writes nothing to the pasteboard and paste reads
nothing from it. Every other event is forwarded to the original implementation
untouched. Idempotent.
"""
function installEditingKeyHandler!()
    _COPY_HOOK_IMP[] != C_NULL && return nothing
    cls = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "NSApplication")
    cls == C_NULL && return nothing
    method = ccall(:class_getInstanceMethod, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), cls,
                   ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "sendEvent:"))
    method == C_NULL && return nothing
    orig = ccall(:method_getImplementation, Ptr{Cvoid}, (Ptr{Cvoid},), method)
    imp = @cfunction(_flootyCopyKeyHandler, Cvoid,
                     (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
    ccall(:method_setImplementation, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}), method, imp)
    _COPY_HOOK_IMP[] = imp
    _COPY_HOOK_ORIG[] = orig
    return nothing
end

"""
    runSessionTerminalHooks!(sess; parent_id = "")

Fire all session-terminal hooks best-effort (never breaks creation). `parent_id`
is the session that requested this terminal (a ChatSession or TerminalSession);
empty means the root chat.
"""
function runSessionTerminalHooks!(sess; parent_id::AbstractString = "",
                                kind::Symbol = :terminal,
                                remote::AbstractString = "")
    for f in copy(SESSION_TERMINAL_HOOKS)
        try
            Base.invokelatest(f, sess, String(parent_id), Symbol(kind), String(remote))
        catch err
            @debug "flootyWebUi: session-terminal hook failed" exception = err
        end
    end
    return nothing
end

"""
    pushOwnershipToWindow(wv, owner::AbstractString;
                          reason::AbstractString = "",
                          sessionId::AbstractString = "")

Tell the page who owns input for ONE session (identified by `sessionId`, the
PTY-bound terminal id; `""` = legacy broadcast): `"agent"` (the AI runs the
show — pill shows ai) or `"user"` (pill shows julia). Same dispatch+eval!
push channel as the popups; no-op without a window, failures never propagate.
"""
function pushOwnershipToWindow(wv, owner::AbstractString;
                               reason::AbstractString = "",
                               sessionId::AbstractString = "")
    wv === nothing && return nothing
    @debug string("[ownership] ", owner,
            isempty(reason) ? "" : " (" * reason * ")",
            isempty(sessionId) ? "" : " sid=" * first(sessionId, 8))
    script = "if (typeof window.handleOwnership === 'function') { " *
             "window.handleOwnership(\"" * escapeJsString(String(sessionId)) *
             "\", \"" * escapeJsString(String(owner)) *
             "\", \"" * escapeJsString(String(reason)) * "\"); }"
    try
        Webviews.dispatch(wv) do
            try
                Webviews.eval!(wv, script)
            catch e2
                @debug "flootyWebUi: eval handleOwnership failed" exception = e2
            end
        end
    catch e
        @debug "flootyWebUi: pushOwnershipToWindow failed" exception = e
    end
    return nothing
end

# Per-session UI bridges: terminal id -> listener removal closure, torn down
# by handleTerminalKill when the tab/session dies.
const OWNERSHIP_UI_REMOVERS = Dict{String,Function}()
const OWNERSHIP_UI_REMOVERS_LOCK = ReentrantLock()

# Rendered xterm viewport per terminal id, reported (debounced) by the page.
# agentSnapshot reads this instead of the raw escape-laden PTY tail so the
# model sees the screen the user sees; "" (or a stale report) falls back to
# the raw tail.
const TERMINAL_BUFFER_CACHE = Dict{String,String}()
# Viewport geometry + scroll-over facts that came WITH the last report:
# (rows, columns, scrollback_rows, scrollback_truncated).
const TERMINAL_BUFFER_META = Dict{String,NamedTuple}()
const TERMINAL_BUFFER_AT = Dict{String,Float64}()
const TERMINAL_BUFFER_LOCK = ReentrantLock()
const TERMINAL_BUFFER_TTL_S = 2.0

"""
    attachOwnershipUi!(sess)

Give THIS session its own control-state listener: when it transitions, repaint
the pill (ai/user) and pop an attention notice on takeover. Pushes are TAGGED
with the session's terminal id so the page keeps per-tab ownership state
(switching tabs repaints correctly instead of sticking to whatever was pushed
last). The push is NOT limited to the active session: a background pane's
ownership changed too, and the page stores that state per tab — dropping it
left the tab showing a stale owner. Returns the removal closure.
"""
function attachOwnershipUi!(sess)
    sid = sessionIdFor(sess)
    # The listener is SCOPED to this session: a process-wide one would report
    # every other session's ownership and repaint the wrong pane.
    return addControlStateListener!(sess.control, function (action::Symbol)
        wv = terminalUiWebview[]
        wv === nothing && return
        pushOwnershipToWindow(wv, action == :takeover ? "user" : "agent";
                              reason = string(action), sessionId = sid)
        if action == :takeover &&
           get(ENV, "FLOOTY_AGENT_POPUP_PROBE", "") != "1"
            pushAgentFeedbackToWindow(wv, "You took over",
                "Agent paused — you have the REPL. It waits here; type in the " *
                "assist box to hand it a new instruction.",
                "warn"; sessionId = sid)
        end
    end)
end

# Instructions that arrived BEFORE any terminal was adopted (chat-only boot): the
# page creates the tab when the user submits, but the session binds a moment
# later. A binding must not block the UI thread waiting, so these are delivered in
# `registerAdoptedSession!` as soon as a session exists.
# KEYED BY THE PAGE'S TAB ID: a prompt belongs to the tab that submitted it. The
# queue used to be global, so a prompt that was never consumed (the tab it was
# meant for never got a session — e.g. an unresponsive terminal the user closed)
# was delivered to the NEXT session that registered, i.e. a brand-new tab started
# running the previous tab's prompt.
const PENDING_ASSIST = Dict{String,Vector{String}}()
const PENDING_ASSIST_LOCK = ReentrantLock()

"Queue one instruction for the tab that submitted it."
pushPendingAssist!(tab::AbstractString, text::AbstractString) =
    lock(PENDING_ASSIST_LOCK) do
        push!(get!(Vector{String}, PENDING_ASSIST, String(tab)), String(text))
    end

"Take and clear the instructions waiting for `tab` (only that tab's)."
takePendingAssistPrompts!(tab::AbstractString = "") = lock(PENDING_ASSIST_LOCK) do
    pop!(PENDING_ASSIST, String(tab), String[])
end

"Drop a closed tab's queued instructions — they are not the next tab's business."
dropPendingAssist!(tab::AbstractString) = lock(PENDING_ASSIST_LOCK) do
    pop!(PENDING_ASSIST, String(tab), nothing)
    nothing
end

# Assist prompts (Warp's agent-editor input, Julia side): while the user owns
# the REPL, the page's assist box submits natural-language instructions FOR THE
# AGENT (not the REPL). They queue here until the loop drains them on resume.
const ASSIST_PROMPTS = IdDict{Any,Vector{String}}()
const ASSIST_PROMPTS_LOCK = ReentrantLock()

"Queue one user instruction for the adopted session's agent."
pushAssistPrompt!(sess, text::AbstractString) =
    lock(ASSIST_PROMPTS_LOCK) do
        push!(get!(Vector{String}, ASSIST_PROMPTS, sess), String(text))
    end

"Drain queued instructions (oldest first); empty vector when none."
takeAssistPrompts!(sess) = lock(ASSIST_PROMPTS_LOCK) do
    prompts = get(ASSIST_PROMPTS, sess, nothing)
    prompts === nothing && return String[]
    delete!(ASSIST_PROMPTS, sess)
    return prompts
end

"""
    ownershipSession(sessionId = "") -> Union{ReplSession,Nothing}

The session an ownership action targets: the one with `sessionId` when the
caller names it, else the ACTIVE one.

Ownership is PER SESSION: the pane the user acts in must flip, not whichever
pane happens to be focused. The active session is only the fallback for
callers that cannot name a session (the legacy form).
"""
function ownershipSession(sessionId::AbstractString = "")
    id = strip(String(sessionId))
    if !isempty(id)
        sess = lock(() -> get(OWNERSHIP_SESSIONS, id, nothing),
                    OWNERSHIP_SESSIONS_LOCK)
        sess === nothing || return sess
    end
    return ACTIVE_OWNERSHIP[]
end

"""
    applyOwnershipAction!(action, sessionId = "") :: Bool

Apply an ownership ACTION (:takeover/:resume/:typing) from the UI to a
session. `sessionId` names the session (the pane the user acted in); without
it the ACTIVE session is used (the legacy form). Returns true when a live
session handled it.
"""
function applyOwnershipAction!(action::Symbol, sessionId::AbstractString = "")::Bool
    sess = ownershipSession(sessionId)
    sess === nothing && return false
    control = try
        sess.control
    catch
        return false
    end
    try
        if action == :typing && currentState(control) == :agent
            # User typed while the agent owned the pane — user priority wins:
            # yield immediately; the loop notices and breaks gracefully.
            tagOut(control)
            @debug string("[ownership] user typing detected — agent yields")
            return true
        elseif action == :takeover
            currentState(control) == :user || tagOut(control)
            return true
        elseif action == :resume
            currentState(control) == :agent || tagIn(control; mode = :agent)
            return true
        end
    catch err
        @debug "flootyWebUi: ownership action failed" exception = err
    end
    return false
end

"""
    sessionIdFor(sess) :: String

Reverse lookup: the terminal id `sess` was adopted under, or "" when this
session is not adopted. Lets per-session producers (agent loop, runner,
monitor) tag their output so the PAGE can decide which tab shows it.
"""
function sessionIdFor(sess)::String
    lock(OWNERSHIP_SESSIONS_LOCK)
    try
        return get(OWNERSHIP_IDS, sess, "")
    finally
        unlock(OWNERSHIP_SESSIONS_LOCK)
    end
end

"""
    terminalBufferFor(sess; maxAgeS = TERMINAL_BUFFER_TTL_S) :: String

Newest rendered xterm viewport reported by the page for `sess`'s terminal id, or
`""` when this session is not adopted, no window reported yet, or the report is
older than `maxAgeS` seconds. Callers fall back to `tailText(sess.tail)` on `""`.
"""
function terminalBufferFor(sess; maxAgeS::Real = TERMINAL_BUFFER_TTL_S)::String
    id = sessionIdFor(sess)
    isempty(id) && return ""
    lock(TERMINAL_BUFFER_LOCK)
    try
        at = get(TERMINAL_BUFFER_AT, id, 0.0)
        (time() - at) > Float64(maxAgeS) && return ""
        return get(TERMINAL_BUFFER_CACHE, id, "")
    finally
        unlock(TERMINAL_BUFFER_LOCK)
    end
end


"""
    shadowTextFor(sess) :: String

The terminal screen as the SERVER sees it, for this session's adopted terminal,
or an empty string when the session is not adopted. Unlike terminalBufferFor
this needs no window and no page round-trip: it is the shadow model, fed the
same PTY bytes, so it cannot lag behind the chunk the agent just produced.
"""
function shadowTextFor(sess)::String
    id = sessionIdFor(sess)
    isempty(id) && return ""
    entry = getEntry(id)
    entry === nothing && return ""
    return shadowText(entry)
end

"Row-per-line form of shadowTextFor."
function shadowLinesFor(sess)::Vector{String}
    id = sessionIdFor(sess)
    isempty(id) && return String[]
    entry = getEntry(id)
    entry === nothing && return String[]
    return shadowLines(entry)
end
"""
    flushTerminalOutput!(sess) :: Bool

Push this session's COALESCED output to the page right now. Output to an adopted
tab is batched (16 ms timer / 4 KiB): the model's snapshot can be requested
inside that window, and the page would then serialize the screen from BEFORE the
chunk the agent just produced. Returns true when something was flushed.
"""
function flushTerminalOutput!(sess)::Bool
    id = sessionIdFor(sess)
    isempty(id) && return false
    entry = getEntry(id)
    entry === nothing && return false
    pending = lock(entry.lock) do
        entry.pendingLive
    end
    isempty(pending) && return false
    flushLiveOutput!(entry)
    return true
end

"""
    splitTerminalUi(wv) :: Bool

Split the terminal pane: create a CHILD terminal (its own session) beside the
active tab, sharing the width equally. Returns false when there is no window.
The child is ordinary in every other way (tab, PTY, agent runner) — closing it
with `closeChildTerminalUi` (or its tab ✕) restores the single pane.
"""
function splitTerminalUi(wv; orientation::AbstractString = "vertical")::Bool
    wv === nothing && return false
    try
        Webviews.dispatch(wv) do
            Webviews.eval!(wv,
                "if (typeof window.flootySplitTerminal === 'function') " *
                "window.flootySplitTerminal(\"" *
                escapeJsString(String(orientation)) * "\");")
        end
        return true
    catch err
        @debug "flootyWebUi: splitTerminalUi failed" exception = err
        return false
    end
end

"Close the child panes of the ACTIVE session's tab (keeps the parent)."
function closeChildTerminalUi(wv)::Bool
    wv === nothing && return false
    sid = ACTIVE_OWNERSHIP_ID[]
    try
        Webviews.dispatch(wv) do
            # Target the tab by SESSION id: the page's notion of the active tab
            # can lag behind ownership, and the wrong target closed nothing.
            script = "if (typeof window.flootyCloseChildTerminalFor === 'function') {" *
                     " window.flootyCloseChildTerminalFor(\"" * escapeJsString(sid) * "\"); }" *
                     " else if (typeof window.flootyCloseChildTerminal === 'function') {" *
                     " window.flootyCloseChildTerminal(); }"
            Webviews.eval!(wv, script)
        end
        return true
    catch err
        @debug "flootyWebUi: closeChildTerminalUi failed" exception = err
        return false
    end
end

"""
    pushTerminalBuffer!(sess; waitS = 0.5) :: String

Ask the page for an IMMEDIATE viewport report and wait (bounded) for it to land,
then return the fresh text ("" when no window has this session, or the page did
not answer in time).

The agent snapshots the pane right after one of its commands finished, which is
exactly when the debounced 120 ms push has NOT fired yet — that stale window is
why a model could re-send an edit whose result was already on screen. This
bypasses the debounce for the snapshot path only.
"""
function pushTerminalBuffer!(sess; waitS::Real = 0.5)::String
    id = sessionIdFor(sess)
    isempty(id) && return ""
    wv = activeTerminalUiWebview()
    wv === nothing && return terminalBufferFor(sess)
    # Order matters: the pending chunk must be POSTed to the page first, or the
    # report below serializes the screen from before the agent's last output.
    try
        flushTerminalOutput!(sess)
    catch e
        @debug "flootyWebUi: flush before viewport report failed" exception = e
    end
    before = lock(TERMINAL_BUFFER_LOCK) do
        get(TERMINAL_BUFFER_AT, id, 0.0)
    end
    script = "if (typeof window.flootyReportTerminalBufferNow === 'function') { " *
             "window.flootyReportTerminalBufferNow(\"" * escapeJsString(id) *
             "\").catch(function () {}); }"
    try
        Webviews.dispatch(wv) do
            try
                Webviews.eval!(wv, script)
            catch e
                @debug "flootyWebUi: immediate viewport push failed" exception = e
            end
        end
    catch e
        @debug "flootyWebUi: immediate viewport dispatch failed" exception = e
        return terminalBufferFor(sess)
    end
    deadline = time() + Float64(waitS)
    while time() < deadline
        at = lock(TERMINAL_BUFFER_LOCK) do
            get(TERMINAL_BUFFER_AT, id, 0.0)
        end
        if at > before
            return lock(TERMINAL_BUFFER_LOCK) do
                get(TERMINAL_BUFFER_CACHE, id, "")
            end
        end
        sleep(0.01)
    end
    return terminalBufferFor(sess)
end

"""
    terminalViewportMeta(sess) :: Union{NamedTuple,Nothing}

Geometry + scroll-over facts that came with the last viewport report:
`(rows, columns, scrollback_rows, scrollback_truncated)`. `scrollback_rows` is
how many lines sit ABOVE the visible window (the report carries up to the page's
scrollback window of them; `scrollback_truncated` says older lines were left
out). `nothing` when no report exists.
"""
function terminalViewportMeta(sess)
    id = sessionIdFor(sess)
    isempty(id) && return nothing
    return lock(TERMINAL_BUFFER_LOCK) do
        get(TERMINAL_BUFFER_META, id, nothing)
    end
end

"""
    sessionForTerminalId(id) :: Union{ReplSession,Nothing}

The live session bound to a terminal id, or `nothing` when that tab is gone.
Lets a model controller keep PER-TAB choices: the page sends its tab's id and the
agent package resolves the session it owns.
"""
function sessionForTerminalId(id::AbstractString)
    isempty(id) && return nothing
    return lock(OWNERSHIP_SESSIONS_LOCK) do
        get(OWNERSHIP_SESSIONS, String(id), nothing)
    end
end

# ---- agent presentation events ----------------------------------------------
# Typed replacement for the old `kind::Symbol` ladder: one struct per surface,
# carrying exactly the fields its payload needs. `session_id` tags the event with
# the owning tab's PTY id (concurrent agents never overwrite each other); ""
# broadcasts. See `pushAgentEventToWindow`.
abstract type AbstractAgentEvent end
struct AgentTextEvent     <: AbstractAgentEvent; session_id::String; text::String; end
struct AgentThinkingEvent <: AbstractAgentEvent; session_id::String; text::String; end
struct AgentToolCallEvent <: AbstractAgentEvent
    session_id::String; id::String; tool::String; detail::String
end
struct AgentToolResultEvent <: AbstractAgentEvent
    session_id::String; id::String; ok::Bool; detail::String
end
"A LIVE phase line (not history): replaced in place, cleared when the agent moves on."
struct AgentProgressEvent <: AbstractAgentEvent; session_id::String; text::String; end
"A browser session navigated (the page panel updates its iframe)."
struct AgentBrowserNavigateEvent <: AbstractAgentEvent
    session_id::String
    url::String
end
"An AMBIENT MONITOR insight: chat text with its own role, never agent history.
`monitor` is the INTERFACE whose monitor produced it: :chat or :terminal."
struct AgentInsightEvent <: AbstractAgentEvent
    session_id::String
    text::String
    monitor::Symbol
end
AgentInsightEvent(session_id::AbstractString, text::AbstractString;
                  monitor::Symbol = :chat) =
    AgentInsightEvent(String(session_id), String(text), monitor)
"The monitor's own REASONING, in that INTERFACE's monitor stream only."
struct AgentMonitorThinkingEvent <: AbstractAgentEvent
    session_id::String
    text::String
    monitor::Symbol
end
AgentMonitorThinkingEvent(session_id::AbstractString, text::AbstractString;
                          monitor::Symbol = :chat) =
    AgentMonitorThinkingEvent(String(session_id), String(text), monitor)
struct AgentStatusEvent   <: AbstractAgentEvent
    session_id::String
    text::String
    level::Symbol            # :info | :warn | :error — warn/error ALSO enter the chat
end
AgentStatusEvent(session_id::AbstractString, text::AbstractString) =
    AgentStatusEvent(String(session_id), String(text), :info)
"Confidence of a monitor/decision observation (0..1), shown as a radial gauge."
struct AgentMonitorConfidenceEvent <: AbstractAgentEvent
    session_id::String
    confidence::Float64
    source::Symbol
end

"Estimated context-window use for THIS run's model (status-bar meter)."
struct AgentContextUsageEvent <: AbstractAgentEvent
    session_id::String
    tokens::Int
    limit::Int
end

"""
    pushAgentEventToWindow(wv, ev::AbstractAgentEvent)

One presentation event from a PTY agent turn. The stream is SPLIT — the same split
FlootyCore's protocol makes with `stream_chunk` (content / thinking), `tool_call`
and `tool_executed` — so every surface renders what it OWNS instead of a consumer
guessing a role from prose:

| event                  | `text` carries                           | surface                |
| `AgentTextEvent`       | what the agent SAYS (`done`, insight)    | chat                   |
| `AgentThinkingEvent`   | its reasoning                            | chat (dim italic)      |
| `AgentToolCallEvent`   | the thing it is running; `tool` names it | status strip (running) |
| `AgentToolResultEvent` | scrubbed outcome; `ok` says how it went  | status strip (ok / err)|
| `AgentStatusEvent`     | progress bookkeeping (turns, guards)     | status strip (info)    |

Tool calls and their output belong to the TERMINAL pane — that is where the
command and its results already are — so they are deliberately absent from the
chat. No-op without a window; failures never propagate.
"""
function pushAgentEventToWindow(wv, ev::AbstractAgentEvent)
    wv === nothing && return nothing
    # A browser session has a NATIVE window too (sites that refuse framing).
    ev isa AgentBrowserNavigateEvent &&
        navigateBrowserWindow!(ev.session_id, ev.url)
    payload = agentEventPayload(ev)
    script = "if (typeof window.handleAgentEvent === 'function') { " *
             "window.handleAgentEvent(\"" * escapeJsString(ev.session_id) *
             "\", " * payload * "); }"
    try
        Webviews.dispatch(wv) do
            try
                Webviews.eval!(wv, script)
            catch e2
                @debug "flootyWebUi: eval handleAgentEvent failed" exception = e2
            end
        end
    catch e
        @debug "flootyWebUi: pushAgentEventToWindow failed" exception = e
    end
    return nothing
end

"JSON payload for `ev`, in the exact shape `window.handleAgentEvent` accepts."
agentEventPayload(ev::AgentTextEvent) =
    "{\"kind\":\"text\",\"text\":\"" * escapeJsString(ev.text) * "\"}"
agentEventPayload(ev::AgentThinkingEvent) =
    "{\"kind\":\"thinking\",\"text\":\"" * escapeJsString(ev.text) * "\"}"
agentEventPayload(ev::AgentMonitorThinkingEvent) =
    "{\"kind\":\"monitorThinking\",\"monitor\":\"" *
    escapeJsString(String(ev.monitor)) * "\",\"text\":\"" *
    escapeJsString(ev.text) * "\"}"
agentEventPayload(ev::AgentInsightEvent) =
    "{\"kind\":\"insight\",\"monitor\":\"" *
    escapeJsString(String(ev.monitor)) * "\",\"text\":\"" *
    escapeJsString(ev.text) * "\"}"
agentEventPayload(ev::AgentProgressEvent) =
    "{\"kind\":\"progress\",\"text\":\"" * escapeJsString(ev.text) * "\"}"
agentEventPayload(ev::AgentBrowserNavigateEvent) =
    "{\"kind\":\"browserNavigate\",\"url\":\"" *
    escapeJsString(ev.url) * "\"}"
agentEventPayload(ev::AgentStatusEvent) =
    "{\"kind\":\"status\",\"level\":\"" * escapeJsString(String(ev.level)) *
    "\",\"text\":\"" * escapeJsString(ev.text) * "\"}"
agentEventPayload(ev::AgentContextUsageEvent) =
    "{\"kind\":\"context\",\"tokens\":" * string(ev.tokens) *
    ",\"limit\":" * string(ev.limit) * "}"
agentEventPayload(ev::AgentMonitorConfidenceEvent) =
    "{\"kind\":\"confidence\",\"confidence\":" *
    string(clamp(ev.confidence, 0.0, 1.0) * 100) *
    ",\"source\":\"" * escapeJsString(String(ev.source)) * "\"}"
agentEventPayload(ev::AgentToolCallEvent) =
    "{\"kind\":\"toolcall\",\"text\":\"" * escapeJsString(ev.detail) * "\"" *
    ",\"tool\":\"" * escapeJsString(ev.tool) * "\"" * agentEventIdJson(ev.id) * "}"
agentEventPayload(ev::AgentToolResultEvent) =
    "{\"kind\":\"toolresult\",\"text\":\"" * escapeJsString(ev.detail) * "\"" *
    ",\"ok\":" * (ev.ok ? "true" : "false") * agentEventIdJson(ev.id) * "}"

"The optional `,\"id\":…` fragment both tool payloads carry; omitted when empty."
agentEventIdJson(id::AbstractString) =
    isempty(id) ? "" : ",\"id\":\"" * escapeJsString(id) * "\""

# ---- exit bookkeeping ---------------------------------------------------------

"""
    recordExit!(entry; exitCode=0, sess=nothing)

Record the exit event exactly once and broadcast it to the frontend.

`sess` is the session whose death this exit belongs to (the pump task that
watched it). It matters after a REVIVE: the terminal id then maps to the NEW
session, and the OLD session's death must not close the new one, unlink it
from the app's tree or tell the page the terminal died. The exit is only
broadcast and only closes a session that is STILL mapped to this id.
"""
function recordExit!(entry::TerminalEntry; exitCode::Int = 0, sess = nothing)
    firstExit = lock(entry.lock) do
        if entry.exited
            false
        else
            entry.exited = true
            entry.exitCode = exitCode
            true
        end
    end
    if firstExit
        # The session that died, and whether it is still the one this terminal
        # id maps to. After a revive it is NOT: the id points at the new
        # session, which must survive the old session's death.
        deadSess, stillMapped = lock(OWNERSHIP_SESSIONS_LOCK) do
            mapped = get(OWNERSHIP_SESSIONS, entry.id, nothing)
            s = sess === nothing ? mapped : sess
            (s, s !== nothing && s === mapped)
        end
        if stillMapped
            pushExitToWindow(terminalUiWebview[], entry.id, exitCode)
            # NATURAL EXIT (ssh drop, shell exit) must notify the app exactly
            # like an explicit kill does, so its tree entry is unlinked and its
            # agents learn the terminal died. The kill path fires these hooks
            # itself while the session is still mapped to its id, so look the
            # session up HERE: after a kill the mapping is gone and this is a
            # no-op (no double fire).
            if deadSess !== nothing
                try
                    deadSess.closed = true
                    runSessionTerminalCloseHooks!(deadSess)
                catch err
                    @debug "flootyWebUi: close hooks failed on exit" id exception = err
                end
            end
        else
            # A revived terminal: the session that died is no longer the one
            # behind this id. Its death is not the terminal's death.
            @debug "flootyWebUi: ignoring the exit of a superseded session" id
        end
    end
    return entry
end

"Block until the terminal records its exit event (or timeout). Returns exited flag."
function waitForExit(entry::TerminalEntry; timeoutS::Float64 = 10.0)
    deadline = time() + timeoutS
    while time() < deadline
        done = lock(entry.lock) do
            entry.exited
        end
        done && return true
        sleep(0.02)
    end
    return lock(entry.lock) do
        entry.exited
    end
end

# ---- plain handler layer (no window required) ---------------------------------

"""
    adoptSessionTerminal!(sess; name = "julia")

Make the UI window display an ALREADY-RUNNING ReplSession (the one the agent
drives). Registers a TerminalEntry around `sess.pty` WITHOUT spawning anything,
replays the session's existing tail so history is visible, installs an output
tap on the session so every future chunk (including agent-typed echoes) is
forwarded live, and returns the new terminal id for `terminal_attach`.
Single-consumer invariant preserved: only the session's reader task drains the
PTY channel; this entry never touches it. Killing this terminal from the UI
closes the underlying session PTY — it is the SAME process.

The SAME session may be adopted multiple times (e.g. a user view AND an agent
view): each call returns a distinct id backed by one shared PTY, and the page
fans output to every tab bound to any of those ids.
"""
const ADOPTED_TAPS = Dict{String,Function}()   # id -> () -> removeOutputTap!

# Eval-bookkeeping lines (FLOOTY_EVAL_BEGIN/END, prefixed with a private-use-
# area glyph) are REQUIRED in the PTY stream/tail for result pairing, but they
# are noise — and render as tofu boxes — in the visible pane.
#
# IMPORTANT: these helpers BLANK such lines (keep the row, drop the text).
# Deleting rows outright desyncs the child REPL's cursor accounting from the
# rendered pane — julia positions follow-up prompts/cursor assuming its own
# full-width layout, so the visible cursor drifted rows below the prompt.
# Same row count ⇒ alignment holds; bookkeeping renders as empty lines.
const _SENTINEL_LINE_RE = r"([^\r\n]*FLOOTY_EVAL_[^\r\n]*)(\r?\n?)"
const _PUA_MARKER = string(Char(0xf0a7))

const _TRAILING_EOL_RE = r"[\r\n]+$"

"Blank sentinel lines (keep line terminators) + drop the PUA glyph.
NOTE: `replace` hands the replacer the WHOLE match, so the newline is
re-extracted from it rather than taken from a capture group.
The EOL is OPTIONAL in _SENTINEL_LINE_RE, so a sentinel line can arrive
WITHOUT one (e.g. an excerpt cut right after the tag). Fall back to the
EMPTY STRING explicitly instead of dereferencing `.match` on the fallback
String (that threw FieldError: type String has no field 'match')."
function scrubSentinelLines(text::AbstractString)
    out = replace(String(text), _SENTINEL_LINE_RE =>
        s -> begin
            eolMatch = match(_TRAILING_EOL_RE, s)
            return eolMatch === nothing ? "" : eolMatch.match
        end)
    return replace(out, _PUA_MARKER => "")
end

"""
Stateful line-aware scrubber. Complete lines are blanked when they are
sentinel bookkeeping. The trailing PARTIAL line is normally flushed straight
through — the REPL's prompt and its cursor-positioning escapes live there, and
holding them back leaves the xterm cursor stranded rows below the prompt.
Only a partial line that already contains sentinel machinery (the PUA marker
or the FLOOTY_EVAL_ tag — bytes a human never types) is held back until its
newline arrives, so split sentinel lines are still blanked exactly once.
"""
mutable struct SentinelScrubber
    pending::String
end
SentinelScrubber() = SentinelScrubber("")

const SENTINEL_TAG = "FLOOTY_EVAL_"

"""
True when `tail` (a trailing partial line) might STILL BE the beginning of a
sentinel line: it contains the tag/PUA marker outright, OR its last few
characters are a PREFIX of "FLOOTY_EVAL_" — a chunk boundary can split the
tag anywhere (`…FLOOTY_EV` + next chunk), and a naive contains() check would
flush the orphan half to the display/snapshot where no later scrubber can
remove it. Cost of a false positive is bounded: the fragment is flushed as
soon as the next chunk/keystroke arrives.
"""
function _looksLikeSentinelFragment(tail::AbstractString)
    occursin(_PUA_MARKER, tail) && return true
    occursin(SENTINEL_TAG, tail) && return true
    # Longest suffix of `tail` that is a strict prefix of the tag.
    maxK = min(length(tail), length(SENTINEL_TAG) - 1)
    for k in maxK:-1:1
        startswith(SENTINEL_TAG, last(tail, k)) && return true
    end
    return false
end

"Feed one decoded chunk; returns the display-safe portion."
function scrubForDisplay!(s::SentinelScrubber, chunk::AbstractString)
    s.pending *= chunk
    i = findlast('\n', s.pending)
    out = ""
    if i !== nothing
        out = scrubSentinelLines(s.pending[1:i])
        s.pending = String(SubString(s.pending, nextind(s.pending, i)))
    end
    if _looksLikeSentinelFragment(s.pending)
        return out                      # hold: probably a split sentinel line
    end
    out *= s.pending                    # prompt/cursor escapes render NOW
    s.pending = ""
    return out
end

# ---- live-push coalescing ------------------------------------------------------
# A build emits PTY chunks far faster than the pane needs repainting, and every
# live push costs a main-thread `dispatch` + `eval!`. Batch them instead:
# accumulate display text and flush on whichever comes first — CHUNK_FLUSH_BYTES
# of pending text, or a CHUNK_FLUSH_MS timer armed when the first byte lands.
const CHUNK_FLUSH_MS = 16
const CHUNK_FLUSH_BYTES = 4096
# The live push is RATE LIMITED, not delayed: at most one push per this
# interval, and a push that arrives after the interval goes out IMMEDIATELY
# (a keystroke echo must not wait for a timer — that wait WAS the typing
# latency). A burst waits out the interval and then goes as one batched frame.
const LIVE_FLUSH_MIN_INTERVAL_S = 0.004

"Push whatever live display text `entry` has accumulated (no-op when empty)."
function flushLiveOutput!(entry::TerminalEntry)
    text = lock(entry.lock) do
        isempty(entry.pendingLive) && return ""
        t = entry.pendingLive
        entry.pendingLive = ""
        entry.lastLiveFlush = time()
        t
    end
    isempty(text) && return nothing
    wv = terminalUiWebview[]
    wv === nothing || pushOutputToWindow(wv, entry.id, text)
    return nothing
end

"Arm `entry`'s coalescing timer if none is pending (no-op otherwise)."
function armLiveFlush!(entry::TerminalEntry)
    lock(entry.lock) do
        entry.liveFlushTimer === nothing || return nothing
        # Wait out the REST of the rate-limit interval (not a fixed delay):
        # the point is to batch a burst, not to postpone the next push.
        wait = max(0.001, LIVE_FLUSH_MIN_INTERVAL_S - (time() - entry.lastLiveFlush))
        entry.liveFlushTimer = Timer(wait) do _
            lock(entry.lock) do
                entry.liveFlushTimer = nothing
            end
            flushLiveOutput!(entry)
        end
    end
    return nothing
end

"Disarm `entry`'s coalescing timer (its terminal is going away)."
function stopLiveFlush!(entry::TerminalEntry)
    t = lock(entry.lock) do
        t = entry.liveFlushTimer
        entry.liveFlushTimer = nothing
        t
    end
    t === nothing || close(t)
    return nothing
end

"""
    registerAdoptedSession!(sess; name = "julia", replay = false)
        :: Tuple{String,TerminalEntry}

Shared registration for ANY terminal-backed session — explicit adoptions AND
page-spawned "+ New Terminal" tabs alike: register a TerminalEntry around
`sess.pty` (no pump task — the session reader stays the single consumer),
capture existing output, forward future chunks through a sentinel-scrubbing
tap, and wire per-session ownership plumbing.

`replay = true` additionally pushes the session's CURRENT (scrubbed) tail to
the live window under this id — needed by `handleTerminalCreate`, whose page
already opened the tab BEFORE the binding ran, so startup bytes (banner +
first prompt) that landed in the tail during `startReplSession`'s prompt wait
would otherwise never render (blank pane, cursor stranded on an empty line).
Adoptions leave it false: `adoptSessionTerminal!` seeds its OWN tab's replay
via `attachSessionTerminalToWindow`.
"""
function registerAdoptedSession!(sess;
                                  name::AbstractString = "julia",
                                  replay::Bool = false,
                                  tab::AbstractString = "",
                                  focus::Bool = true)
    id = string(uuid4())
    entry = TerminalEntry(id, sess.pty, String[], ReentrantLock(),
                          false, nothing, nothing, false, String[], "", nothing,
                          0.0,     # lastLiveFlush
                          ShadowTerminal.ShadowTerm(ptyWinsize(sess.pty)[2], ptyWinsize(sess.pty)[1]),
                          Channel{String}(), 0, 0, nothing, ReentrantLock())
    startShadowWorker!(entry)
    lock(terminalRegistryLock) do
        ACTIVE_TERMINALS[id] = entry
    end
    # Replay existing output (banner + everything evaluated so far). Raw goes
    # to `collected`; SCRUBBED goes to the window (no tofu boxes).
    existing = tailText(sess.tail)
    isempty(existing) || lock(entry.lock) do
        push!(entry.collected, existing)
    end
    isempty(existing) || try
        enqueueShadow!(entry, existing)
    catch
    end
    # Forward all future chunks to the window while the entry lives. Explicit
    # closure (not do-block): addOutputTap!'s signature is (sess, tap).
    scrub = SentinelScrubber()
    tap = function (chunk)
        text = decodeChunk(chunk)
        display = scrubForDisplay!(scrub, text)
        # The shadow model is fed on its own task: parsing the stream into it
        # must never throttle the live push to the page.
        enqueueShadow!(entry, text)
        lock(entry.lock) do
            push!(entry.collected, text)      # raw, for headless inspection
            overflow = length(entry.collected) - COLLECTED_MAX_CHUNKS
            overflow > 0 && deleteat!(entry.collected, 1:overflow)
        end
        isempty(display) && return nothing
        # Before the page's begin_live pull, BUFFER display text (a live push
        # here would reach an unloaded page and vanish). Afterwards, COALESCE it
        # and let the flush timer / size threshold push it (see CHUNK_FLUSH_*).
        action = lock(entry.lock) do
            if !entry.livePush
                push!(entry.pendingDisplay, display)
                :buffered
            else
                entry.pendingLive *= display
                # RATE LIMIT, don't delay: a keystroke echo is ONE byte and the
                # user is waiting for it, so it leaves NOW. Only a burst (we
                # just pushed) waits out the interval, and then it goes as one
                # batched frame. terax coalesces at 4 ms for a focused pane.
                now = time()
                if now - entry.lastLiveFlush >= LIVE_FLUSH_MIN_INTERVAL_S
                    :flush
                else
                    :arm
                end
            end
        end
        action === :flush && flushLiveOutput!(entry)
        action === :arm && armLiveFlush!(entry)
        return nothing
    end
    # Explicit ADOPTION (replay=false) must stream LIVE immediately: its tab
    # binds via the page's adoptTerminal, which never calls terminal_begin_live
    # — holding chunks here stranded EVERY later byte (typed echo, agent
    # output) in pendingDisplay forever. Only CREATED tabs (replay=true) rely
    # on the server-side hold for their begin_live bind handshake.
    replay || lock(entry.lock) do
        entry.livePush = true
    end
    addOutputTap!(sess, tap)
    lock(terminalRegistryLock) do
        ADOPTED_TAPS[id] = () -> removeOutputTap!(sess, tap)
    end
    # Created-tab path: the page opened this tab BEFORE terminal_create ran,
    # so everything the child printed while startReplSession waited for the
    # idle prompt (banner + the prompt itself) exists ONLY in the tail. Push
    # it now — after the tap is live — or the pane starts BLANK with the
    # cursor on an empty line below an invisible prompt. (Pushed as display
    # text via the regular output channel; `collected` already holds it.)
    if replay
        wv0 = terminalUiWebview[]
        wv0 === nothing ||
            pushOutputToWindow(wv0, id, scrubSentinelLines(existing))
    end
    # Exit notification duty the old bare-PTY pump performed: watch for PTY
    # closure and fire the page's handleTerminalExit exactly once.
    entry.pumpTask = @async begin
        # This task owns the pty it watches. It must NOT read entry.pty at
        # exit time: a REVIVE swaps entry.pty for a new one, and reaping
        # "the entry's pty" would then kill the REVIVED terminal (SIGKILL ->
        # exit status 137) instead of the dead one.
        ownPty = sess.pty
        while !sess.closed && try
            ownPty.open
        catch
            false
        end
            sleep(0.25)
        end
        try; reapChild!(ownPty; graceS = 0.5); catch; end
        recordExit!(entry; exitCode = something(ownPty.exitStatus, 0), sess = sess)
    end
    # Per-session arbitration: register under the terminal id, make it the
    # active ownership target (tab focus overrides later), and give THIS
    # session its own pill-repainting listener.
    lock(OWNERSHIP_SESSIONS_LOCK) do
        OWNERSHIP_SESSIONS[id] = sess
        OWNERSHIP_IDS[sess] = id
    end
    # A SPLIT CHILD must not steal the active slot: it has no tab, the page never
    # focused it, and the next assist prompt would be routed to a pane the user
    # is not looking at (observed: the agent's `send` output appeared in no
    # visible terminal). Focus follows the page's flootyTabFocused call.
    if focus
        ACTIVE_OWNERSHIP[] = sess
        ACTIVE_OWNERSHIP_ID[] = id
    end
    # Deliver instructions that arrived before THIS TAB's session existed
    # (chat-only boot: the first prompt creates the terminal and binds a moment
    # later). Tab-scoped: another tab's unconsumed prompt must never start here.
    for text in takePendingAssistPrompts!(tab)
        pushAssistPrompt!(sess, text)
    end
    remover = attachOwnershipUi!(sess)
    lock(() -> OWNERSHIP_UI_REMOVERS[id] = remover, OWNERSHIP_UI_REMOVERS_LOCK)
    return id, entry
end

"""Adopt + immediately push the (scrubbed) replay to a live window."""
function adoptSessionTerminal!(sess; name::AbstractString = "julia",
                               parent_session::AbstractString = "")::String
    id, _entry = registerAdoptedSession!(sess; name = String(name))
    # SCRUBBED history goes to the window (no tofu boxes).
    wv0 = terminalUiWebview[]
    if wv0 !== nothing
        hist = tailText(sess.tail)
        raw = length(hist) > 4000 ? last(hist, 4000) : hist
        attachSessionTerminalToWindow(wv0, id, String(name);
                                      replayText = scrubSentinelLines(raw),
                                      parent_session = parent_session)
    end
    @info "flootyWebUi: session terminal adopted" id pid = sess.pty.pid
    return id
end

"""
    handleTerminalCreate([cmd::Cmd]; rows=24, cols=80)

Spawn a FULL ReplSession running `cmd` (not a bare PTY): the new tab gets
the tail/sentinel/control-machine contract, per-session ownership plumbing,
and fires the session-terminal hooks — so Flooty attaches a standing
agent + ambient monitor automatically. Every "+ New Terminal" tab therefore
carries its own commissionable agent. Returns the new terminal id.

The page opens the tab BEFORE this binding runs, and the child's startup
output (banner + first `julia> ` prompt) lands in the tail moments later —
registration replays whatever exists and the output tap + page-side bind
drain deliver the rest. Without that plumbing the tab looks prompt-less.

NON-BLOCKING BY DESIGN: bindings run on the MAIN THREAD, and waiting here
for the child's first prompt froze the whole GUI for a julia boot per tab.
The spawn returns at once; prompt detection + flavor classification finish
in an @async task (input typed ahead of the prompt is buffered by the tty
line discipline, so early input is never lost).
"""
function handleTerminalCreate(cmd::Cmd = defaultTerminalCmd();
                              rows::Int = 24, cols::Int = 80,
                              tab::AbstractString = "",
                              focus::Bool = true,
                              parent_id::AbstractString = "",
                              kind::Symbol = :terminal,
                              remote::AbstractString = "")
    sess = startReplSession(cmd; rows = rows, cols = cols, waitPrompt = false)
    id, _entry = registerAdoptedSession!(sess; replay = true, tab = tab, focus = focus)
    runSessionTerminalHooks!(sess; parent_id = parent_id, kind = kind, remote = remote)
    @info "flootyWebUi: terminal created (session-backed)" id pid = sess.pty.pid
    @async begin
        try
            (_, ok) = waitForText(() -> tailText(sess.tail),
                                  t -> occursin("julia>", t) ||
                                      detectReplFlavor(t) !== nothing ||
                                      oscIntegrated(sess) ||
                                      _tailAtIdlePrompt(t);
                                  timeoutS = PROMPT_TIMEOUT_S,
                                  label = "repl-startup-async")
            ok || @warn "flootyWebUi: terminal boot had no idle prompt " *
                        "within $PROMPT_TIMEOUT_S s" id
            sess.flavor = detectReplFlavor(tailText(sess.tail))
        catch err
            @debug "flootyWebUi: async boot completion failed" id exception = err
        end
    end
    return id
end

"""
    handleTerminalInput(id::String, input)

Write user-attributed bytes into the PTY. `input` is a String or a byte vector.
Returns true when the terminal exists and the write succeeded.

The keystroke also YIELDS that session's agent: the user typed in THIS
terminal, so this session's agent must step aside — and only this one. (The
ownership state is per session; a global "the user is typing" ping would take
the pane away from whichever agent happened to be active.)
"""
function handleTerminalInput(id::String, input::Union{AbstractString, Vector{UInt8}})
    entry = getEntry(id)
    entry === nothing && return false
    bytes = input isa AbstractString ? Vector{UInt8}(codeunits(input)) : input
    ok = try
        writeToPty(entry.pty, bytes; source = :user)
        true
    catch e
        @debug "flootyWebUi: handleTerminalInput failed" id exception = e
        false
    end
    # User priority, per session: the agent of THIS terminal yields.
    if ok
        sess = lock(() -> get(OWNERSHIP_SESSIONS, String(id), nothing),
                    OWNERSHIP_SESSIONS_LOCK)
        if sess !== nothing
            try
                control = sess.control
                if currentState(control) == :agent
                    tagOut(control)
                    @debug string("[ownership] user typed in ", id,
                                  " — that session's agent yields")
                end
            catch e
                @debug "flootyWebUi: user-typing yield failed" id exception = e
            end
        end
    end
    return ok
end

"""
    handleTerminalResize(id::String, rows::Integer, cols::Integer)

Resize the PTY window size (child receives SIGWINCH). Returns true on success.
"""
function handleTerminalResize(id::String, rows::Integer, cols::Integer)
    entry = getEntry(id)
    entry === nothing && return false
    ok = try
        resizePty!(entry.pty, Int(rows), Int(cols))
        shadowResize!(entry, Int(cols), Int(rows))
        true
    catch e
        @debug "flootyWebUi: handleTerminalResize failed" id exception = e
        false
    end
    return ok
end

"""
    handleTerminalKill(id::String; force = false)

Close the PTY (SIGTERM → reap → SIGKILL fallback) and drop the terminal from
the registry. Returns true when a terminal was found and torn down.
"""
function handleTerminalKill(id::String; force = false)
    entry = getEntry(id)
    entry === nothing && return false
    # Adopted sessions: uninstall the UI tap BEFORE closing the PTY so the
    # EOF/close burst is not forwarded to a dead terminal id.
    teardownTap = lock(() -> get(ADOPTED_TAPS, id, nothing), terminalRegistryLock)
    teardownTap !== nothing && try
        teardownTap()
    catch e
        @debug "flootyWebUi: adopted-tap teardown failed" id exception = e
    end
    teardownTap !== nothing && lock(() -> delete!(ADOPTED_TAPS, id),
                                    terminalRegistryLock)
    stopLiveFlush!(entry)
    # The shadow worker keeps the entry alive; close its queue so it exits.
    stopShadowWorker!(entry)
    # Per-session ownership bookkeeping: uninstall this session's UI listener,
    # forget the session, and clear the active pointer if it pointed here.
    teardownUi = lock(() -> get(OWNERSHIP_UI_REMOVERS, id, nothing),
                      OWNERSHIP_UI_REMOVERS_LOCK)
    teardownUi !== nothing && try
        teardownUi()
    catch e
        @debug "flootyWebUi: ownership-ui teardown failed" id exception = e
    end
    teardownUi !== nothing && lock(() -> delete!(OWNERSHIP_UI_REMOVERS, id),
                                   OWNERSHIP_UI_REMOVERS_LOCK)
    # Mark the underlying session closed so ITS standing agent's poll loop
    # exits (runners watch sess.closed) even when killed from the page.
    # Close hooks run while the session is STILL mapped to its id: the app unlinks
    # the session from its tree using sessionIdFor(sess), which is "" afterwards.
    deadSess = lock(() -> get(OWNERSHIP_SESSIONS, id, nothing),
                    OWNERSHIP_SESSIONS_LOCK)
    if deadSess !== nothing
        try
            deadSess.closed = true
        catch
        end
        runSessionTerminalCloseHooks!(deadSess)
    end
    lock(OWNERSHIP_SESSIONS_LOCK) do
        s = pop!(OWNERSHIP_SESSIONS, id, nothing)
        s === nothing || delete!(OWNERSHIP_IDS, s)
        nothing
    end
    if ACTIVE_OWNERSHIP_ID[] == id
        ACTIVE_OWNERSHIP[] = nothing
        ACTIVE_OWNERSHIP_ID[] = ""
    end
    clearSplitChild!(id)
    try
        closePty!(entry.pty)
    catch e
        @debug "flootyWebUi: closePty! failed during kill" id exception = e
    end
    lock(terminalRegistryLock) do
        delete!(ACTIVE_TERMINALS, id)
    end
    recordExit!(entry; exitCode = 0)
    return true
end

"""
    handleTerminalReconnect(id::String; cmd = nothing) :: Bool

Respawn the PTY behind an EXISTING terminal tab, keeping the tab, its id,
its scrollback and its collected output. Used when a session dies for a
reason the user can fix - an ssh link that dropped ("Host is down"), a
shell that exited - so the pane comes back without losing the tab.

`cmd` overrides the command to spawn (e.g. the same ssh host with a
different address); when empty the session original command line is reused.
"""
function handleTerminalReconnect(id::String; cmd = nothing)::Bool
    entry = getEntry(id)
    entry === nothing && return false
    oldSess = lock(() -> get(OWNERSHIP_SESSIONS, id, nothing),
                   OWNERSHIP_SESSIONS_LOCK)
    teardown = lock(() -> get(ADOPTED_TAPS, id, nothing), terminalRegistryLock)
    if teardown !== nothing
        try; teardown(); catch; end
        lock(terminalRegistryLock) do
            delete!(ADOPTED_TAPS, id)
        end
    end
    cmdText = cmd === nothing ? "" : strip(String(cmd))
    if isempty(cmdText) && oldSess !== nothing
        cmdText = oldSess.cmd
    end
    isempty(cmdText) && return false
    rows, cols = ptyWinsize(entry.pty)
    newSess = try
        startReplSession(parseCmdString(cmdText); rows = rows, cols = cols,
                         waitPrompt = false)
    catch e
        @debug "flootyWebUi: reconnect spawn failed" id exception = e
        return false
    end
    lock(entry.lock) do
        entry.pty = newSess.pty
        entry.exited = false
        entry.exitCode = nothing
        entry.livePush = true
    end
    # A revived terminal can die AGAIN: release the once-only close-hook claim
    # so the next exit notifies the app instead of being swallowed.
    lock(CLOSE_HOOKED_LOCK) do
        delete!(CLOSE_HOOKED, id)
    end
    lock(OWNERSHIP_SESSIONS_LOCK) do
        OWNERSHIP_SESSIONS[id] = newSess
        OWNERSHIP_IDS[newSess] = id
    end
    # The OLD session is dead: mark it closed so its ambient monitor unwinds
    # (the monitor loop breaks on state.sess.closed), then let the app
    # reattach machinery to the NEW session. Without this the monitor kept
    # sampling a frozen tail and reported the terminal as exited forever.
    if oldSess !== nothing && oldSess !== newSess
        try; oldSess.closed = true; catch; end
    end
    try
        runSessionReconnectHooks!(oldSess, newSess)
    catch err
        @debug "flootyWebUi: reconnect hooks failed" id exception = err
    end
    scrub = SentinelScrubber()
    tap = function (chunk)
        text = decodeChunk(chunk)
        display = scrubForDisplay!(scrub, text)
        # The shadow model is fed on its own task (see enqueueShadow!): the
        # model parse must not throttle the live push.
        enqueueShadow!(entry, text)
        lock(entry.lock) do
            push!(entry.collected, text)
            overflow = length(entry.collected) - COLLECTED_MAX_CHUNKS
            overflow > 0 && deleteat!(entry.collected, 1:overflow)
            entry.pendingLive *= display
        end
        flushLiveOutput!(entry)
        return nothing
    end
    addOutputTap!(newSess, tap)
    lock(terminalRegistryLock) do
        ADOPTED_TAPS[id] = () -> removeOutputTap!(newSess, tap)
    end
    entry.pumpTask = @async begin
        # Own the pty it watches (a second revive would swap entry.pty again).
        ownPty = newSess.pty
        while !newSess.closed && try
            ownPty.open
        catch
            false
        end
            sleep(0.25)
        end
        try; reapChild!(ownPty; graceS = 0.5); catch; end
        recordExit!(entry; exitCode = something(ownPty.exitStatus, 0), sess = newSess)
    end
    wv = terminalUiWebview[]
    wv === nothing || pushOutputToWindow(wv, id,
        "\r\n\x1b[2m[reconnected]\x1b[0m\r\n")
    return true
end

# ---- binding payload parsing ---------------------------------------------------
#
# Every binding receives ONE JSON object string. Parsing it once with JSON3.jl is
# both cheaper and correct, unlike the hand-rolled scanners this replaces: those
# could not tell `"force": false` from a `false` nested anywhere later in the
# payload, and lost a field whenever an escape or stray space broke the flat
# `"key":` pattern.

const EMPTY_PAYLOAD = NamedTuple()

# JSON3 decodes escaped strings lazily; visit every value before accepting a payload.
function validateJsonPayload(obj::Union{JSON3.Object, JSON3.Array})
    for value in values(obj)
        if value isa Union{JSON3.Object, JSON3.Array}
            validateJsonPayload(value)
        end
    end
    return nothing
end

"""
Binding payload: `args` — the binding's argument tuple, or a bare JSON string —
parsed as one JSON object. A missing or malformed payload yields an empty object,
so every field lookup falls through to its default.
"""
function jsonArgs(args)
    raw = args isa Tuple ? (isempty(args) ? "" : args[1]) : args
    raw isa AbstractString || return EMPTY_PAYLOAD
    s = String(raw)
    isempty(s) && return EMPTY_PAYLOAD
    obj = try
        parsed = JSON3.read(s)
        parsed isa JSON3.Object || return EMPTY_PAYLOAD
        validateJsonPayload(parsed)
        parsed
    catch
        return EMPTY_PAYLOAD
    end
    return obj isa JSON3.Object ? obj : EMPTY_PAYLOAD
end

"String field `key` of a parsed payload (`default` when absent or not a string)."
function jsonStr(obj, key::Symbol, default::AbstractString = "")
    v = get(obj, key, nothing)
    return v isa AbstractString ? String(v) : String(default)
end

"Integer field `key` (default when absent or not a number)."
function jsonInt(obj, key::Symbol, default::Int)
    v = get(obj, key, nothing)
    return v isa Real ? round(Int, v) : default
end

"Boolean field `key` (default when absent or not a boolean)."
function jsonBool(obj, key::Symbol, default::Bool)
    v = get(obj, key, nothing)
    return v isa Bool ? v : default
end

"Shell-like splitting of a command string into a Cmd (quotes respected)."
function parseCmdString(s::AbstractString)
    parts = String[]
    io = IOBuffer()
    inSingle = false
    inDouble = false
    for c in s
        if inSingle
            if c == '\''
                inSingle = false
            else
                print(io, c)
            end
        elseif inDouble
            if c == '"'
                inDouble = false
            else
                print(io, c)
            end
        elseif c == '\''
            inSingle = true
        elseif c == '"'
            inDouble = true
        elseif isspace(c)
            position(io) > 0 && push!(parts, String(take!(io)))
        else
            print(io, c)
        end
    end
    position(io) > 0 && push!(parts, String(take!(io)))
    isempty(parts) && throw(ArgumentError("empty command string"))
    return Cmd(parts)
end

# ---- ObjC fixes for Webviews v1.2.0 latent bugs ----------------------------------
#
# Webviews v1.2.0's apple backend registers a callback that calls a function
# which does not exist anywhere in the package:
#   * WebviewAppDelegate.applicationShouldTerminateAfterLastWindowClosed:
#       -> body calls undefined `_terminate()` (and returns false).
# A Julia UndefVarError thrown across an @cfunction boundary lands INSIDE
# AppKit's run loop (undefined behaviour) — observed as a permanently FROZEN
# window/process. Upstream's own test never closes the window natively, so it
# never trips over this.
#
# The delegate is a custom subclass created during Webviews' __init__ —
# patch THAT class, not NSApplication.
# Additionally, create_window_delegate(w) never assigns its delegate to the
# NSWindow (no setDelegate:), so we install our own — see the "Window-close
# watcher" section below.

"""
Answer NO to applicationShouldTerminateAfterLastWindowClosed.

Returning YES would make AppKit run `_scheduleCheckForTerminateAfterLast-
WindowClosed`, whose timer block CRASHES (SIGBUS, KERN_PROTECTION_FAILURE
inside NSWindowCache code) in an unbundled, non-NSApplicationMain process
like ours — confirmed via macOS crash reports. Returning NO skips that
machinery entirely; loop shutdown is instead driven by Webviews' OWN chain:
on_window_close -> destroy(webview) -> on_window_destroy -> terminate()
([NSApp stop:]) when no windows remain.
"""
_appShouldTerminateAfterLastWindowClosed(_self::Ptr{Cvoid}, _cmd::Ptr{Cvoid},
                                         _notification::Ptr{Cvoid})::UInt8 =
    UInt8(0)  # NO — see docstring above

const _selWindow = let sel = Ref{Ptr{Cvoid}}(C_NULL)
    () -> begin
        if sel[] == C_NULL
            sel[] = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "window")
        end
        sel[]
    end
end

# Keep the IMP alive for the life of the process (non-capturing closure).
const _appTerminatePolicyImp = @cfunction(
    _appShouldTerminateAfterLastWindowClosed, UInt8,
    (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))

const _appTerminatePolicyPatched = Ref(false)

"""
Replace WebviewAppDelegate's broken applicationShouldTerminateAfterLast-
WindowClosed: IMP (once). The class exists right after Webviews' __init__.
"""
function ensureAppTerminatePolicy!()
    _appTerminatePolicyPatched[] && return nothing
    cls = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "WebviewAppDelegate")
    if cls != C_NULL
        sel = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                    "applicationShouldTerminateAfterLastWindowClosed:")
        try
            ccall(:class_replaceMethod, Ptr{Cvoid},
                  (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cstring),
                  cls, sel, _appTerminatePolicyImp, "c@:@")
            _appTerminatePolicyPatched[] = true
        catch err
            @debug "flootyWebUi: could not patch app termination policy" exception = err
        end
    else
        @debug "flootyWebUi: WebviewAppDelegate class not found yet"
    end
    ensureScheduleCheckNoop!()
    return nothing
end

# Event-delivery probe ----------------------------------------------------------------
#
# Swizzles -[NSApplication sendEvent:] (chaining to the ORIGINAL IMP after
# logging) so we can see whether mouse/key events reach this process AT ALL
# while the user interacts. Gated behind ENV["FLOOTY_EVENT_LOG"]=="1".

const _sendEventOrigImp = Ref{Ptr{Cvoid}}(C_NULL)
const _sendEventCount = Ref(0)

function _flootySendEventLog(_self::Ptr{Cvoid}, _cmd::Ptr{Cvoid}, ev::Ptr{Cvoid})
    _sendEventCount[] += 1
    t = try
        ccall(:objc_msgSend, Clong, (Ptr{Cvoid}, Ptr{Cvoid}), ev,
              ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "type"))
    catch
        Clong(-1)
    end
    selActive = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "isActive")
    selKeyWin = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "keyWindow")
    active = ccall(:objc_msgSend, Bool, (Ptr{Cvoid}, Ptr{Cvoid}),
                   _self, selActive)
    kw = ccall(:objc_msgSend, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}),
               _self, selKeyWin)
    @debug string("[events] #", _sendEventCount[], " t=", t,
            " active=", active,
            " kw=", kw != C_NULL ?
                unsafe_string(ccall(:object_getClassName, Cstring,
                                    (Ptr{Cvoid},), kw)) : "none")
    imp = _sendEventOrigImp[]
    imp == C_NULL && return nothing
    ccall(imp, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
          _self, _cmd, ev)
    return nothing
end

"Install the sendEvent: logger once (no-op unless FLOOTY_EVENT_LOG=1)."
function installSendEventProbe!()
    get(ENV, "FLOOTY_EVENT_LOG", "") != "1" && return nothing
    _sendEventOrigImp[] != C_NULL && return nothing
    cls = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "NSApplication")
    cls == C_NULL && return nothing
    sel = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "sendEvent:")
    method = ccall(:class_getInstanceMethod, Ptr{Cvoid},
                   (Ptr{Cvoid}, Ptr{Cvoid}), cls, sel)
    method == C_NULL && return nothing
    orig = ccall(:method_getImplementation, Ptr{Cvoid}, (Ptr{Cvoid},), method)
    newImp = @cfunction(_flootySendEventLog, Cvoid,
                        (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
    ccall(:method_setImplementation, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}),
          method, newImp)
    _sendEventOrigImp[] = orig
    @debug string("[events] sendEvent: logger installed")
    return nothing
end

"No-op replacement for AppKit's crashing private scheduler method."
_flootyScheduleCheckNoop(_self::Ptr{Cvoid}, _cmd::Ptr{Cvoid})::Cvoid =
    return nothing

const _scheduleCheckNoopImp = Ref{Ptr{Cvoid}}(C_NULL)
const _scheduleCheckNoopDone = Ref(false)

"""
Neutralize -[NSApplication _scheduleCheckForTerminateAfterLastWindowClosed]
(once): macOS schedules this private check whenever a window closes in an
unbundled/non-NSApplicationMain process like ours, and its timer block
SIGBUSes writing NSWindowCache state (confirmed via crash reports:
__79-[NSApplication(NSWindowCache) _scheduleCheck...]_block_invoke).
The policy delegate's answer cannot save us — the timer is scheduled
REGARDLESS and crashes when it fires. Replacing the scheduler with a no-op
prevents the timer from existing at all; loop shutdown is handled entirely
by our explicit destroy->terminate chain.
"""
function ensureScheduleCheckNoop!()
    _scheduleCheckNoopDone[] && return nothing
    cls = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "NSApplication")
    cls == C_NULL && return nothing
    sel = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                "_scheduleCheckForTerminateAfterLastWindowClosed")
    imp = @cfunction(_flootyScheduleCheckNoop, Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}))
    try
        ccall(:class_replaceMethod, Ptr{Cvoid},
              (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cstring),
              cls, sel, imp, "v@:@")
        _scheduleCheckNoopImp[] = imp
        _scheduleCheckNoopDone[] = true
    catch err
        @debug "flootyWebUi: could not no-op schedule-check scheduler" exception = err
    end
    return nothing
end

# Window-close watcher -----------------------------------------------------------
#
# ANOTHER latent Webviews v1.2.0 bug: create_window_delegate(w) builds a
# "WebviewWindowDelegate" instance but NEVER assigns it to the NSWindow
# (no setDelegate: anywhere) — so windowWillClose: NEVER fires, neither for
# upstream's broken implementation nor for anything else. Native close (red
# traffic light / [NSWindow close]) therefore cannot end the event loop.
#
# Fix: register OUR OWN minimal NSWindowDelegate subclass whose
# windowWillClose: posts the wake-event + [NSApp stop:] pair, and INSTALL it
# on the window after webview creation. We deliberately do NOT destroy the
# webview from inside the callback — API.run's finally-block does that after
# run() returns; destroying mid-close would re-enter [NSWindow close].

"windowWillClose: body — end the NSApplication loop with a wake event."
function _flootyWindowWillClose(_self::Ptr{Cvoid}, _cmd::Ptr{Cvoid},
                                _notification::Ptr{Cvoid})
    try
        stopSharedAppWithWake!()
    catch err
        # NEVER let a Julia throw escape across the ObjC frame — it wedges
        # AppKit's run loop (observed as a fully stuck window).
        @debug "flootyWebUi: windowWillClose handler failed" exception = err
    end
    return nothing
end

const _windowWillCloseCFun = Ref{Ptr{Cvoid}}(C_NULL)
const _flootyWindowDelegateClass = Ref{Ptr{Cvoid}}(C_NULL)
const _flootyWindowDelegateInstance = Ref{Ptr{Cvoid}}(C_NULL)  # GC anchor

"Register (once) the FlootyWindowDelegate ObjC class with windowWillClose:."
function ensureFlootyWindowDelegateClass!()
    _flootyWindowDelegateClass[] != C_NULL && return true
    superCls = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "NSResponder")
    superCls == C_NULL && return false
    cls = ccall(:objc_allocateClassPair, Ptr{Cvoid},
                (Ptr{Cvoid}, Cstring, Csize_t),
                superCls, "FlootyWindowDelegate", 0)
    cls == C_NULL && return false
    if _windowWillCloseCFun[] == C_NULL
        _windowWillCloseCFun[] = @cfunction(_flootyWindowWillClose, Cvoid,
            (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}))
    end
    sel = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "windowWillClose:")
    ok = ccall(:class_addMethod, Bool,
               (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Cstring),
               cls, sel, _windowWillCloseCFun[], "v@:@")
    ccall(:objc_registerClassPair, Cvoid, (Ptr{Cvoid},), cls)
    _flootyWindowDelegateClass[] = cls
    return ok
end

"""
Install our FlootyWindowDelegate instance on the webview's NSWindow so native
window closes reach windowWillClose: and can end run(wv)'s loop. Call once
right after the Webview is constructed (see launchTerminalUI).
"""
function installFlootyWindowDelegate!(wv)
    win = try
        Webviews.window_handle(wv)::Ptr{Cvoid}
    catch
        return nothing
    end
    win == C_NULL && return nothing
    ensureFlootyWindowDelegateClass!() || return nothing
    if _flootyWindowDelegateInstance[] == C_NULL
        cls = _flootyWindowDelegateClass[]
        selNew = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "new")
        inst = ccall(:objc_msgSend, Ptr{Cvoid},
                     (Ptr{Cvoid}, Ptr{Cvoid}), cls, selNew)
        _flootyWindowDelegateInstance[] = inst  # anchor forever
    end
    inst = _flootyWindowDelegateInstance[]
    inst == C_NULL && return nothing
    selSetDel = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "setDelegate:")
    ccall(:objc_msgSend, Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), win, selSetDel, inst)
    return nothing
end

"""
    openTerminalWindow(url; width=1100, height=750, title="Flooty Terminal")

Open a NATIVE window (a WKWebView on macOS) showing `url`.

The window is the app's surface: the page inside it may be served by any HTTP
server (this package's own asset server, or another app's). The focus
plumbing Webviews gets wrong — it activates the app before `[NSApp run]`
exists and never makes the web content first responder, so keystrokes go
nowhere — is fixed by `runTerminalWindow`, which re-activates the app and
makes the WKWebView first responder once the loop is running.

Returns the window handle (a `Webviews.Webview`).
"""
function openTerminalWindow(url::AbstractString; width::Int=1100, height::Int=750,
                            title::AbstractString="Flooty Terminal")
    # Webviews v1.2.0 ships a broken applicationShouldTerminateAfterLast-
    # WindowClosed (calls an undefined `_terminate`); patch it before the
    # window exists.
    ensureAppTerminatePolicy!()
    wv = Webviews.Webview((width, height); title=title, debug=false,
                          enable_webio=false, auto_terminate=true)
    # Our own delegate: windowWillClose: ends the event loop cleanly.
    installFlootyWindowDelegate!(wv)
    Webviews.navigate!(wv, String(url))
    return wv
end

"""
    runTerminalWindow(wv; focusDelays=(0.05, 1.5))

Run the native event loop for a window opened with `openTerminalWindow` and
return when the window closes.

The focus activation passes are scheduled AFTER the loop starts (Webviews
activates the app too early to have any effect), so the web content really
gets the keyboard.
"""
function runTerminalWindow(wv; focusDelays=(0.05, 1.5))
    for d in focusDelays
        scheduleFocusActivation!(wv; delayS=d)
    end
    try
        Webviews.run(wv)
    catch err
        @debug "flootyWebUi: webview event loop ended" exception = err
    end
    return nothing
end

# Focus plumbing -------------------------------------------------------------------
#
# Webviews' Apple backend activates the app ONCE at window construction
# (activateIgnoringOtherApps:) — before [NSApp run] exists, which usually does
# nothing — and never calls makeFirstResponder: on its WKWebView. Result: the
# window shows and renders fine, but keystrokes go nowhere.

"Re-activate the app + make the web content first responder (main thread)."
function ensureWindowFocus!(wv)
    try
        appCls = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "NSApplication")
        selShared = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                          "sharedApplication")
        app = ccall(:objc_msgSend, Ptr{Cvoid},
                    (Ptr{Cvoid}, Ptr{Cvoid}), appCls, selShared)
        app == C_NULL && return nothing
        selActivate = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                            "activateIgnoringOtherApps:")
        ccall(:objc_msgSend, Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Bool), app, selActivate, true)
        win = try
            Webviews.window_handle(wv)::Ptr{Cvoid}
        catch
            return nothing
        end
        win == C_NULL && return nothing
        selMakeKey = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                           "makeKeyAndOrderFront:")
        ccall(:objc_msgSend, Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), win, selMakeKey, C_NULL)
        # Locate the WKWebView: the Apple backend sets it DIRECTLY as the
        # window's contentView (so contentView.class == WKWebView); fall back
        # to scanning subviews in case that ever changes.
        content = ccall(:objc_msgSend, Ptr{Cvoid},
                        (Ptr{Cvoid}, Ptr{Cvoid}), win,
                        ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                              "contentView"))
        content == C_NULL && return nothing
        clsWk = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "WKWebView")
        clsWk == C_NULL && return nothing
        selClass = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "class")
        wkView = C_NULL
        if ccall(:objc_msgSend, Ptr{Cvoid},
                 (Ptr{Cvoid}, Ptr{Cvoid}), content, selClass) == clsWk
            wkView = content
        else
            subviews = ccall(:objc_msgSend, Ptr{Cvoid},
                             (Ptr{Cvoid}, Ptr{Cvoid}), content,
                             ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                                   "subviews"))
            subviews == C_NULL && return nothing
            selCount = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "count")
            selObjAt = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                             "objectAtIndex:")
            selIsKind = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                              "isKindOfClass:")
            n = ccall(:objc_msgSend, Clong,
                      (Ptr{Cvoid}, Ptr{Cvoid}), subviews, selCount)
            for i in 0:(n - 1)
                v = ccall(:objc_msgSend, Ptr{Cvoid},
                          (Ptr{Cvoid}, Ptr{Cvoid}, Clong),
                          subviews, selObjAt, i)
                (v != C_NULL && ccall(:objc_msgSend, Bool,
                                      (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                                      v, selIsKind, clsWk)) && (wkView = v;
                                                                break)
            end
        end
        wkView == C_NULL && return nothing
        # makeFirstResponder: is an NSWindow method (NOT NSApplication).
        ccall(:objc_msgSend, Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
              win,
              ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                    "makeFirstResponder:"),
              wkView)
# Diagnostics: is the app active, is our window key, who is first resp?
        selActive = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "isActive")
        selKeyWin = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                          "isKeyWindow")
        selFr = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                      "firstResponder")
        selIgnores = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                           "ignoresMouseEvents")
        selCanBecome = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                             "canBecomeKeyWindow")
        selWindows = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "windows")
        selCountW = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "count")
        appActive = ccall(:objc_msgSend, Bool,
                          (Ptr{Cvoid}, Ptr{Cvoid}), app, selActive)
        winKey = ccall(:objc_msgSend, Bool,
                       (Ptr{Cvoid}, Ptr{Cvoid}), win, selKeyWin)
        winIgnores = ccall(:objc_msgSend, Bool,
                           (Ptr{Cvoid}, Ptr{Cvoid}), win, selIgnores)
        winCanBecomeKey = ccall(:objc_msgSend, Bool,
                                (Ptr{Cvoid}, Ptr{Cvoid}), win, selCanBecome)
        fr = ccall(:objc_msgSend, Ptr{Cvoid},
                   (Ptr{Cvoid}, Ptr{Cvoid}), win, selFr)
        frName = fr == C_NULL ? "null" :
                 unsafe_string(ccall(:object_getClassName, Cstring,
                                     (Ptr{Cvoid},), fr))
        allWins = ccall(:objc_msgSend, Ptr{Cvoid},
                        (Ptr{Cvoid}, Ptr{Cvoid}), app, selWindows)
        nWins = allWins == C_NULL ? Clong(-1) :
                ccall(:objc_msgSend, Clong, (Ptr{Cvoid}, Ptr{Cvoid}),
                      allWins, selCountW)
        @debug string("[focus] appActive=", appActive, " keyWindow=", winKey,
                " firstResponder=", frName,
                " ignoresMouse=", winIgnores,
                " canBecomeKey=", winCanBecomeKey,
                " nWindows=", nWins)
    catch err
        @debug "flootyWebUi: could not force window focus" exception = err
    end
    return nothing
end

"""
Run `ensureWindowFocus!` on the main thread as soon as run(wv) has started,
via a one-shot NSTimer scheduled INSIDE the loop (no polling: fires exactly
once). Timers need a live run loop, which is precisely the point — activation
before this moment reliably fails for unbundled processes.
"""
function scheduleFocusActivation!(wv; delayS::Real = 0.05)
    block = function ()
        ensureWindowFocus!(wv)
        nothing
    end
    try
        # Webviews.set_timeout(f, w, interval; repeat) — returns a dispatch ptr.
        Webviews.set_timeout(block, wv, delayS; repeat = false)
    catch err
        @debug "flootyWebUi: scheduling focus activation failed" exception = err
        return nothing
    end
end

# Layout-compatible with AppKit's NSPoint/CGPoint (CGFloat == Cdouble).
struct _NSPoint
    x::Cdouble
    y::Cdouble
end

const _NSEVENT_APPLICATION_DEFINED = UInt(15)

"""
Post [NSApp stop:] WITH a preceding synthetic wake event.

[NSApp stop:] only sets a flag that AppKit consults when it DEQUEUES an
actual NSEvent. With no event pending, -nextEventMatchingMask keeps blocking
and the flag is never examined — the loop hangs forever (observed). Posting
an ApplicationDefined event guarantees the next event fetch wakes up,
dequeues it, notices the stop flag, and unwinds run(wv).
"""
function stopSharedAppWithWake!()
    appCls = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "NSApplication")
    appCls == C_NULL && return false
    selShared = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                      "sharedApplication")
    app = ccall(:objc_msgSend, Ptr{Cvoid},
                (Ptr{Cvoid}, Ptr{Cvoid}), appCls, selShared)
    app == C_NULL && return false
    evCls = ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), "NSEvent")
    evCls == C_NULL && return false
    selOther = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                     "otherEventWithType:location:modifierFlags:timestamp:" *
                     "windowNumber:context:subtype:data1:data2:")
    ev = ccall(:objc_msgSend, Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}, UInt, _NSPoint, Culong, Cdouble,
                Clong, Ptr{Cvoid}, Int16, Clong, Clong),
               evCls, selOther,
               _NSEVENT_APPLICATION_DEFINED,
               _NSPoint(0.0, 0.0),
               Culong(0),          # modifierFlags
               Cdouble(0.0),       # timestamp
               Clong(0),           # windowNumber
               C_NULL,             # context (unused for app-defined)
               Int16(0),           # subtype
               Clong(0), Clong(0)) # data1, data2
    if ev != C_NULL
        selPost = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                        "postEvent:atStart:")
        ccall(:objc_msgSend, Cvoid,
              (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Bool),
              app, selPost, ev, false)
    end
    selStop = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,), "stop:")
    ccall(:objc_msgSend, Cvoid,
          (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), app, selStop, C_NULL)
    return true
end

"""
requestStopEventLoop!(wv) — end run(wv)'s event loop from anywhere.
Marshals stopSharedAppWithWake! onto the main thread via Webviews.dispatch.
Safe from bindings (already main-thread), REPL tasks, and probes.
"""
function requestStopEventLoop!(wv)
    stopBlock = function ()
        try
            stopSharedAppWithWake!()
        catch err
            @debug "flootyWebUi: requestStopEventLoop! failed" exception = err
        end
        nothing
    end
    try
        # Bindings already run on the main thread; dispatch is still safe there
        # and also covers calls from arbitrary tasks (REPL, probes).
        Webviews.dispatch(wv, stopBlock)
    catch err
        @debug "flootyWebUi: dispatching stop failed" exception = err
    end
    return nothing
end

# ---- Webviews transport layer ---------------------------------------------------

"""
Register the frontend-callable bindings on the webview. Each bound name becomes
a global async JS function taking ONE JSON-string argument and resolving with a
JSON-string result ({success, ...}) — mirroring the old webui.call contract so
the frontend diff stays minimal. Handlers delegate to the plain functions above.
"""
function registerTerminalBindings!(wv)
    bindSlashCommands!(wv)
    Webviews.bind(wv, "flootyThemeGet") do _
        try
            JSON3.write(merge(Dict("success" => true), ThemeManager.themeState()))
        catch err
            errorJson(sprint(showerror, err))
        end
    end
    Webviews.bind(wv, "flootyThemeSet") do args
        try
            name = jsonStr(jsonArgs(args), :name)
            JSON3.write(merge(Dict("success" => true), ThemeManager.selectTheme!(name)))
        catch err
            errorJson(sprint(showerror, err))
        end
    end

    Webviews.bind(wv, "terminal_create") do args
        resp = try
            payload = jsonArgs(args)
            cmdText = jsonStr(payload, :cmd)
            cmd = isempty(strip(cmdText)) ?
                  terminalUiConfig[].cmd : parseCmdString(cmdText)
            rows = jsonInt(payload, :rows, 24)
            cols = jsonInt(payload, :cols, 80)
            tab = jsonStr(payload, :tab)
            focus = jsonBool(payload, :focus, true)
            parent_id = jsonStr(payload, :parent_session)
            kind = Symbol(jsonStr(payload, :kind) == "ssh" ? :ssh : :terminal)
            remote = jsonStr(payload, :remote)
            jsonCreated(handleTerminalCreate(cmd; rows = rows, cols = cols,
                                             tab = tab, focus = focus,
                                             parent_id = parent_id,
                                             kind = kind, remote = remote))
        catch err
            @error "flootyWebUi: terminal_create failed" exception = (err, catch_backtrace())
            errorJson(sprint(showerror, err))
        end
        resp
    end

    # Bind-handshake PULL: the page calls this once its tab/xterm exists and
    # is bound to `id`. Everything display-safe buffered since creation (boot
    # banner, first prompt) is drained IN ORDER into the reply and live push
    # flips on atomically — no chunk can be lost between buffer and stream,
    # no duplicate can be produced (buffer drains exactly once).
    Webviews.bind(wv, "terminal_begin_live") do args
        resp = try
            payload = jsonArgs(args)
            id = jsonStr(payload, :terminal_id)
            entry = isempty(id) ? nothing : getEntry(id)
            entry === nothing && return jsonCreated(id)   # idempotent-ish ack
            flushLiveOutput!(entry)     # nothing is stranded if it already flipped
            buffered = lock(entry.lock) do
                entry.livePush = true
                chunks = entry.pendingDisplay
                entry.pendingDisplay = String[]
                join(chunks)
            end
            "{\"success\":true,\"replay\":\"" *
                escapeJsString(buffered) * "\"}"
        catch err
            @error "flootyWebUi: terminal_begin_live failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    Webviews.bind(wv, "terminal_reconnect") do args
        resp = try
            payload = jsonArgs(args)
            id = jsonStr(payload, :terminal_id)
            cmd = jsonStr(payload, :cmd)
            ok = handleTerminalReconnect(id; cmd = isempty(cmd) ? nothing : cmd)
            ok ? "{\"success\":true}" : errorJson("reconnect failed")
        catch err
            @error "flootyWebUi: terminal_reconnect failed" exception = (err, catch_backtrace())
            errorJson(sprint(showerror, err))
        end
        resp
    end

    Webviews.bind(wv, "terminal_kill") do args
        resp = try
            payload = jsonArgs(args)
            id = jsonStr(payload, :terminal_id)
            force = jsonBool(payload, :force, false)
            tab = jsonStr(payload, :tab)
            isempty(tab) || dropPendingAssist!(tab)
            # The dead session's queued instructions die with it: a new tab must
            # never drain what the closed one never consumed.
            if !isempty(id)
                sess = lock(() -> get(OWNERSHIP_SESSIONS, id, nothing),
                            OWNERSHIP_SESSIONS_LOCK)
                sess === nothing || lock(ASSIST_PROMPTS_LOCK) do
                    pop!(ASSIST_PROMPTS, sess, nothing)
                    nothing
                end
            end
            (!isempty(id) && handleTerminalKill(id; force = force)) ?
                jsonOk() : errorJson("terminal not found")
        catch err
            @error "flootyWebUi: terminal_kill failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    # Attach an EXISTING backend terminal (session adoption): payload is
    # {"terminal_id": "<uuid from adoptSessionTerminal!>", "name": "..."}.
    # No process is spawned; the page just opens a tab bound to that id so
    # pushed chunks land in it.
    Webviews.bind(wv, "terminal_attach") do args
        resp = try
            payload = jsonArgs(args)
            id = jsonStr(payload, :terminal_id)
            name = jsonStr(payload, :name, "julia")
            entry = isempty(id) ? nothing : getEntry(id)
            entry === nothing && return errorJson("terminal not found")
            jsonOkWith("attachedName", String(name))
        catch err
            @error "flootyWebUi: terminal_attach failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    # Custom glass close button: close the NATIVE window (same path as the
    # red traffic light). Our patched windowWillClose: posts [NSApp stop:],
    # API.run unwinds and destroys the webview in its finally-block — never
    # destroy from inside this callback (re-enters [NSWindow close]).
    # [NSApp stop:] alone is unreliable when posted outside an event pass;
    # going through [NSWindow close] (+ API.close's resize-nudge) matches
    # upstream's own test teardown and always ends the loop.
    # Name note: "flooty" prefix — a binding literally named "closeWindow"
    # collides with index.html's own function declaration, which parses after
    # bind()'s document-start shim and silently shadows it (flaky autoclose).
    Webviews.bind(wv, "flootyCloseWindow") do _
        resp = jsonOk()
        try
            Webviews.close(wv)
        catch err
            @debug "flootyWebUi: closeWindow native close failed" exception = err
            try
                requestStopEventLoop!(wv)
            catch err2
                @debug "flootyWebUi: closeWindow fallback stop failed" exception = err2
            end
        end
        resp
    end

    # Native miniaturize of the backing NSWindow (main-thread op).
    Webviews.bind(wv, "flootyMinimizeWindow") do _
        try
            Webviews.dispatch(wv) do
                try
                    wh = Webviews.window_handle(wv)::Ptr{Cvoid}
                    wh == C_NULL && return
                    selMin = ccall(:sel_registerName, Ptr{Cvoid}, (Cstring,),
                                   "miniaturize:")
                    ccall(:objc_msgSend, Ptr{Cvoid},
                          (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}),
                          wh, selMin, C_NULL)
                catch err
                    @debug "flootyWebUi: native minimize failed" exception = err
                end
                nothing
            end
        catch err
            @debug "flootyWebUi: minimizeWindow failed" exception = err
        end
        jsonOk()
    end

    # Frontend debug channel: mirrors page-side diagnostics to stdout.
    Webviews.bind(wv, "flootyDebug") do args
        msg = args isa Tuple && !isempty(args) ? String(args[1]) : ""
        @debug string("[ui-debug] ", msg)
        return jsonOk()
    end

    # GPU renderer report: the page pushes which xterm renderer is painting
    # (WebGL2 canvas vs DOM fallback). Stored so the app and the probes can
    # assert the GPU path is actually live.
    Webviews.bind(wv, "flootyGpuReport") do args
        payload = jsonArgs(args)
        GPU_RENDERER_REPORT[] = payload
        @debug string("[ui-gpu] ", JSON3.write(payload))
        return jsonOk()
    end

    # Agent-feedback popup acknowledgment: the page calls this every time a
    # pushed popup actually rendered. Stdout line doubles as a deterministic
    # assertion target for runAgentUiProbe.jl.
    Webviews.bind(wv, "flootyAgentPopupShown") do args
        msg = args isa Tuple && !isempty(args) ? String(args[1]) : ""
        @debug string("[agent-popup-shown] ", isempty(msg) ? "(no payload)" : msg)
        return jsonOk()
    end

    Webviews.bind(wv, "flootyChannelClosed") do args
        # The page's channel socket died (closed, or never opened): unbind it so
        # the PTY bytes go back over the eval path instead of into the void.
        payload = jsonArgs(args)
        id = jsonStr(payload, :terminal_id)
        isempty(id) || unbind_channel!(id)
        jsonOk()
    end

    # Ownership arbitration (Warp TuiInputTarget-style, made visible):
    #   flootyTakeOver     — user pressed "Take Over": agent yields to :user.
    #   flootyAssistResume — pill/assist pressed: hand the pane back to :agent.
    #   flootyUserTyping   — raw keydown reached the page while assist owned
    #                        input; user priority wins, agent yields at once.
    # PER SESSION: the payload names the session (the pane's juliaId), so the
    # action hits THAT session's agent — not whichever session happens to be
    # the active one. Without the id the active session is used (legacy).
    Webviews.bind(wv, "flootyTakeOver") do args
        applyOwnershipAction!(:takeover, jsonStr(jsonArgs(args), :session_id))
        return jsonOk()
    end
    Webviews.bind(wv, "flootyAssistResume") do args
        ok = applyOwnershipAction!(:resume, jsonStr(jsonArgs(args), :session_id))
        ok ? jsonOk() : errorJson("no adopted agent session")
    end
    Webviews.bind(wv, "flootyUserTyping") do args
        applyOwnershipAction!(:typing, jsonStr(jsonArgs(args), :session_id))
        return jsonOk()
    end
    # Tab focus routing: the page reports the PTY-backed id behind whichever
    # tab became active, so per-session ownership actions land on ITS session.
    # We ALSO re-report THAT session's CURRENT owner (tagged push) so the pill
    # repaints correctly on every switch instead of sticking to the last
    # transition of some other tab.
    # CHAT INTERFACE traces: the page reports submit / typing / pause events for the
    # chat window. Only pauses are model-worthy (the app gates them by cooldown).
    Webviews.bind(wv, "flootyChatInteraction") do args
        resp = try
            payload = jsonArgs(args)
            sid = jsonStr(payload, :session_id)
            kind = jsonStr(payload, :kind)
            text = jsonStr(payload, :text)
            isSlashCommandDraft(text) && return jsonOk()
            idle = Float64(jsonInt(payload, :idle_ms, 0)) / 1000.0
            handler = CHAT_INTERACTION_HANDLER[]
            if isempty(sid) || isempty(kind)
                errorJson("missing session_id or kind")
            elseif handler === nothing
                errorJson("no chat interaction handler")
            else
                Base.invokelatest(handler, sid, kind, text, idle)
                jsonOk()
            end
        catch err
            @error "flootyWebUi: flootyChatInteraction failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    Webviews.bind(wv, "flootyTabFocused") do args
        payload = jsonArgs(args)
        id = jsonStr(payload, :terminal_id)
        isempty(id) && return jsonOk()
        sess = setActiveOwnershipSession(id)
        sess === nothing && return jsonOk()
        try
            owner = currentState(sess.control) == :user ? "user" : "agent"
            pushOwnershipToWindow(wv, owner; reason = "focus", sessionId = id)
        catch err
            @debug "flootyWebUi: focus ownership report failed" exception = err
        end
        return jsonOk()
    end

    # Warp-style assist input: submitting the assist box is how the user TALKS
    # TO THE AGENT while owning the pane. The text queues as intent, ownership
    # flips back to :agent (submitting IS handing control over), and a paused
    # loop wakes up to read it.
    Webviews.bind(wv, "flootyAssistSubmit") do args
        payload = jsonArgs(args)
        text = jsonStr(payload, :text)
        isempty(strip(text)) && return errorJson("empty assist prompt")
        command = dispatchAssistSlash(payload)
        get(command, :handled, false) && return JSON3.write(merge((success = true,), command))
        # The focused pane's SESSION (the page names it); without a name the
        # ACTIVE session is used (legacy).
        sess = ownershipSession(jsonStr(payload, :terminal_session_id))
        # A CHAT session prompt carries its session id explicitly: it must reach
        # the chat-only responder even when a terminal tab exists elsewhere.
        chat_sid = jsonStr(payload, :session_id)
        if !isempty(chat_sid)
            handler = AGENT_CHAT_ONLY_HANDLER[]
            if handler !== nothing
                try
                    Base.invokelatest(handler, wv, String(text),
                                      jsonStr(payload, :tab), String(chat_sid))
                    return jsonOk()
                catch err
                    @error "flootyWebUi: chat session handler failed" exception = err
                end
            end
        end
        if sess === nothing
            # CHAT-ONLY: no terminal exists. Let the app answer conversationally
            # (it owns the model and the thread); only when no handler is
            # installed (a bare UI process) queue for the next adopted session.
            handler = AGENT_CHAT_ONLY_HANDLER[]
            if handler !== nothing
                try
                    Base.invokelatest(handler, wv, String(text),
                                      jsonStr(payload, :tab), String(chat_sid))
                    return jsonOk()
                catch err
                    @error "flootyWebUi: chat-only handler failed" exception = err
                end
            end
            pushPendingAssist!(jsonStr(payload, :tab), text)
            @debug string("[assist] queued for the next session: ", first(strip(text), 80))
            return jsonOk()
        end
        # Loud hint when this process cannot EVER consume the instruction:
        # standing agents attach through the session-terminal hook, which is
        # registered by Flooty's __init__. Without it the prompt sits
        # queued forever and the model is never called.
        if isempty(SESSION_TERMINAL_HOOKS)
            @warn "flootyWebUi: no standing agents attached — this process " *
                  "never loaded Flooty, so assist prompts queue " *
                  "unconsumed. Launch via demoAgentUi.jl or agentUi.jl instead."
        end
        pushAssistPrompt!(sess, text)
        @debug string("[assist] prompt queued: ", first(strip(text), 120))
        # Submitting IS handing the pane back to the agent — THAT session's
        # agent, not the globally active one.
        applyOwnershipAction!(:resume, sessionIdFor(sess))
        pushAgentFeedbackToWindow(wv, "Assist", strip(text), "info";
                                  sessionId = sessionIdFor(sess))
        return jsonOk()
    end

    # Lifecycle probe hook: called by index.html once the DOM + xterm are up
    # (i.e. from INSIDE the event loop, on the main thread — exactly like
    # upstream's passing end_test). Returns "autoclose" to make THIS page ask
    # for a native close itself, or "stay" otherwise. With BOTH
    # FLOOTY_UI_AUTOCLOSE=1 AND FLOOTY_AGENT_POPUP_PROBE=1 we additionally
    # push two synthetic feedback popups right here (dispatch+eval!) and set
    # popupProbe:"1" so the page keeps the window open ~2s to let them render
    # and acknowledge before closing.
    Webviews.bind(wv, "flootyUiReady") do _
        autoclose = get(ENV, "FLOOTY_UI_AUTOCLOSE", "") == "1"
        popupProbe = get(ENV, "FLOOTY_AGENT_POPUP_PROBE", "") == "1"
        # Warp-style single-PTY embedding: caller adopts its own session and
        # the page must NOT spawn its own default terminal.
        cfg = terminalUiConfig[]
        skipAuto = get(ENV, "FLOOTY_UI_SKIP_AUTOCREATE", "") == "1" || cfg.chatOnly
        if popupProbe
            pushAgentFeedbackToWindow(wv, "Flooty Agent",
                "Monitoring this session. Long-running statements get an insight popup.",
                "info")
            pushAgentFeedbackToWindow(wv, "Insight",
                "Statement running >10s — consider collect()ing lazily.",
                "warn")
        end
        parts = ["\"success\":true"]
        autoclose && push!(parts, "\"action\":\"autoclose\"")
        popupProbe && push!(parts, "\"popupProbe\":\"1\"")
        skipAuto && push!(parts, "\"skipAutocreate\":\"1\"")
        chatProvider = CHAT_SESSION_PROVIDER[]
        if chatProvider !== nothing
            try
                cid = String(Base.invokelatest(chatProvider))
                isempty(cid) || push!(parts, "\"chatSession\":\"" * escapeJsString(cid) * "\"")
            catch err
                @debug "flootyWebUi: chat session provider failed" exception = err
            end
        end
        # Initial ownership state: the ACTIVE adopted session already held by
        # the agent renders as ai from the first paint (no flash of julia).
        sess = ACTIVE_OWNERSHIP[]
        if sess !== nothing
            try
                currentState(sess.control) == :agent &&
                    push!(parts, "\"initialOwner\":\"agent\"")
            catch err
                @debug "flootyWebUi: ownership init failed" exception = err
            end
        end
        return "{" * join(parts, ",") * "}"
    end

    # Scratch space: write raw lines from the page's scratch editor into the
    # PTY as USER-typed input (loopback-safe via the line discipline, same as
    # terminal_input — the REPL echoes it back through the tail tap).
    Webviews.bind(wv, "terminal_scratchSubmit") do args
        resp = try
            payload = jsonArgs(args)
            id = jsonStr(payload, :terminal_id)
            text = jsonStr(payload, :text)
            entry = isempty(id) ? nothing : getEntry(id)
            if entry === nothing
                errorJson("terminal not found")
            elseif isempty(strip(text))
                errorJson("empty scratch content")
            else
                bytes = codeunits(text)
                writeToPty(entry.pty, Vector{UInt8}(bytes); source = :user)
                jsonOk()
            end
        catch err
            @error "flootyWebUi: terminal_scratchSubmit failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    # Return the collected output tail for a terminal (syntax view readback).
    # Uses the entry's rolling `collected` buffer — scrubSentinelLines is NOT
    # applied here; the syntax pane is a debugging lens, not the live display.
    Webviews.bind(wv, "terminal_tail") do args
        resp = try
            payload = jsonArgs(args)
            id = jsonStr(payload, :terminal_id)
            entry = isempty(id) ? nothing : getEntry(id)
            if entry === nothing
                errorJson("terminal not found")
            else
                tail = collectedText(entry)
                "{\"success\":true,\"tail\":\"" * escapeJsString(tail) * "\"}"
            end
        catch err
            @error "flootyWebUi: terminal_tail failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    # Rendered xterm viewport reports: the page serializes the visible lines of a
    # terminal's xterm buffer and posts them here; agentSnapshot reads the cache.
    Webviews.bind(wv, "flootyTerminalBufferReport") do args
        resp = try
            payload = jsonArgs(args)
            id = jsonStr(payload, :terminal_id)
            text = jsonStr(payload, :text)
            if isempty(id)
                errorJson("missing terminal_id")
            else
                # No char cap: the snapshot is the whole buffer (bounded only by
                # xterm's own scrollback setting).
                clipped = text
                meta = (rows = jsonInt(payload, :rows, 0),
                        columns = jsonInt(payload, :columns, 0),
                        scrollback_rows = jsonInt(payload, :scrollback_rows, 0),
                        scrollback_truncated = jsonBool(payload, :scrollback_truncated,
                                                        false))
                lock(TERMINAL_BUFFER_LOCK) do
                    TERMINAL_BUFFER_CACHE[id] = clipped
                    TERMINAL_BUFFER_META[id] = meta
                    TERMINAL_BUFFER_AT[id] = time()
                end
                jsonOk()
            end
        catch err
            @error "flootyWebUi: flootyTerminalBufferReport failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    # Agent model picker: what the endpoint offers, and what the user picked.
    # Handled by the agent package's controller (see addAgentModelHooks!); with no
    # controller installed the page simply gets an empty list.
    Webviews.bind(wv, "flootyModels") do args
        payload = try
            jsonArgs(args)
        catch
            nothing
        end
        tid = payload === nothing ? "" : jsonStr(payload, :terminal_id)
        jsonAgentModelState(Base.invokelatest(agentModelState, String(tid)))
    end
    Webviews.bind(wv, "flootySplitCreated") do args
        resp = try
            payload = jsonArgs(args)
            # The page knows the split's PARENT TAB; pass its session id so a
            # focus race (or a background tab) cannot parent the child to the
            # wrong main. Fall back to the active ownership for old pages.
            parent = jsonStr(payload, :parent_id)
            isempty(parent) && (parent = ACTIVE_OWNERSHIP_ID[])
            setSplitSession!(jsonStr(payload, :terminal_id); parent_id = parent)
            jsonOk()
        catch err
            @error "flootyWebUi: flootySplitCreated failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    Webviews.bind(wv, "flootySplitClosed") do args
        resp = try
            payload = jsonArgs(args)
            setSplitSession!(jsonStr(payload, :terminal_id); closed = true)
            jsonOk()
        catch err
            @error "flootyWebUi: flootySplitClosed failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    # Re-parent a session: the tree navigation's "make this my parent". The
    # page has already moved the pane; the backend moves the session's edge
    # so the app's tree (and its monitors) agree.
    Webviews.bind(wv, "flootyReparent") do args
        resp = try
            payload = jsonArgs(args)
            child = jsonStr(payload, :terminal_id)
            parent = jsonStr(payload, :parent_id)
            isempty(child) && return errorJson("no session to re-parent")
            # The layer's split registry holds the PANE tree (the terminals);
            # a chat or browser session only moves in the app's tree.
            if sessionForTerminalId(child) !== nothing
                setSplitSession!(child; parent_id = parent)
            end
            notifySessionReparent!(child, parent)
            jsonOk()
        catch err
            @error "flootyWebUi: flootyReparent failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    Webviews.bind(wv, "flootySetMonitorModel") do args
        resp = try
            payload = jsonArgs(args)
            name = jsonStr(payload, :model)
            tid = jsonStr(payload, :terminal_id)
            selector = AGENT_MONITOR_SELECTOR[]
            if isempty(name)
                errorJson("missing model")
            elseif selector === nothing
                errorJson("no agent monitor controller installed")
            else
                state = Base.invokelatest(selector, String(name), String(tid))
                jsonAgentModelState(state)
            end
        catch err
            @error "flootyWebUi: flootySetMonitorModel failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    Webviews.bind(wv, "flootyNewChildChat") do args
        resp = try
            payload = jsonArgs(args)
            handler = NEW_CHAT_HANDLER[]
            handler === nothing && return errorJson("no chat spawner installed")
            sess = Base.invokelatest(handler, jsonStr(payload, :parent_id),
                                     jsonStr(payload, :name), false)
            jsonOk()
        catch err
            @error "flootyWebUi: flootyNewChildChat failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    Webviews.bind(wv, "flootyNewRootChat") do args
        resp = try
            payload = jsonArgs(args)
            handler = NEW_CHAT_HANDLER[]
            handler === nothing && return errorJson("no chat spawner installed")
            sess = Base.invokelatest(handler, "", jsonStr(payload, :name), true)
            jsonOk()
        catch err
            @error "flootyWebUi: flootyNewRootChat failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    Webviews.bind(wv, "flootyCloseChat") do args
        resp = try
            payload = jsonArgs(args)
            handler = CHAT_CLOSE_HANDLER[]
            handler === nothing && return errorJson("no chat closer installed")
            sid = jsonStr(payload, :session_id)
            isempty(sid) && return errorJson("missing session_id")
            closed = Base.invokelatest(handler, sid)
            JSON3.write(Dict("success" => true, "closed" => closed))
        catch err
            @error "flootyWebUi: flootyCloseChat failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    Webviews.bind(wv, "flootyNewBrowserSession") do args
        resp = try
            payload = jsonArgs(args)
            handler = NEW_BROWSER_HANDLER[]
            handler === nothing && return errorJson("no browser spawner installed")
            Base.invokelatest(handler, jsonStr(payload, :parent_id),
                              jsonStr(payload, :url), jsonStr(payload, :name))
            jsonOk()
        catch err
            @error "flootyWebUi: flootyNewBrowserSession failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    Webviews.bind(wv, "flootyBrowserNavigate") do args
        resp = try
            payload = jsonArgs(args)
            handler = BROWSER_NAV_HANDLER[]
            handler === nothing && return errorJson("no browser navigator installed")
            sid = jsonStr(payload, :session_id)
            url = jsonStr(payload, :url)
            navigateBrowserWindow!(sid, url)
            Base.invokelatest(handler, sid, url)
            jsonOk()
        catch err
            @error "flootyWebUi: flootyBrowserNavigate failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    Webviews.bind(wv, "flootySetModel") do args
        resp = try
            payload = jsonArgs(args)
            name = jsonStr(payload, :model)
            tid = jsonStr(payload, :terminal_id)
            selector = AGENT_MODEL_SELECTOR[]
            if isempty(name)
                errorJson("missing model")
            elseif selector === nothing
                errorJson("no agent model controller installed")
            else
                state = Base.invokelatest(selector, String(name), String(tid))
                jsonAgentModelState(state)
            end
        catch err
            @error "flootyWebUi: flootySetModel failed" exception = err
            errorJson(sprint(showerror, err))
        end
        resp
    end

    # The page reports its live selection (document selection, else the active
    # terminal's): the app's ⌘C handler copies THAT, because WebKit's own copy
    # cannot reach the pasteboard here.
    Webviews.bind(wv, "flootySelectionChanged") do args
        payload = jsonArgs(args)
        setLiveSelection!(jsonStr(payload, :text))
        jsonOk()
    end

    @info "flootyWebUi: terminal bindings registered"
    return wv
end

const ASSET_MIME_TYPES = Dict(
    ".html"  => "text/html; charset=utf-8",
    ".js"    => "application/javascript",
    # ES modules (the markdown bundles): browsers refuse a module served as
    # octet-stream, so `.mjs` needs a JavaScript MIME type of its own.
    ".mjs"   => "text/javascript",
    ".css"   => "text/css",
    ".jpg"   => "image/jpeg",
    ".jpeg"  => "image/jpeg",
    ".png"   => "image/png",
    ".gif"   => "image/gif",
    ".svg"   => "image/svg+xml",
    ".ico"   => "image/x-icon",
    ".woff"  => "font/woff",
    ".woff2" => "font/woff2",
)

const _INPUT_PROBE_HTML = """
<!doctype html><html><head><meta charset=\"utf-8\"><title>Input Probe</title>
<style>body{background:#111;color:#eee;font:14px monospace;padding:20px}
input{width:60%%;font:14px monospace;background:#222;color:#fff;border:1px solid #555}
#log{white-space:pre;margin-top:12px;color:#9f9}</style></head><body>
<h3>Minimal input probe</h3>
<input id=\"i\" placeholder=\"type here\"><div id=\"log\"></div>
<script>
const log=document.getElementById('log');
document.getElementById('i').addEventListener('input',e=>{log.textContent+='key:'+e.data+'\\\\n';});
document.addEventListener('keydown',e=>{log.textContent+='kd:'+e.key+'\\\\n';});
window.addEventListener('focus',()=>{log.textContent+='WIN-FOCUS\\\\n';});
setTimeout(()=>{document.getElementById('i').focus();log.textContent+='autofocus-done\\\\n';},300);
callJulia('flootyUiReady','{}').catch(()=>{});
</script></body></html>
"""

"""
    launchInputProbeUI(; width = 700, height = 420) -> Webviews.Webview

DIAGNOSTIC window serving `_INPUT_PROBE_HTML` (plain <input>, no xterm).
If keystrokes reach this page, the native focus path is fine and the problem
is in index.html/xterm; if not, events never reach web content at all.
"""
function launchInputProbeUI(; width::Int = 700, height::Int = 420)
    terminalUiRunning[] &&
        error("launchInputProbeUI: UI already running")
    ensureAppTerminatePolicy!()
    installSendEventProbe!()
    startAssetServer!()
    port = terminalUiAssetPort[]

    wv = Webviews.Webview((width, height);
                          title = "Flooty Input Probe",
                          debug = false,
                          enable_webio = false,
                          auto_terminate = true)
    installFlootyWindowDelegate!(wv)
    Webviews.bind(wv, "flootyUiReady") do _
        @debug string("[binding] probe flootyUiReady entered")
        return jsonOk()
    end
    Webviews.bind(wv, "closeWindow") do _
        @debug string("[binding] probe closeWindow entered")
        try
            Webviews.close(wv)
        catch err
            @debug "probe close failed" exception = err
            requestStopEventLoop!(wv)
        end
        return jsonOk()
    end
    Webviews.html!(wv, _INPUT_PROBE_HTML)
    terminalUiWebview[] = wv
    terminalUiRunning[] = true
    @debug string("[probe] input-probe window launched")
    return wv
end

"""
Start serving the package assets dir on 127.0.0.1 with an EPHEMERAL port and
store it in terminalUiAssetServer/terminalUiAssetPort. Returns the port.

Webviews has no virtual-file root like webui_set_root_folder, so the UI is
backed by a tiny localhost-only HTTP server (loopback interface only).
"""
function startAssetServer!()
    assetsDir = normpath(@__DIR__, "..", "assets")
    server = HTTP.serve!("127.0.0.1", 0) do req
        target = string(req.target)
        path = split(target, '?'; limit = 2)[1]
        if isempty(path) || path == "/"
            path = "/index.html"
        end
        # The page's runtime configuration: the data-plane channel's port.
        if path == "/runtime.js"
            body = string("window.FLOOTY_TERMINAL = { channelPort: ",
                           CHANNEL_PORT[], " };\n")
            return HTTP.Response(200, ["Content-Type" => "application/javascript"];
                                 body = body)
        end
        if path == "/theme.js"
            body = "window.FLOOTY_THEME_STATE = " *
                   JSON3.write(ThemeManager.themeState()) * ";\n"
            return HTTP.Response(200, ["Content-Type" => "application/javascript",
                                       "Cache-Control" => "no-store"]; body = body)
        end
        file = normpath(assetsDir, lstrip(path, '/'))
        notFound = HTTP.Response(404, ["Content-Type" => "text/plain"];
                                 body = "not found")
        # Path traversal guard: resolved path must stay inside assetsDir.
        withinAssets = file == assetsDir ||
                       startswith(file, assetsDir * Base.Filesystem.path_separator)
        (withinAssets && isfile(file)) || return notFound
        mime = get(ASSET_MIME_TYPES, lowercase(splitext(file)[2]),
                   "application/octet-stream")
        return HTTP.Response(200, ["Content-Type" => mime]; body = read(file))
    end
    # With an ephemeral port (0), HTTP's Listener.addr keeps the ORIGINAL port
    # 0 — the really-bound address lives on the inner Base.IOServer socket,
    # which Sockets.listen binds synchronously inside the Listener constructor
    # (i.e. before serve! returns). Read it straight from there.
    hostBound, portBound = try
        Sockets.getsockname(server.listener.server::Base.IOServer)
    catch err
        error("startAssetServer!: could not read bound asset-server socket: ",
              sprint(showerror, err))
    end
    port = Int(portBound)
    port > 0 || error("startAssetServer!: asset server socket unbound")
    terminalUiAssetServer[] = server
    terminalUiAssetPort[] = port
    return port
end

"Build {success:true[, action=...]} response JSON."
jsonOkWith(actionKey::String, actionValue::String) =
    "{\"success\":true,\"" * escapeJsString(actionKey) * "\":\"" *
    escapeJsString(actionValue) * "\"}"

"Stop the asset HTTP server (idempotent)."
function stopAssetServer!()
    server = terminalUiAssetServer[]
    terminalUiAssetServer[] = nothing
    terminalUiAssetPort[] = 0
    server === nothing && return nothing
    try
        close(server)
    catch e
        @debug "flootyWebUi: closing asset server failed" exception = e
    end
    return nothing
end

# ---- window lifecycle -----------------------------------------------------------

"Tear down every active terminal and clear window state. Idempotent."
function shutdownTerminalTerminals!()
    lock(terminalUiShutdownLock) do
        for id in activeTerminalIds()
            handleTerminalKill(id)
        end
        closeAllBrowserWindows!()
        terminalUiWebview[] = nothing
        terminalUiRunning[] = false
    end
    return nothing
end

"""
    launchTerminalUI(; cmd = defaultTerminalCmd(),
                     width = 1100, height = 750) -> Webviews.Webview

Open the xterm.js terminal UI in a NATIVE WebView window (WKWebView on macOS)
and return immediately. The window gets a Dock icon and standard traffic-light
controls. Key/mouse FOCUS is handled natively here: Webviews' backend only
calls activateIgnoringOtherApps once at construction — before any run loop
exists, where it silently no-ops — and never makes the WKWebView first
responder, so keystrokes never reach web content. We re-activate after the
loop starts and force makeFirstResponder: on the WKWebView.

Most users want the BLOCKING `runTerminalUI()`; this variant exists for REPL
experimentation and probes. Call `stopTerminalUI()` to tear down. Assets are
served from an ephemeral 127.0.0.1 HTTP server started here.

There is deliberately NO polling anywhere: no is_shown loops, no timers.
Output reaches the page via dispatch+eval! pushes only.
"""
function launchTerminalUI(; cmd::Cmd = defaultTerminalCmd(),
                          width::Int = 1100, height::Int = 750,
                          chatOnly::Bool = false)
    terminalUiRunning[] &&
        error("launchTerminalUI: UI already running; call stopTerminalUI() first")

    terminalUiConfig[] = TerminalUiConfig(cmd = cmd, width = width, height = height,
                                          chatOnly = chatOnly)

    # Webviews v1.2.0 ships a broken applicationShouldTerminateAfterLast-
    # WindowClosed: (calls an undefined `_terminate`), so patch it on
    # WebviewAppDelegate (registered in Webviews' __init__, safe anytime).
    ensureAppTerminatePolicy!()
    installSendEventProbe!()
    # ⌘C/⌘V are ours: in this process WebKit's own copy writes nothing to the
    # pasteboard and its paste reads nothing from it (see handleEditingKey).
    installEditingKeyHandler!()

    startAssetServer!()
    baseUrl = "http://127.0.0.1:$(terminalUiAssetPort[])/"
    # The data plane: one binary WebSocket per terminal (the PTY bytes). The
    # page reads the port from /runtime.js, which the asset server serves.
    channelPort = start_channel_server!()

    wv = Webviews.Webview((width, height);
                          title = "Flooty Terminal",
                          debug = false,
                          enable_webio = false,
                          auto_terminate = true)
    installFlootyWindowDelegate!(wv)
    registerTerminalBindings!(wv)
    Webviews.navigate!(wv, baseUrl * "index.html")

    terminalUiWebview[] = wv
    terminalUiRunning[] = true

    @info "flootyWebUi: terminal UI running" url = baseUrl * "index.html" width height
    @info "flootyWebUi: data channel" port = channelPort
    return wv
end

"""
Block in Webviews' native NSApplication event loop, then tear down terminals
and the asset server. MUST be called from the caller's thread directly (never
wrap in @async — AppKit wants the loop on the invoking thread, exactly like
upstream's test suite).
"""
function runEventLoopAndTeardown!(wv)
    # Two one-shot activation passes: immediately after the loop starts and
    # again once the page has settled (activation can be swallowed while
    # AppKit is still spinning up the first event passes).
    scheduleFocusActivation!(wv; delayS = 0.05)
    scheduleFocusActivation!(wv; delayS = 1.5)
    try
        Webviews.run(wv)
    catch err
        @debug "flootyWebUi: webview event loop ended" exception = err
    finally
        shutdownTerminalTerminals!()
        stopAssetServer!()
    end
    return nothing
end

"""
    runTerminalUI(; cmd = defaultTerminalCmd(),
                  width = 1100, height = 750, chatOnly = false) -> Nothing

Launch the UI (same kwargs as `launchTerminalUI`) and then BLOCK the CALLING
thread inside Webviews' native NSApplication event loop (`run(wv)`) until the
app terminates — the user closing the window / red traffic light (our patched
delegates answer safely) or the closeWindow glass button (native close path).
Afterwards all terminals are killed and the asset server stopped. This is the
FlootyOldPTY-style usage: make it the LAST statement of a standalone script.
No polling of any kind.
"""
function runTerminalUI(; kwargs...)
    wv = launchTerminalUI(; kwargs...)
    runEventLoopAndTeardown!(wv)
    return nothing
end

"""
    stopTerminalUI()

Programmatic teardown: close/destroy the WebView window, kill every active
terminal, and stop the asset server. Safe to call repeatedly and when the UI
was never launched. (When `runTerminalUI` is driving the loop, prefer closing
the window or the closeWindow button so the loop unwinds through its own
finally-block.)
"""
function stopTerminalUI()
    terminalUiRunning[] = false
    wv = terminalUiWebview[]
    if wv !== nothing
        terminalUiWebview[] = nothing
        try
            Webviews.close(wv)
        catch e
            @debug "flootyWebUi: webview close failed" exception = e
        end
        try
            requestStopEventLoop!(wv)
        catch e
            @debug "flootyWebUi: requestStopEventLoop! failed" exception = e
        end
        try
            Webviews.destroy(wv)
        catch e
            @debug "flootyWebUi: webview destroy failed" exception = e
        end
    end
    shutdownTerminalTerminals!()
    stopAssetServer!()
    return nothing
end
"""
    addChatTabUi(wv, id, name) :: Bool

Create (or focus) a CHAT tab for `id` (the ChatSession id). The page owns the
tab rail and the transcript; this is how a child chat session becomes visible.
"""
function addChatTabUi(wv, id::AbstractString, name::AbstractString = "Chat";
                      root_id::AbstractString = "",
                      parent_id::AbstractString = "")::Bool
    wv === nothing && return false
    try
        Webviews.dispatch(wv) do
            Webviews.eval!(wv,
                "if (typeof window.flootyAddChatTab === 'function') " *
                "window.flootyAddChatTab(\"" * escapeJsString(String(id)) * "\", \"" *
                escapeJsString(String(name)) * "\", \"" *
                escapeJsString(String(root_id)) * "\", \"" *
                escapeJsString(String(parent_id)) * "\");")
        end
        return true
    catch err
        @debug "flootyWebUi: addChatTabUi failed" exception = err
        return false
    end
end

"""
    addBrowserTabUi(wv, id, name, url; root_id, parent_id) :: Bool

Create (or focus) a BROWSER tab for a BrowserSession; the page owns the panel
(URL bar + iframe) and the chat rail.
"""
function addBrowserTabUi(wv, id::AbstractString, name::AbstractString,
                         url::AbstractString; root_id::AbstractString = "",
                         parent_id::AbstractString = "")::Bool
    wv === nothing && return false
    try
        Webviews.dispatch(wv) do
            Webviews.eval!(wv,
                "if (typeof window.flootyAddBrowserTab === 'function') " *
                "window.flootyAddBrowserTab(\"" * escapeJsString(String(id)) * "\", \"" *
                escapeJsString(String(name)) * "\", \"" * escapeJsString(String(url)) *
                "\", \"" * escapeJsString(String(root_id)) * "\", \"" *
                escapeJsString(String(parent_id)) * "\");")
        end
        # The panel is only an iframe; many sites refuse it. Give the session
        # a real top-level WKWebView window as well.
        openBrowserWindow!(String(id), String(url); name = String(name))
        return true
    catch err
        @debug "flootyWebUi: addBrowserTabUi failed" exception = err
        return false
    end
end

"""Create a new terminal tab in the live UI (best-effort)."""
function createTerminalUi(wv; parent_id::AbstractString = "")
    wv === nothing && return false
    try
        Webviews.dispatch(wv) do
            Webviews.eval!(wv,
                "if (typeof createTerminal === 'function') createTerminal(" *
                "{ parentSession: \"" * escapeJsString(String(parent_id)) * "\" });")
        end
        return true
    catch err
        @debug "flootyWebUi: createTerminalUi failed" exception = err
        return false
    end
end
