# terminalPerfProbe.jl — WHERE does the terminal's time go? (WebGL vs DOM)
#
# The WebGL addon only changes how the GRID IS PAINTED. If enabling it does
# not feel faster, the bottleneck is somewhere else: the PTY -> Julia -> page
# bridge, xterm's JS parser, or the page's own per-chunk work. This probe
# measures each stage, A/B (WebGL renderer vs xterm's DOM renderer), so the
# next optimization is aimed at the measured hot spot instead of a guess.
#
# MEASURED STAGES
#   latency   : write one char into the terminal -> time until the frame that
#               shows it is painted (the floor is one display frame)
#   burst     : write N lines -> time until the last line is PAINTED, and how
#               many frames the burst produced
#   repaint   : force a full-viewport repaint -> time to the painted frame
#   scroll    : scroll the viewport -> time to the painted frame
#   bridge    : (Julia side) push N bytes of PTY output through the real
#               tap -> bridge -> page path and measure the delivery rate
#
# Run it on a machine WITH A DISPLAY (the window must be visible: a hidden
# webview page suspends requestAnimationFrame, so painting cannot be measured
# and the probe says so).
#
#   julia --startup-file=no --project=. test/terminalPerfProbe.jl
import FlootyPTYLayer
using FlootyPTYLayer
using JSON3
using Logging
using Webviews

# ---- page-side benchmark suite ------------------------------------------------
const BENCH_JS = """
window.__perf = function (kind, arg) {
  return new Promise(function (resolve) {
    var ids = Object.keys(typeof terminals !== 'undefined' ? terminals : {});
    if (!ids.length) { resolve({ error: 'no-terminal' }); return; }
    var info = terminals[ids[0]];
    var t = info.term;
    var dom = !!(info.gpu && info.gpu.state === 'dom');

    function withFrame(cb) {
      // A frame only happens when something changes; force one by flipping an
      // option (the renderer repaints on any option change).
      var before = frames;
      try { t.options.cursorBlink = !t.options.cursorBlink; } catch (e) {}
      var wait = setInterval(function () {
        if (frames > before) { clearInterval(wait); cb(true); }
      }, 2);
      setTimeout(function () { clearInterval(wait); cb(false); }, 2000);
    }

    var frames = 0, tFirst = 0;
    var off = t.onRender(function () { frames++; if (!tFirst) tFirst = performance.now(); });
    var t0 = performance.now();
    function finish(extra) {
      var t1 = performance.now();
      off.dispose();
      var res = { kind: kind, dom: dom, frames: frames,
                  vis: document.visibilityState,
                  totalMs: +(t1 - t0).toFixed(2),
                  firstFrameMs: tFirst ? +(tFirst - t0).toFixed(2) : null };
      for (var k in extra) res[k] = extra[k];
      resolve(res);
    }

    if (kind === 'latency') {
      t.write('x', function () { withFrame(function () { finish({}); }); });
    } else if (kind === 'repaint') {
      t.refresh(0, t.rows - 1);
      withFrame(function () { finish({}); });
    } else if (kind === 'scroll') {
      t.scrollLines(-20);
      withFrame(function () { finish({}); });
    } else if (kind === 'burst') {
      var text = '';
      for (var i = 0; i < arg.lines; i++) {
        text += 'line ' + i + ' lorem ipsum dolor sit amet consectetur ' + i + '\\r\\n';
      }
      var tParsed = 0;
      t.write(text, function () { tParsed = performance.now(); });
      var deadline = t0 + 20000;
      function check() {
        var b = t.buffer.active;
        var found = false;
        for (var k = 0; k < 3 && !found; k++) {
          var idx = b.baseY + b.cursorY - k;
          if (idx < 0) continue;
          var line = b.getLine(idx);
          if (line && line.translateToString(true).indexOf('line ' + (arg.lines - 1)) >= 0) found = true;
        }
        if (found || performance.now() > deadline) {
          withFrame(function () {
            finish({ lines: arg.lines, bytes: text.length,
                     parseMs: +(tParsed - t0).toFixed(2), painted: found });
          });
        } else { setTimeout(check, 5); }
      }
      check();
    }
  });
};
"""

# ---- capture the page's debug channel -----------------------------------------
mutable struct CaptureLogger <: Logging.AbstractLogger
    lines::Vector{String}
end
CaptureLogger() = CaptureLogger(String[])
Logging.min_enabled_level(::CaptureLogger) = Logging.Debug
Logging.shouldlog(::CaptureLogger, args...) = true
function Logging.handle_message(l::CaptureLogger, level, msg, mod, group, id,
                                file, line; kwargs...)
    push!(l.lines, string(msg))
    return nothing
end

cap = CaptureLogger()
wv = launchTerminalUI()
results = Any[]

watchdog = @async begin
    deadline = time() + 300
    while time() < deadline && FlootyPTYLayer.gpuRendererReport() === nothing
        sleep(0.25)
    end
    sleep(1.0)

    # Is the page actually visible? A hidden webview suspends rAF, so the
    # paint stages cannot be measured (and every frame time is a timeout).
    try
        Webviews.dispatch(wv) do
            Webviews.eval!(wv, "window.flootyDebug('VIS ' + document.visibilityState + " *
                              "' hidden=' + document.hidden + ' focus=' + document.hasFocus());")
        end
    catch e
    end
    sleep(0.5)

    # The page must be VISIBLE: WebKit suspends requestAnimationFrame for an
    # OCCLUDED window (one fully covered by other windows), and then xterm
    # never paints, so every paint stage would time out. Keep the window in
    # front and wait for the page to report itself visible.
    visible = false
    for attempt in 1:40
        try
            FlootyPTYLayer.ensureWindowFocus!(wv)
        catch e
        end
        try
            Webviews.dispatch(wv) do
                Webviews.eval!(wv, "window.flootyDebug('VIS ' + document.visibilityState + " *
                                  "' hidden=' + document.hidden);")
            end
        catch e
        end
        sleep(0.3)
        for l in cap.lines
            if startswith(l, "[ui-debug] VIS visible")
                visible = true
            end
        end
        visible && break
    end
    if !visible
        println("!! the webview page never became VISIBLE (occluded window /")
        println("!!   no display): WebKit suspends requestAnimationFrame, so the")
        println("!!   PAINT stages cannot be measured. Bring the window to the")
        println("!!   front and re-run.")
    else
        println("page is visible — paint stages are measurable")
    end

    try
        Webviews.dispatch(wv) do
            Webviews.eval!(wv, BENCH_JS)
        end
    catch e
        println("BENCH_INSTALL_FAILED: ", sprint(showerror, e))
    end
    sleep(0.5)

    function runBench(kind, arg, label)
        done = Ref(false)
        # The window must be in FRONT: WebKit suspends requestAnimationFrame
        # for an occluded window, and then xterm never paints. Re-activate and
        # wait for the page to be visible before EVERY stage.
        visible = false
        for attempt in 1:20
            try
                FlootyPTYLayer.ensureWindowFocus!(wv)
                Webviews.dispatch(wv) do
                    Webviews.eval!(wv, "window.flootyDebug('VIS ' + document.visibilityState);")
                end
            catch e
            end
            sleep(0.25)
            for l in cap.lines
                startswith(l, "[ui-debug] VIS visible") && (visible = true)
            end
            visible && break
        end
        visible || println("!! ", label, ": the page stayed occluded — no paint measured")
        # __perf(kind, arg): kind is a STRING, arg is the JSON options object.
        script = "window.__perf(\"" * kind * "\", " * arg * ").then(function (r) {" *
                 " window.flootyDebug('PERF ' + JSON.stringify(r)); });"
        Webviews.dispatch(wv) do
            Webviews.eval!(wv, script)
        end
        t0 = time()
        consumed = length(cap.lines)
        while time() - t0 < 12 && !done[]
            seen = cap.lines
            for i in (consumed + 1):length(seen)
                l = seen[i]
                startswith(l, "[ui-debug] PERF {") || continue
                v = try
                    JSON3.read(replace(l, "[ui-debug] PERF " => ""))
                catch
                    nothing
                end
                v === nothing && continue
                v["label"] = label
                push!(results, v)
                done[] = true
            end
            consumed = length(seen)
            done[] && break
            sleep(0.05)
        end
        done[] || println("TIMEOUT: ", label)
        return nothing
    end

    println("--- renderer: WEBGL ---")
    runBench("latency", "{}", "webgl/latency")
    runBench("repaint", "{}", "webgl/repaint")
    runBench("scroll", "{}", "webgl/scroll")
    runBench("burst", "{\"lines\":2000}", "webgl/burst-2000")

    println("--- renderer: DOM (the WebGL addon is disposed) ---")
    runBench("latency", "{}", "dom/latency")
    runBench("repaint", "{}", "dom/repaint")
    runBench("scroll", "{}", "dom/scroll")
    runBench("burst", "{\"lines\":2000}", "dom/burst-2000")

    # ---- bridge: PTY bytes -> tap -> Julia -> page --------------------------
    # The page counts the bytes that reach it, so this measures the transport
    # (not the painting).
    try
        Webviews.dispatch(wv) do
            Webviews.eval!(wv, """
            (function () {
              if (window.__bytesHooked) return;
              window.__bytesHooked = true;
              window.__bytes = 0;
              var orig = window.handleTerminalOutput;
              window.handleTerminalOutput = function (id, out) {
                window.__bytes += out.length;
                return orig(id, out);
              };
              setInterval(function () {
                try { window.flootyDebug('BYTES ' + window.__bytes); } catch (e) {}
              }, 50);
            })()
            """)
        end
    catch e
    end
    sleep(0.5)
    # Drive the LIVE terminal from the page (the real path: page -> binding ->
    # Julia -> PTY -> back). The PTY is in canonical mode, so the stream must
    # be newline-terminated.
    try
        Webviews.dispatch(wv) do
            Webviews.eval!(wv, """
            (function () {
              var ids = Object.keys(terminals);
              terminals[ids[0]].term.write('seq 1 200000\\n');
            })()
            """)
        end
    catch e
        println("BRIDGE_CMD_FAILED: ", sprint(showerror, e))
    end
    t0 = time()
    seen = 0
    while time() - t0 < 60
        for l in cap.lines
            startswith(l, "[ui-debug] BYTES ") || continue
            n = tryparse(Int, replace(l, "[ui-debug] BYTES " => ""))
            n === nothing && continue
            seen = max(seen, n)
        end
        seen >= 1_000_000 && break
        sleep(0.05)
    end
    t1 = time()
    push!(results, Dict("label" => "bridge/1MiB",
                        "bytes" => seen,
                        "totalMs" => round((t1 - t0) * 1000; digits = 1),
                        "MiBps" => round(seen / 1_048_576 / max(t1 - t0, 1e-9); digits = 2)))

    # ---- report -------------------------------------------------------------
    println()
    println("=== terminal perf (WebGL vs DOM) ===")
    for r in results
        println("  ", r)
    end
    try
        Webviews.dispatch(wv) do
            Webviews.close(wv)
        end
    catch e
    end
end

Logging.with_logger(cap) do
    FlootyPTYLayer.runEventLoopAndTeardown!(wv)
end
wait(watchdog)
println("PERF_PROBE_DONE")
