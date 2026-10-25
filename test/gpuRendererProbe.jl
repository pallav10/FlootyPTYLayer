# gpuRendererProbe.jl — prove the terminal really renders on the GPU.
#
# The UI page activates xterm's WebGL2 renderer (xterm-addon-webgl) for every
# terminal it creates and reports the live renderer state back over the
# flootyGpuReport binding. This probe launches the REAL UI (WKWebView — the
# same webview the Flooty app uses) and asserts:
#
#   * webgl2 == true                  — the webview has a WebGL2 context
#   * state == "webgl" for a terminal — the addon activated (not the fallback)
#   * canvas == true                  — xterm paints into a <canvas> now, which
#                                       the DOM renderer never creates
#   * bright > 0 and a background majority
#                                     — PIXELS: the WebGL framebuffer really
#                                       holds the painted glyphs (read back
#                                       inside a render callback, because the
#                                       drawing buffer is not preserved across
#                                       frames)
#
# A DOM fallback would still RENDER, so this probe exists to catch a silent
# regression to the slow path (no addon, no WebGL2, context lost, blank canvas).
#
# Run as a script:
#   julia --startup-file=no --project=. test/gpuRendererProbe.jl [timeoutS]
import FlootyPTYLayer
using FlootyPTYLayer
using JSON3
using Logging
using Webviews

timeoutS = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 90.0

# Captures the page's flootyDebug channel (the page pushes its diagnostics
# there) while still letting the app's own log output through.
mutable struct CaptureLogger <: Logging.AbstractLogger
    lines::Vector{String}
    inner::Logging.AbstractLogger
end
CaptureLogger() = CaptureLogger(String[], ConsoleLogger(stderr, Logging.Info))
Logging.min_enabled_level(::CaptureLogger) = Logging.Debug
Logging.shouldlog(::CaptureLogger, args...) = true
function Logging.handle_message(l::CaptureLogger, level, msg, mod, group, id,
                                file, line; kwargs...)
    push!(l.lines, string(msg))
    # Forward only what the app itself logs; the page's debug channel is
    # captured, not printed (it would drown the probe output).
    if level >= Logging.Info
        try
            Logging.handle_message(l.inner, level, msg, mod, group, id, file, line;
                                   kwargs...)
        catch
        end
    end
    return nothing
end

# Reads the WebGL framebuffer of the live terminal INSIDE a render callback
# (the drawing buffer is not preserved between frames) and pushes the count of
# non-background pixels back to Julia over the flootyDebug channel.
const PIXEL_PROBE_JS = """
(function () {
  var t = null;
  try {
    var ids = Object.keys(typeof terminals !== 'undefined' ? terminals : {});
    if (ids.length) t = terminals[ids[0]].term;
  } catch (e) { t = null; }
  if (!t) { window.flootyDebug('GPUPIX no-terminal'); return; }
  var canvases = Array.prototype.slice.call(t.element.querySelectorAll('canvas'));
  var info = [];
  for (var i = 0; i < canvases.length; i++) {
    var cv = canvases[i];
    var isGl = !!cv.getContext('webgl2');
    var r = cv.getBoundingClientRect();
    info.push({cls: cv.className, w: cv.width, h: cv.height,
               cssW: Math.round(r.width), cssH: Math.round(r.height), gl: isGl});
  }
  var c = null, gl = null;
  for (var i = 0; i < canvases.length; i++) {
    var g = canvases[i].getContext('webgl2');
    if (g) { c = canvases[i]; gl = g; break; }
  }
  if (!gl) {
    window.flootyDebug('GPUPIX no-gl canvases=' + JSON.stringify(info));
    return;
  }
  // Keep the best sample seen: WebKit clears the drawing buffer right after
  // presenting it, so a painted frame is only readable INSIDE the render
  // callback (or in the rare window before the composite).
  var best = {tag: 'none', bright: 0, modalCount: 0, total: 1};
  function read(tag) {
    try {
      var w = c.width, h = c.height;
      var buf = new Uint8Array(w * h * 4);
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf);
      // Histogram of the RGBA quadruples: the modal colour is the background
      // the renderer painted, every other colour is content (glyphs, cursor).
      var hist = {};
      var bright = 0;
      for (var i = 0; i < buf.length; i += 4) {
        var r = buf[i], g = buf[i + 1], b = buf[i + 2], a = buf[i + 3];
        var key = (r << 24) | (g << 16) | (b << 8) | a;
        hist[key] = (hist[key] || 0) + 1;
        if (r > 0x80 && g > 0x80 && b > 0x80) bright++;
      }
      var modal = 0, modalCount = 0;
      for (var k in hist) {
        if (hist[k] > modalCount) { modalCount = hist[k]; modal = parseInt(k, 10); }
      }
      if (bright > best.bright) {
        var hex = ('00000000' + (modal >>> 0).toString(16)).slice(-8);
        best = {tag: tag, w: w, h: h, modal: '#' + hex,
                modalAlpha: (modal >>> 24) & 0xff, modalCount: modalCount,
                bright: bright, total: w * h, different: w * h - modalCount,
                canvases: info, rows: t.rows, cols: t.cols};
      }
    } catch (e) { best.error = String(e); }
  }
  function hasContent() {
    try {
      var b = t.buffer.active;
      for (var i = 0; i < b.length; i++) {
        var line = b.getLine(i);
        if (line && line.translateToString(true).trim().length > 0) return true;
      }
    } catch (e) {}
    return false;
  }
  // Read inside the render callback: that is the one moment the drawing
  // buffer still holds the frame.
  t.onRender(function () { read('onRender'); });
  // Sample for a couple of seconds as well, and force extra repaints by
  // flipping an option (the blinking cursor repaints on its own). A hidden
  // window throttles timers, so report EVERY sample that shows glyphs
  // instead of waiting for the end of the run.
  var n = 0;
  var iv = setInterval(function () {
    if (hasContent()) {
      if (n % 12 === 0) { try { t.options.cursorBlink = !t.options.cursorBlink; } catch (e) {} }
      var before = best.bright;
      read('poll');
      if (best.bright > 0 && best.bright > before) {
        window.flootyDebug('GPUPIX ' + JSON.stringify(best));
      }
    }
    n++;
    if (n > 40) {
      clearInterval(iv);
      try { t.options.cursorBlink = true; } catch (e) {}
      window.flootyDebug('GPUPIX ' + JSON.stringify(best));
    }
  }, 50);
})()
"""

println("loading OK; launching UI...")
wv = try
    launchTerminalUI()
catch e
    println("LAUNCH_FAILED: ", sprint(showerror, e))
    exit(1)
end
println("launch returned handle type: ", typeof(wv))

cap = CaptureLogger()
samples = Any[]

# Watchdog + pixel probe: runs off the main thread while the event loop owns
# it. Closes the window once the pixels came back.
watchdog = @async begin
    deadline = time() + timeoutS
    probeSent = false
    seen = Set{String}()
    while time() < deadline
        if !probeSent && FlootyPTYLayer.gpuRendererReport() !== nothing
            # The page created a terminal: install the pixel probe.
            try
                Webviews.dispatch(wv) do
                    Webviews.eval!(wv, PIXEL_PROBE_JS)
                end
            catch e
                println("PROBE_EVAL_FAILED: ", sprint(showerror, e))
            end
            probeSent = true
        end
        for line in cap.lines
            startswith(line, "[ui-debug] GPUPIX") || continue
            line in seen && continue
            push!(seen, line)
            println("PAGE: ", line)
            if startswith(line, "[ui-debug] GPUPIX {")
                parsed = try
                    JSON3.read(replace(line, "[ui-debug] GPUPIX " => ""))
                catch
                    nothing
                end
                parsed === nothing && continue
                push!(samples, parsed)
            end
        end
        # A sample counts once it actually shows glyphs.
        if any(s -> get(s, "bright", 0) > 0, samples)
            break
        end
        sleep(0.25)
    end
    if !any(s -> get(s, "bright", 0) > 0, samples)
        println("NO_PIXEL_REPORT: the page never returned painted pixels")
    end
    # Close the native window so the event loop unwinds.
    try
        Webviews.dispatch(wv) do
            Webviews.close(wv)
        end
    catch e
        println("CLOSE_FAILED: ", sprint(showerror, e))
    end
end

println("entering runTerminalUI event loop on main thread ...")
Logging.with_logger(cap) do
    FlootyPTYLayer.runEventLoopAndTeardown!(wv)
end
wait(watchdog)

report = FlootyPTYLayer.gpuRendererReport()
if report === nothing
    println("GPU_REPORT: (none — the page never reported)")
    exit(1)
end
println("GPU_REPORT: ", JSON3.write(report))

webgl2 = get(report, "webgl2", false) === true
entries = get(report, "terminals", Any[])
gpuTerminals = [e for e in entries if get(e, "state", "") == "webgl"]
canvasTerminals = [e for e in gpuTerminals if get(e, "canvas", false) === true]

println("webgl2 available: ", webgl2)
println("terminals on the WebGL renderer: ", length(gpuTerminals), "/", length(entries))
println("terminals with a WebGL canvas: ", length(canvasTerminals), "/", length(entries))
for e in entries
    println("  terminal ", get(e, "id", "?"), ": state=", get(e, "state", "?"),
            " canvas=", get(e, "canvas", false), " reason=", get(e, "reason", ""))
end
for s in samples
    println("pixel sample: ", JSON3.write(s))
end

# A painted frame is a mostly-uniform background with a small number of
# brighter pixels on top (the glyphs). "every pixel differs from the
# background" is a broken readback, not a painted frame — so require BOTH: a
# background majority AND bright glyph pixels.
painted = any(s -> get(s, "modalCount", 0) > 0.5 * get(s, "total", 1) &&
                      get(s, "bright", 0) > 0, samples)
println("WebGL framebuffer holds painted glyphs: ", painted)

ok = webgl2 && !isempty(canvasTerminals) && painted
println(ok ? "GPU_RENDERER_PROBE_OK" : "GPU_RENDERER_PROBE_FAILED")
exit(ok ? 0 : 1)
