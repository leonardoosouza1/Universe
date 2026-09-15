#!/usr/bin/env python3
"""Check actual playback motion and completed laps in a local Chrome session."""

import argparse
import base64
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import urllib.request

import websocket


class Browser:
    def __init__(self, socket):
        self.socket = socket
        self.sequence = 0
        self.errors = []

    def call(self, method, params=None):
        self.sequence += 1
        self.socket.send(json.dumps({"id": self.sequence, "method": method, "params": params or {}}))
        while True:
            response = json.loads(self.socket.recv())
            if response.get("method") == "Runtime.exceptionThrown":
                self.errors.append(response)
            if response.get("id") == self.sequence:
                if "error" in response:
                    raise RuntimeError(response["error"])
                return response.get("result", {})

    def evaluate(self, expression):
        response = self.call("Runtime.evaluate", {"expression": expression, "returnByValue": True})
        if "exceptionDetails" in response:
            raise RuntimeError(response["exceptionDetails"])
        return response["result"].get("value")

    def screenshot(self, path):
        data = self.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
        path.write_bytes(base64.b64decode(data["data"]))


def check(browser, html, output):
    browser.call("Runtime.enable")
    browser.call("Page.enable")
    browser.call("Emulation.setDeviceMetricsOverride", {
        "width": 1600, "height": 1000, "deviceScaleFactor": 1, "mobile": False,
    })
    browser.call("Page.navigate", {"url": html.resolve().as_uri()})
    deadline = time.monotonic() + 30
    while not browser.evaluate("Boolean(window.__UNIVERSE_REPLAY__)"):
        if time.monotonic() >= deadline:
            raise RuntimeError("replay did not initialize")
        time.sleep(0.05)
    evaluate = browser.evaluate
    terrain = evaluate("__UNIVERSE_REPLAY__.data.meta.terrain || null")
    if terrain is not None:
        assert terrain["alignment"] == "cell_center"
        assert "recorded cells" in evaluate('document.getElementById("terrain-legend").textContent')
        if terrain["hole_cells"]:
            evaluate('document.getElementById("inspect-hole").click()')
            visible_hole = evaluate("""(() => {
                const scene = document.getElementById('scene'), ctx = scene.getContext('2d');
                const pixels = ctx.getImageData(scene.width/2-6,scene.height/2-6,12,12).data;
                let count=0; for(let i=0;i<pixels.length;i+=4) if(pixels[i]===53&&pixels[i+1]===36&&pixels[i+2]===15)count++;
                return count;
            })()""")
            assert visible_hole > 0, "actual hole cells are absent from the canvas"
            browser.screenshot(output / "holes-inspected.png")
            evaluate('document.getElementById("fit-track").click()')
    assert evaluate("window.REPLAY_VALIDATION?.valid === true"), "behavioral gate failed or missing"
    assert evaluate("window.__UNIVERSE_REPLAY__.state.playing"), "expected automatic playback"
    evaluate('document.getElementById("play").click()')
    evaluate('document.getElementById("timeline").value = 63; document.getElementById("timeline").dispatchEvent(new Event("input"))')
    frame_expression = "UniverseReplayModel.frameAtOrBefore(__UNIVERSE_REPLAY__.data.beings[0], __UNIVERSE_REPLAY__.data.ticks[__UNIVERSE_REPLAY__.state.index])"
    before = evaluate(frame_expression)
    evaluate('document.getElementById("playback-speed").value = "8"; document.getElementById("play").click()')
    time.sleep(0.8)
    evaluate('document.getElementById("play").click()')
    after = evaluate(frame_expression)
    displacement = math.hypot(after["x"] - before["x"], after["y"] - before["y"])
    assert after["t"] > before["t"] + 64, "clock did not advance during playback"
    assert displacement > 10, f"being did not move visibly: {displacement:.3f} m"
    paused_index = evaluate("__UNIVERSE_REPLAY__.state.index")
    time.sleep(0.2)
    assert evaluate("__UNIVERSE_REPLAY__.state.index") == paused_index, "pause did not hold"
    assert "Paused" in evaluate('document.getElementById("playback-status").textContent')
    # Inspect the middle of the actual circuit, then the final verified count.
    evaluate('document.getElementById("timeline").value = 3000; document.getElementById("timeline").dispatchEvent(new Event("input"))')
    browser.screenshot(output / "replay-track.png")
    evaluate('document.getElementById("focus-being").click()')
    browser.screenshot(output / "replay-follow.png")
    evaluate('document.getElementById("timeline").value = document.getElementById("timeline").max; document.getElementById("timeline").dispatchEvent(new Event("input"))')
    expected = evaluate("REPLAY_VALIDATION.beings[0].lap_completion_ticks.length")
    actual = evaluate('document.getElementById("stat-lap").textContent')
    assert actual == str(expected), f"displayed laps {actual} != verified {expected}"
    assert evaluate("document.documentElement.scrollWidth <= innerWidth")
    browser.call("Emulation.setDeviceMetricsOverride", {
        "width": 768, "height": 900, "deviceScaleFactor": 1, "mobile": False,
    })
    time.sleep(0.1)
    assert evaluate("document.documentElement.scrollWidth <= innerWidth"), "narrow viewport overflows"
    assert not browser.errors, browser.errors
    return {
        "passed": True, "controller": evaluate('document.getElementById("run-source").textContent'),
        "behavior": evaluate("REPLAY_VALIDATION.valid"), "being_count": evaluate("REPLAY_BEINGS.length"),
        "tick_before": before["t"], "tick_after": after["t"], "playback_displacement_m": displacement,
        "selected_completed_laps": expected, "pause_verified": True,
        "desktop_and_narrow_viewport_fit": True, "runtime_errors": browser.errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("html", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--chrome", default="/usr/bin/google-chrome")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profile = Path(tempfile.mkdtemp(prefix="replay-check-", dir=args.output))
    process = subprocess.Popen([
        args.chrome, "--headless=new", "--disable-gpu",
        "--remote-allow-origins=*", "--remote-debugging-port=0",
        f"--user-data-dir={profile}", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    socket = None
    try:
        deadline = time.monotonic() + 15
        while not (profile / "DevToolsActivePort").exists():
            if process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("Chrome did not start")
            time.sleep(0.05)
        port = (profile / "DevToolsActivePort").read_text().splitlines()[0]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=10) as response:
            tabs = json.load(response)
        page = next(tab for tab in tabs if tab["type"] == "page")
        socket = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=30)
        result = check(Browser(socket), args.html, args.output)
        (args.output / "browser-validation.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result))
    finally:
        if socket:
            socket.close()
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        shutil.rmtree(profile)


if __name__ == "__main__":
    main()
