"""
iPad control panel: slider in, flashing light out.

Run this on the external computer, then open http://<this-computer-ip>:8000
in Safari on the iPad.

    pip install websockets

Usage:

    from panel import Panel

    p = Panel()
    p.wait_for_ipad()

    p.light(True)         # light on, stays on
    p.light(False)        # light off
    p.flash(0.25)         # single pulse, 250 ms
    p.blink(2.0)          # free-running at 2 Hz
    p.blink(0)            # stop blinking

    print(p.slider)       # where the slider is right now, integer 1 - 5
    p.rating(3)           # move the slider back to the middle

    value = p.wait_for_send()      # blocks until Send is tapped
    print(p.submitted)             # same value, also kept here
"""

import asyncio
import functools
import http.server
import json
import queue
import socket
import threading

import websockets

HTTP_PORT = 8000
WS_PORT = 8765


def local_ip():
    """Best guess at this machine's LAN address (no traffic is actually sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("192.0.2.1", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


class Panel:
    """A slider and a light, living on an iPad, driven from here."""

    def __init__(self, http_port=HTTP_PORT, ws_port=WS_PORT, directory=".", quiet=False):
        self.slider = 3              # where the slider is right now, 1 - 5
        self.submitted = None        # value from the last Send tap
        self.on_slider = None        # optional callback: f(value) on every move
        self.on_send = None          # optional callback: f(value) on Send

        self._sends = queue.Queue()

        self._ws_port = ws_port
        self._clients = set()
        self._loop = None
        self._stop = None
        self._connected = threading.Event()
        self._ready = threading.Event()

        handler = functools.partial(
            _QuietHTTPHandler if quiet else http.server.SimpleHTTPRequestHandler,
            directory=directory,
        )
        self._http = http.server.ThreadingHTTPServer(("", http_port), handler)
        threading.Thread(target=self._http.serve_forever, daemon=True).start()

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        self._ready.wait(5)

        print(f"Open this on the iPad:  http://{local_ip()}:{http_port}")

    # ---- the bits you call -------------------------------------------------

    def light(self, on=True):
        """Turn the light on or off and leave it there."""
        self._send({"light": bool(on)})

    def flash(self, seconds=0.2):
        """One pulse, then back to off."""
        self._send({"flash": round(seconds * 1000)})

    def blink(self, hz=2.0):
        """Blink continuously at hz. Pass 0 to stop."""
        self._send({"blink": float(hz)})

    def wait_for_send(self, timeout=None, discard_stale=True):
        """Block until Send is tapped and return the rating (None on timeout).

        discard_stale drops any taps that arrived before this call, so a
        double-tap on the previous trial can't satisfy the next one.
        """
        if discard_stale:
            self.clear_sends()
        try:
            return self._sends.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_sends(self):
        """Throw away any taps that have queued up but not been read."""
        while not self._sends.empty():
            try:
                self._sends.get_nowait()
            except queue.Empty:
                break

    def rating(self, value):
        """Move the slider from here, e.g. to reset to 3 between trials."""
        value = max(1, min(5, int(round(value))))
        self.slider = value
        self._send({"rating": value})

    def wait_for_ipad(self, timeout=None):
        """Block until the page connects. Returns True if it did."""
        return self._connected.wait(timeout)

    @property
    def connected(self):
        return bool(self._clients)

    def close(self):
        """Shut both servers down cleanly."""
        self._http.shutdown()
        if self._loop and self._stop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
            self._thread.join(timeout=3)

    # ---- plumbing ----------------------------------------------------------

    def _send(self, msg):
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(json.dumps(msg)), self._loop)

    async def _broadcast(self, text):
        for ws in list(self._clients):
            try:
                await ws.send(text)
            except Exception:
                self._clients.discard(ws)

    async def _handler(self, ws, *_):
        self._clients.add(ws)
        self._connected.set()
        print("iPad connected")
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if "slider" in msg:
                    self.slider = int(msg["slider"])
                    if self.on_slider:
                        self.on_slider(self.slider)

                if "submit" in msg:
                    value = int(msg["submit"])
                    self.slider = value
                    self.submitted = value
                    self._sends.put(value)
                    if self.on_send:
                        self.on_send(value)
        except Exception:
            pass
        finally:
            self._clients.discard(ws)
            if not self._clients:
                self._connected.clear()
            print("iPad disconnected")

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        self._stop = asyncio.Event()

        async def main():
            async with websockets.serve(self._handler, "", self._ws_port):
                self._ready.set()
                await self._stop.wait()
                for ws in list(self._clients):
                    await ws.close()

        self._loop.run_until_complete(main())
        self._loop.close()


class _QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


if __name__ == "__main__":
    import time

    p = Panel(quiet=True)
    print("Waiting for the iPad...")
    p.wait_for_ipad()

    # Demo: pick a rating, tap Send, and the light pulses that many times.
    try:
        while True:
            print("Waiting for Send...")
            value = p.wait_for_send()
            print(f"got {value}")
            for _ in range(value):
                p.flash(0.15)
                time.sleep(0.35)
            p.rating(3)                 # reset the scale for the next round
    except KeyboardInterrupt:
        p.light(False)
        p.close()
