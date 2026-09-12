"""
voice_alert.py – Text-to-speech fall alerts using pyttsx3.

Works completely offline — no internet or API key needed.
Uses Windows SAPI voice engine (built into Windows).

Usage:
    voice = VoiceAlert()
    voice.speak_fall("Ward 1")          # "Fall detected in Ward 1"
    voice.speak_recovery("Ward 2")      # "Patient recovered in Ward 2"
    voice.speak_custom("Check room 3")  # any custom message
"""

import threading
import logging
import time

log = logging.getLogger(__name__)

# Try to import pyttsx3 — auto-installed by run.py if missing
try:
    import pyttsx3
    _TTS_AVAILABLE = True
except ImportError:
    _TTS_AVAILABLE = False
    log.warning("pyttsx3 not installed — voice alerts disabled. "
                "Run: pip install pyttsx3")


class VoiceAlert:
    """
    Thread-safe text-to-speech alert system.
    Uses a background thread so speech never blocks detection.
    Queues messages — if already speaking, waits until done.
    """

    def __init__(self, rate: int = 150, volume: float = 1.0):
        self._rate    = rate
        self._volume  = volume
        self._queue   = []
        self._lock    = threading.Lock()
        self._speaking = False
        self._enabled  = _TTS_AVAILABLE
        self._cooldowns: dict = {}   # message → last spoken time
        self._cooldown_s = 10.0     # don't repeat same message within 10s

        if not _TTS_AVAILABLE:
            log.warning("Voice alerts disabled — pyttsx3 not available")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self):
        self._enabled = _TTS_AVAILABLE

    def disable(self):
        self._enabled = False

    # ── Public speak methods ──────────────────────────────────

    def speak_fall(self, location: str = "", person_id: int = None):
        """Announce a fall detection."""
        if person_id is not None:
            msg = f"Warning! Fall detected. Person {person_id}"
            if location:
                msg += f" in {location}"
        else:
            msg = f"Warning! Fall detected"
            if location:
                msg += f" in {location}"
        self._enqueue(msg, priority=True)

    def speak_recovery(self, location: str = ""):
        """Announce that a person has recovered."""
        msg = f"Patient recovered"
        if location:
            msg += f" in {location}"
        self._enqueue(msg)

    def speak_camera_offline(self, location: str = ""):
        """Announce that a camera has gone offline."""
        msg = f"Camera offline"
        if location:
            msg += f" — {location}"
        self._enqueue(msg)

    def speak_custom(self, message: str):
        """Speak any custom message."""
        self._enqueue(message)

    # ── Internal ──────────────────────────────────────────────

    def _enqueue(self, message: str, priority: bool = False):
        if not self._enabled:
            return

        # Cooldown check — don't repeat same message too quickly
        now = time.time()
        last = self._cooldowns.get(message, 0)
        if now - last < self._cooldown_s:
            return
        self._cooldowns[message] = now

        with self._lock:
            if priority:
                self._queue.insert(0, message)
            else:
                self._queue.append(message)

        if not self._speaking:
            thread = threading.Thread(
                target=self._speak_worker, daemon=True, name="FG-Voice"
            )
            thread.start()

    def _speak_worker(self):
        self._speaking = True
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate',   self._rate)
            engine.setProperty('volume', self._volume)

            # Use a clear voice if available
            voices = engine.getProperty('voices')
            for v in voices:
                if 'english' in v.name.lower() or 'zira' in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break

            while True:
                with self._lock:
                    if not self._queue:
                        break
                    msg = self._queue.pop(0)
                log.info("Voice alert: %s", msg)
                engine.say(msg)
                engine.runAndWait()
                time.sleep(0.3)

        except Exception as e:
            log.error("Voice alert error: %s", e)
        finally:
            self._speaking = False
