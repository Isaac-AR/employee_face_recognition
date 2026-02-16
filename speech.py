# speech.py (OLD. Keeping this in case I successfuly implement multithreading)
import pyttsx3
import ctypes
from multiprocessing import Process, Queue
import win32com.client

speaker = win32com.client.Dispatch("SAPI.SpVoice")
#speaker.Speak("Hello, this is SAPI speaking.")

def _tts_worker(q: Queue, rate: int):
    # Initialize COM inside this process
    ctypes.windll.ole32.CoInitialize(None)
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", rate)

        while True:
            text = q.get()
            if text is None:
                break

            print("[TTS]", text)  # keep for debugging
            engine.say(text)
            engine.runAndWait()

        engine.stop()
    finally:
        ctypes.windll.ole32.CoUninitialize()

class Speech:
    def __init__(self, rate: int = 150):
        self._q = Queue()
        self._p = Process(target=_tts_worker, args=(self._q, rate), daemon=True)
        self._p.start()

    def say(self, text: str):
        self._q.put(text)

    def close(self):
        self._q.put(None)
        # Optional: give it a moment to shut down cleanly
        try:
            self._p.join(timeout=2)
        except Exception:
            pass
