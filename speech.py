import pyttsx3

_engine = None

#Takes a String as an input and has a computer voice say it out loud
def speak(text: str):
    global _engine
    if _engine is None:
        _engine = pyttsx3.init()
        #Set voice speed
        _engine.setProperty("rate", 150)
    _engine.say(text)
    _engine.runAndWait()

