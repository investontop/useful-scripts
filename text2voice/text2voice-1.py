# Install pyttsx3: pip install pyttsx3

import pyttsx3

def list_voices():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for index, voice in enumerate(voices):
        print(f"Voice {index}: {voice.name} ({voice.id})")

def text_to_speech(text, voice_index=0):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[voice_index].id)  # Select voice by index
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    print("Available voices:")
    list_voices()
    text = "Hello! This is a natural-sounding text-to-speech conversion."
    text_to_speech(text, voice_index=1)  # Change index to try different voices