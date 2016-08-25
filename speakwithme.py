import speech_recognition as sr
from cobe import brain
from os import path
import os

# TODO: 
# -test mbrola-ee1: Estonian Male voice (9.2Mb)
# -calculate pitch with PDA of the input and then adjust (needed to understand kids)
# -remove utterances when terminated

WAV_FILE = path.join(path.dirname(path.realpath(__file__)), "utterance.wav")
LANG = "fi-FI"
# Note: only 50 queries / day / key
GOOGLE_API_KEY = you_have_to_create_a_Google_API_key_here # eg. "AIzaSyAEM7z2ePcoJnyUl42o9ZJ0_EZMwBgKC2A"
BRAIN_DATABASE = r"keskustelija.brain"

def finnish_text_to_speech(phrase):
    os.system("espeak -v mb-hu1 -s 220 \"%s\" > /dev/null 2>&1" % str(phrase))

def record_phrase_to_file(phrasefile, levelp_start=0.03, levelp_end=0.05, pitch_change=1):
    # sox records an utterance
    cmd = "sox -t alsa default \"%s\" silence 1 0.1 %d%% 1 0.1 %d%% pitch %d > /dev/null 2>&1" %\
        (phrasefile, int(levelp_start*100), int(levelp_end*100), pitch_change)
    os.system( cmd )


# init brain (use "cobe init", "cobe console" to teach your own )
bot = brain.Brain(BRAIN_DATABASE )

# init recognizer 
r = sr.Recognizer()

#with sr.Microphone() as source:
#print("A moment of silence, please...")
#r.adjust_for_ambient_noise(source)
#print("Set minimum energy threshold to {}".format(r.energy_threshold))
#print("Say something!")

wid = 0
while True:
    print("?")            
    loop_wav_file = WAV_FILE.replace("utterance.wav", "utterance%d.wav" % wid)
    wid+=1
    #audio = r.listen(source)
    record_phrase_to_file(loop_wav_file, levelp_start=0.03, levelp_end=0.05)
    
    with sr.WavFile(loop_wav_file) as source:
        audio = r.record(source) # read the entire WAV file
    
    # recognize speech using Google Speech Recognition
    try:
        recognized_phrase = r.recognize_google(audio, language = LANG, key=GOOGLE_API_KEY)
        print("(Google Speech Recognition thinks you said \"" + recognized_phrase +"\")")
        reply_phrase = bot.reply(recognized_phrase , loop_ms=0)
        print(reply_phrase)
        finnish_text_to_speech(reply_phrase)
        
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
