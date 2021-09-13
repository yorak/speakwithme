#!/usr/bin/env python 

import argparse
import fcntl
import wave
import contextlib
import tempfile

from os import path, system, listdir, fsdecode
from multiprocessing import Pool, Process, Manager
from subprocess import Popen
from time import sleep, time
from random import randint, choice
from io import IOBase

import cv2
import numpy as np
import v4l2


#import re
#only_letters_and_spaces = re.compile('[^a-zA-Z ]')
#First parameter is the replacement, second parameter is your input string
#regex.sub('', 'ab3d*E')

try:
    from pocketsphinx.pocketsphinx import Decoder
    has_pocketsphinx = True
except:
    has_pocketsphinx = False

try:    
    from deepspeech import Model
    has_deepspeech = True
except:
    has_deepspeech = False
    

# Chatbot
try:
    from cobe import brain
    has_cobe = True
except:
    has_cobe = False

# DeepSpeech params
DS_BEAM_WIDTH = 500 #Bigger gets more accurate results
SCORER_ALPHA = 0.75
SCORER_BETA = 1.85

YAWN_FREQ = 25 # larger, blink faster, max 50
SPEECH_DURATION_OFFSET = 0.2 # seconds
    
# TODO: 
# -test mbrola-ee1: Estonian Male voice (9.2Mb)
# -calculate pitch with PDA of the input and then adjust (needed to understand kids)
# -remove utterances when terminated

BRAIN_DATABASE = r"chatterbot_data.brain"
MODELDIR = r"/usr/share/pocketsphinx/model"

class WordSaladBot:
    def __init__(self, word_file=r'/usr/share/dict/words'):
        self.known_words = open(word_file).read().splitlines()
    
    def reply(question, loop_ms):
        # ignore the question, just generate some words
        answer = ""
        word_count = randint(1, 10)
        for i in range(word_count):
            choice.append( choice(self.known_words)+" " )
        return answer

def TTS_worker(process_name, tasks, results):
    
    #system("espeak -v mb-hu1 -s 220 \"%s\" > /dev/null 2>&1" % str(phrase))
    #pico2wave -w=/tmp/test.wav -l en-US "<pitch level='75'> I think I can help" && 

    while True:
        phrase = tasks.get()
        
        temp_file_path = path.join( tempfile._get_default_tempdir(),
                                    next(tempfile._get_candidate_names())+".wav" )
        
        cmd = f'pico2wave -w={temp_file_path}'+\
              f' -l en-US \"<pitch level=\'85\'>{phrase}"'+\
              r' > /dev/null 2>&1'
        system(cmd)
        
        # Check length
        duration = 1.0
        with contextlib.closing(wave.open(temp_file_path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            
        results.put([temp_file_path, phrase, duration])

def record_phrase_to_file(phrasefile, levelp_start=0.03, levelp_end=0.04, pitch_change=1):
    # sox records an utterance
    cmd = "sox -t alsa default -r 16000 -c 1 \"%s\" silence 1 0.1 %d%% 1 1.0 %d%%" %\
        (phrasefile, int(levelp_start*100), int(levelp_end*100))+r" > /dev/null 2>&1"
    #cmd = f'sox -t alsa default -r 16000 -c 1 "{phrasefile}" silence 1 0.1 3% 1 0.1 5% vad gain -n > /dev/null 2>&1'
    system( cmd )

def recorder_worker(process_name, results):
    wid = 0
    while True:
        print("?")            
        loop_wav_file = path.join(path.dirname(path.realpath(__file__)),
                                  "utterance%d.wav" % wid)
        print("Recorded a file")
        wid+=1
        record_phrase_to_file(loop_wav_file, levelp_start=0.03, levelp_end=0.05)
        results.put(loop_wav_file)

#Pocketsphinx model
def ps_STT_worker(process_name, tasks, results):
    if not has_pocketsphinx:
        return
        
     # Create a decoder with certain model
    stt_config = Decoder.default_config()
    stt_config.set_string('-hmm', path.join(MODELDIR, r'en-us/en-us'))
    stt_config.set_string('-lm', path.join(MODELDIR, r'en-us/en-us.lm.bin'))
    stt_config.set_string('-dict', path.join(MODELDIR, r'en-us/cmudict-en-us.dict'))
    stt_config.set_string('-logfn', '/dev/null')
    stt_decoder = Decoder(stt_config)
    while True:
        recorded_wav_file = tasks.get()
        
        stt_decoder.start_utt()
        stream = open(recorded_wav_file, "rb")
        while True:
            buf = stream.read(1024)
            if not buf:
                break
            #TODO: call to avoid guide words?
            stt_decoder.process_raw(buf, False, False)
        stt_decoder.end_utt()
        words = [seg.word for seg in stt_decoder.seg()]
        recognized_phrase = " ".join(words)
        
        results.put(recognized_phrase)

def sp_STT_worker(process_name, tasks, results):
    if not has_deepspeech:
        return
        
    model = Model("../models/deepspeech-0.8.2-models.pbmm")
    model.enableExternalScorer("../models/deepspeech-0.8.2-models.scorer")
    model.setScorerAlphaBeta(SCORER_ALPHA, SCORER_BETA)
    model.setBeamWidth(DS_BEAM_WIDTH)
    
    while True:
        recorded_wav_file = tasks.get()
        w = wave.open(recorded_wav_file, 'r')
        frames = w.getnframes()
        buf = w.readframes(frames)
        data16 = np.frombuffer(buf, dtype=np.int16)
        
        recognized_phrase  = model.stt(data16)
        results.put(recognized_phrase)

#Send image data to v4l2loopback using python
#Remember to do sudo modprobe v4l2loopback first!
#Released under CC0 by Tim Sheerman-Chase, 2013
#
# The same in ffmpeg
# ffmpeg -re -loop 1 -i fine.png -filter_complex "nullsrc=size=640x480, format=yuv420p [base]; [0:v] format=pix_fmts=yuva420p, setpts=PTS-STARTPTS, scale=640x480 [left]; [base][left] overlay=shortest=1,format=yuv420p" -f v4l2 /dev/video0

def ConvertToYUYV(sizeimage, bytesperline, im):
    padding = 4096
    buff = np.zeros((sizeimage+padding, ), dtype=np.uint8)
    imgrey = im[:,:,0] * 0.299 + im[:,:,1] * 0.587 + im[:,:,2] * 0.114
    Pb = im[:,:,0] * -0.168736 + im[:,:,1] * -0.331264 + im[:,:,2] * 0.5
    Pr = im[:,:,0] * 0.5 + im[:,:,1] * -0.418688 + im[:,:,2] * -0.081312

    for y in range(imgrey.shape[0]):
        #Set lumenance
        cursor = y * bytesperline + padding
        for x in range(imgrey.shape[1]):
            try:
                buff[cursor] = imgrey[y, x]
            except IndexError:
                pass
            cursor += 2
    
        #Set color information for Cb
        cursor = y * bytesperline + padding
        for x in range(0, imgrey.shape[1], 2):
            try:
                buff[cursor+1] = 0.5 * (Pb[y, x] + Pb[y, x+1]) + 128
            except IndexError:
                pass
            cursor += 4

        #Set color information for Cr
        cursor = y * bytesperline + padding
        for x in range(0, imgrey.shape[1], 2):
            try:
                buff[cursor+3] = 0.5 * (Pr[y, x] + Pr[y, x+1]) + 128
            except IndexError:
                pass
            cursor += 4

    return buff.tobytes()

def load_image(image_path, for_target):
    if for_target: # v4l device handle
        return ConvertToYUYV(format_setting.fmt.pix.sizeimage,
                             format_setting.fmt.pix.bytesperline,
                             cv2.imread(image_path))
    else:
        return cv2.imread(image_path)
        
            
def main():
    parser = argparse.ArgumentParser(description='Mascot to discuss with.')
    parser.add_argument('--say', action='store', default=None, dest='say',
        help="Just say this one sentence")
    parser.add_argument('--subs', action='store_true', default=False, dest='subs',
        help="Show subtitles")
    parser.add_argument('-d', default='/dev/video1', action='store', dest='out_device',
        help="Draw the mascot to this v4l2 capture device.")
    args = parser.parse_args()

    say_just_one = args.say
    output_dev_name = args.out_device
    output_target = None
    width = 640
    height = 480
    
    output_to_v4l2_device = False
    if not path.exists(output_dev_name):
        print("Warning: device %s does not exist, output mascot to window"%output_dev_name)
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import matplotlib.patheffects as path_effects

    else:
        output_target = open(output_dev_name, 'wb')
        capability = v4l2.v4l2_capability()
        print("get capabilities result", (fcntl.ioctl(output_target, v4l2.VIDIOC_QUERYCAP, capability)))
        print("capabilities", hex(capability.capabilities))

        fmt = v4l2.V4L2_PIX_FMT_YUYV
        #fmt = V4L2_PIX_FMT_YVU420

        print("v4l2 driver: " + str(capability.driver))
        format_setting = v4l2.v4l2_format()
        format_setting.type = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT
        format_setting.fmt.pix.pixelformat = fmt
        format_setting.fmt.pix.width = width
        format_setting.fmt.pix.height = height
        format_setting.fmt.pix.field = v4l2.V4L2_FIELD_NONE
        format_setting.fmt.pix.bytesperline = width * 2
        format_setting.fmt.pix.sizeimage = width * height * 2
        format_setting.fmt.pix.colorspace = v4l2.V4L2_COLORSPACE_JPEG

        print("set format result", (fcntl.ioctl(output_target, v4l2.VIDIOC_S_FMT, format_setting)))
        
        output_to_v4l2_device = True
        #Note that format.fmt.pix.sizeimage and format.fmt.pix.bytesperline 
        #may have changed at this point
        

    #Create image buffers
    fine_buff = load_image(r"faces/fine.png", output_target)
    blink_buff = load_image(r"faces/blink.png", output_target)
    yawn_buff = load_image(r"faces/yawn.png", output_target)
    
    phonome_buff_map = {}
    for fn in listdir("phonomes"):
        filename = fsdecode(fn)
        if filename.endswith(".png"):
            mouth_file_path = path.join(r"phonomes", filename)
            mouth_buff = load_image(mouth_file_path, output_target)
            letters = filename.split(".")[0].split(",")
            for letter in letters:
                phonome_buff_map[letter] = mouth_buff
    
    manager = Manager()
    rec_results = manager.Queue()
    STT_results = manager.Queue()
    TTS_tasks = manager.Queue()
    TTS_results = manager.Queue()
    
    num_processes = 4
    pool = Pool(processes=num_processes)
    processes = []
    
    # Create the process, and connect it to the worker function
    rec_process = Process(target=recorder_worker,
                          args=("rec",rec_results))
    STT_process = Process(target=sp_STT_worker,
                          args=("STT",rec_results,STT_results))
    TTS_process = Process(target=TTS_worker,
                          args=("TTS",TTS_tasks,TTS_results))
                      
    # Add new process to the list of processes
    processes.append(rec_process)
    processes.append(STT_process)
    processes.append(TTS_process)
    
    # For one sentence mode, just start TTS
    for p in processes: 
        if not say_just_one or p==TTS_process:
            p.start()

    buff = fine_buff
    last_activity_ts = time()
    if has_cobe:
        bot = brain.Brain(BRAIN_DATABASE)
    else:
        bot = WordSaladBot()
    speaking_timings = []
    queued_reply_wav_file = ""
    reply_phrase = ""
    one_say_queued = False
    iteration = 0
    while True:
        if time()-last_activity_ts>3:
            if randint(0,50-max(45,YAWN_FREQ))==0:
                buff = blink_buff
                last_activity_ts = time()
        
        if time()-last_activity_ts>20:
            buff = yawn_buff
            last_activity_ts = time()
        
        if not speaking_timings:
            if buff is yawn_buff and time()-last_activity_ts>1.0:
                buff = fine_buff
                last_activity_ts = time()
            if buff is blink_buff and time()-last_activity_ts>0.5:
                buff = fine_buff
                last_activity_ts = time()
                
            if not TTS_results.empty():
                queued_reply_wav_file, speaking_phrase, duration = TTS_results.get()
                print("PLAYING BACK:", queued_reply_wav_file, "(%.2f s)"%duration)
                speaking_timings = []
                letter_duration = (duration-SPEECH_DURATION_OFFSET)/len(speaking_phrase)
                i = 0
                while i<len(speaking_phrase):
                    phonome = speaking_phrase[i]
                    if i+1<len(speaking_phrase):
                        two_letter_phonome = speaking_phrase[i]
                        if two_letter_phonome in phonome_buff_map:
                            phonome = two_letter_phonome
                            i+=1
                    speaking_timings.append( (phonome, time()+(i+1)*letter_duration) )
                    i+=1
                    
        # Note, may have been set above, 
        if speaking_timings:
            if queued_reply_wav_file:
                proc = Popen([f"aplay {queued_reply_wav_file} && rm {queued_reply_wav_file}"], shell=True,
                stdin=None, stdout=None, stderr=None, close_fds=True)
                #sleep(duration)
                queued_reply_wav_file = ""
            phonome, end_time = speaking_timings[0]
            if time()>end_time:
                speaking_timings.pop(0)
            if speaking_timings:
                last_activity_ts = time()
                if phonome in phonome_buff_map:
                    buff = phonome_buff_map[phonome]
                elif 'x' in phonome_buff_map:
                    # X is the fallback
                    buff = phonome_buff_map['x']
            else:
                buff = fine_buff
        else:
            reply_phrase = ""
        
        if say_just_one:
            # Wait 2s before starting to speak
            if not one_say_queued and time()-last_activity_ts>2.0:
                TTS_tasks.put(say_just_one)
                reply_phrase = say_just_one
                one_say_queued = True
            # After all has been said, wait 2s before ending the program
            elif one_say_queued and time()-last_activity_ts>2.0:
                break # Exit the "while True" main loop
        
        if not STT_results.empty():
            question = STT_results.get().strip()
            if not question:
                continue
                
            print("USER ASKED:", question)
            reply_phrase = bot.reply(question , loop_ms=0)
            print("BOT WILL REPLY:", reply_phrase)
            
            TTS_tasks.put(reply_phrase)
            
            
        if not output_target:
            plt.ion()
            output_target = plt.imshow(buff)
            if args.subs:
                txto = plt.text(width/2,height,reply_phrase,ha='center',
                    fontfamily='sans-serif', fontsize='x-large', fontweight='demibold')
                txto.set_path_effects([path_effects.withStroke(linewidth=2, foreground='lightgray')])
        if output_target is IOBase:
            print(type(output_target))
            output_target.write(buff)
            if args.subs:
                raise NotImplementedError("v4l2 output does not currently support subtitles")
        else:
            output_target.set_data(buff)
            if args.subs:
                txto.set_text(reply_phrase)
            plt.draw()
            plt.pause(0.01)
            plt.savefig("frame%04d.png"%iteration)
            
        sleep(1./15.)
        iteration+=1
    
    for p in processes:
        if p:
            p.terminate()
    #TODO: research how to cleanly shut down worker threads!
       
    # Creaders decoder object for streaming data.
    #with sr.Microphone() as source:
    #print("A moment of silence, please...")
    #r.adjust_for_ambient_noise(source)
    #print("Set minimum energy threshold to {}".format(r.energy_threshold))
    #print("Say something!")


if __name__=='__main__':
    main()
