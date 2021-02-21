# Libraries
import pandas as pd
import os
import glob
import argparse
import numpy as np
#from datetime import datetime, timedelta
import datetime as dt
import random
# Read in relevant data files
samples_df = pd.read_csv('samples_short.csv')
ground_truth = pd.read_csv('ground_truth_short.csv')
perfect = pd.read_csv('perfect.csv')

directory_of_sounds = 'sounds/samples/'
os.path.isdir(directory_of_sounds)
len(os.listdir(directory_of_sounds + '/vi95kMQ65UeU7K1wae12D1GUeXd2')) # should be 22

#########################
# TASK DESCRIPTION
#########################

### The data ############
# You've been provided with 22 thirty-second sound files, spanning 11 minutes.
# The user of the device coughed 10 times during this 11 minute period.
# The exact timestamps of these coughs are in `ground_truth`
# `samples_df` has the mapping of the 30 second files, along with the time ("timestamp") at which time the 30 second
# recording started

### The task ############
# Write a function in python which takes a sound file as an argument
# and returns a dataframe of times (number of second since file start) at which a cough occurred.
# This is an event detection task in which it is far better to OVER-detect than
# to UNDER-detect. In regards to tuning prec/recall, you should consider
# a correctly detected cough to be worth 10 "points" and a "false positive"
# (ie, a "peak" which is not a cough) to be worth -1 points.
# The simpler, the better. No need to use Tensorflow, pre-trained models, or anything like that.
# Feel free to use libraries, but know that this is not a test of your modeling skills.

def detect_coughs(file = 'sounds/samples/vi95kMQ65UeU7K1wae12D1GUeXd2/sample-1613658921823.m4a'):
    # Replace the below random code with something meaningful which
    # generates a one-column dataframe with a column named "peak_start"

    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent



    AudioSegment.converter= 'C:\\Users\\MONSTER\\Desktop\\fmpeg\\ffmpeg-N-101185-g029e3c1c70-win64-gpl-shared-vulkan\\bin\\ffmpeg.exe'
    AudioSegment.ffmpeg = "C:\\Users\\MONSTER\\Desktop\\fmpeg\\ffmpeg-N-101185-g029e3c1c70-win64-gpl-shared-vulkan\\bin\\ffmpeg.exe"
    AudioSegment.ffprobe = "C:\\Users\\MONSTER\\Desktop\\fmpeg\\ffmpeg-N-101185-g029e3c1c70-win64-gpl-shared-vulkan\\bin\\ffprobe.exe"

    wav_filename = os.path.splitext(os.path.basename(file)[0]) + tuple(".wav")

    audio_segment = AudioSegment.from_file(file,format='m4a').export(out_f=wav_filename,format="wav")
    change_dbfs = -20.0 - audio_segment.dBFS
    normalized_sound = audio_segment.apply_gain(change_dbfs)
    nosilent_data = detect_nonsilent(normalized_sound,min_silence_len=500,silence_thresh=-20,seek_step=1)
    peak_start_time = []
    for chunks in nosilent_data:
        for chunk in chunks:
            peak_start_time.append(chunk[0])

    peaks = peak_start_time
    peaks.sort()
    print(peaks)
    out = pd.DataFrame(peaks,columns= ['peak_start'])                          # changed
    return out

# Run function on all sounds
sounds_dir = directory_of_sounds + 'vi95kMQ65UeU7K1wae12D1GUeXd2/'
all_sounds = os.listdir(sounds_dir)
out_list = []
for i in range(len(all_sounds)):                            #in folder open each sound file in order ,
                                                                            # then all the peak times save in final.csv as dataframe
    this_file = sounds_dir + all_sounds[i]                                   # By the way, save all the peak times file path named 'file' column
    this_result = detect_coughs(file = this_file)
    this_result['file'] = this_file
    out_list.append(this_result)
final = pd.concat(out_list)
final.to_csv('final.csv')

# Grade the approach
true_positives = 0
# Detect if coughs were correctly corrected
for i in range(len(perfect)):                                           #with all cough peak time (as a perfect dataframe),
    this_cough = perfect.iloc[i]                                            #mistake # pick each cough in order,
    same_file = final[final['file'] == this_cough['file']]                          #mistake? #check,is picked cough file path  same with final file path, is it true or not?
    # Get time differences
    same_file['time_diff'] = this_cough['peak_start'] - same_file['peak_start']
    keep = same_file[same_file['time_diff'] <= 0.4]                                 # to samefile dataframe as a timediff column
    keep = keep[keep['time_diff'] >= -0.4]                                          #to keep dataframe as timediff column
    if len(keep) > 0:
        print('Correctly found the cough at ', str(round(this_cough['peak_start'], 2)) + ' in ' + this_cough['file']) #peaktime and file path
        true_positives = true_positives + 1
    else:
        print('Missed the cough at ', str(round(this_cough['peak_start'], 2)) + ' in ' + this_cough['file']) #peaktime and file path
        pass
# Now measure false positives
false_positives = len(final) - true_positives
print('Detected ' + str(false_positives) + ' false positives')
# Calculate final score
final_score = (true_positives * 10) - false_positives
print('FINAL SCORE OF: ' + str(final_score))
