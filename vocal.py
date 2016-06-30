''' 
Created on Jun 22, 2016

@author: georgid
'''
import essentia
from essentia.standard import PredominantPitchMelodia
# from essentia.standard import PredominantMelody

import mir_eval
import numpy as np
from Parameters import Parameters
from vocalVariance import extractVocalVar, extract_features
from sklearn.ensemble.forest import RandomForestClassifier
from matplotlib.pyplot import imshow
from math import floor
from numpy import math
from matplotlib import pyplot


def getTimeStamps(audioSamples,  feature_series):
    '''
    utility
    '''
    duration = essentia.standard.Duration()
    duration_ = duration(audioSamples)
    timestamps = np.arange(len(feature_series)) / float(len(feature_series)) * duration_ 
    timestamps += (float(Parameters.wSize / 2) / Parameters.fs)
    return timestamps

def extractMelody(audioSamples, wSize, hSize):
    fs = 44100
    vTol = 1.4


    # extract f0 using ESSENTIA
    input = essentia.array(audioSamples)
    pitchTracker = PredominantPitchMelodia(frameSize = wSize, hopSize = hSize, sampleRate = fs,
        voicingTolerance = vTol, voiceVibrato = False, filterIterations=10, 
        peakDistributionThreshold=0.9, guessUnvoiced=True)
    
#     pitchTracker = PredominantMelody(frameSize = wSize, hopSize = hSize, sampleRate = fs,
#         voicingTolerance = vTol, voiceVibrato = False, filterIterations=10, 
#         peakDistributionThreshold=0.9, guessUnvoiced=True)
    
    f0, pitchConf = pitchTracker(input)
    
    timestamps = getTimeStamps(audioSamples, f0)
    
    
    f0 = mir_eval.multipitch.frequencies_to_midi(f0)    
    return timestamps, np.array(f0)

def detect_vocal(audioSamples,  rfc, est_f0):
    
    
    # 2 vocal variance
    features_array = extract_features(audioSamples)
    
    voiced_flag = np.zeros(len(features_array)) 
    
#     vocal_var_array_T = features_array.T
#     imshow(vocal_var_array_T, aspect = 'auto', interpolation='none') 
    
    # prepare array indices
    resFactor = Parameters.hSize/Parameters.hSizePitch 
    aLen = min([len(features_array)*resFactor, len(est_f0)])
    features_array = features_array[0:floor(aLen / resFactor)]
    est_f0 = est_f0[0:aLen]

    for i in range (0,len(features_array)): 
        vv = features_array[i]
        midIdx = i*resFactor
        if est_f0[midIdx] > 0: ## estimate only for frames with middle frame from melodias being voiced
            voiced_flag[i] = rfc.predict(vv.reshape(1,-1))
            est_f0[midIdx - resFactor/2 -1: midIdx + resFactor/2+1] *= voiced_flag[i] # make range of f0-s zero is not voiced
    
# smoothing
#     TODO:
    return voiced_flag, est_f0



def resampleSeries(audioSamples, source_f0, dest_f0, source_voicing):
    '''
    utility
    '''


# resample estimated : from its original to referernce timing
    times = getTimeStamps(audioSamples, source_f0)
    times_new = getTimeStamps(audioSamples, dest_f0)
    
    res_est_f0, res_est_voicing = mir_eval.melody.resample_melody_series(times, source_f0, source_voicing, times_new)
    return res_est_f0, res_est_voicing, times_new



def eval_voicing(audioSamples, ref_f0,  est_f0 ):
    
    
    ref_freq, ref_voicing = mir_eval.melody.freq_to_voicing(ref_f0)
    est_freq, est_voicing = mir_eval.melody.freq_to_voicing(est_f0)

    res_est_f0, resampled_est_voicing, times_resampled = resampleSeries(audioSamples,  est_f0, ref_f0, est_voicing)
    
   
         
    mir_eval.melody.validate_voicing(ref_voicing, resampled_est_voicing)
    
#     (ref_v, ref_c, est_v, est_c) = mir_eval.melody.to_cent_voicing(ref_time,ref_freq, est_time, est_freq)
    recall, false_alarm = mir_eval.melody.voicing_measures(ref_voicing, resampled_est_voicing)
    
    return recall, false_alarm
    

 
    

