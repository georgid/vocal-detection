'''
Created on Jul 1, 2016

@author: georgid
'''
import os
import numpy as np
import essentia.standard
import mir_eval
from vocal import getTimeStamps
from Parameters import Parameters
from vocalVariance import extractVocalVar, extractMFCCs, extract_features
from sklearn.ensemble.forest import RandomForestClassifier
import pickle
from smstools.workspace.harmonicModel_function import extractHarmSpec,\
    resynthesize

def train(iKalaTrainURI):
    _filenames = []
    monoURI  = iKalaTrainURI + '/Wavfile/'
    for file in os.listdir(monoURI):
        if file.endswith(".wav"):
            _filenames.append( os.path.join(monoURI, file) )

            
    
    vocal_frames_all = np.empty((0,Parameters.FEATURE_DIMENSION))
    non_vocal_frames_all =  np.empty((0,Parameters.FEATURE_DIMENSION))
    
    for i, _filename in enumerate(_filenames):
        print 'on training file ' + str(i)
         
        _filename_base = os.path.basename(_filename)[:-4]
#         fileWavURI = os.path.join(iKalaURI + '/Wavfile/mono/', _filename + '.wav')
        ref_MIDI_URI = os.path.join(iKalaTrainURI + '/PitchLabel/', _filename_base + '_resampled.pv')
        if not os.path.isfile(ref_MIDI_URI): continue
        ref_MIDI = np.loadtxt(ref_MIDI_URI)
        
        for_vocal = 1
        fileWavURI_resynth = resynth_vocal(_filename, ref_MIDI)
        vocal_frames = get_class_frames(fileWavURI_resynth, ref_MIDI, for_vocal)
        vocal_frames_all = np.append(vocal_frames_all, vocal_frames, axis=0)
        for_vocal = 0
        non_vocal_frames = get_class_frames(_filename, ref_MIDI, for_vocal)
        non_vocal_frames_all = np.append(non_vocal_frames_all, non_vocal_frames, axis=0)

    
    rfc = train_classifier(vocal_frames_all, non_vocal_frames_all)
    return rfc
 



def get_class_frames(_filename, ref_MIDI, forVocal):
    
    '''
    get frames for one class
    '''
    
    sampleRate = 44100
    if forVocal: downmix = 'right'
    else: downmix = 'left'
    loader = essentia.standard.MonoLoader(filename = _filename, sampleRate = sampleRate, downmix = downmix)
    audioSamples = loader()
    
    ######## get labels 
    ref_freq, ref_voicing = mir_eval.melody.freq_to_voicing(ref_MIDI)
    if not forVocal:
        ref_voicing = ref_voicing = np.logical_not(ref_voicing)
    
    ts = getTimeStamps(audioSamples, ref_voicing)
    
    ########## init with 5 dimensions
    feature_bag = np.empty((0,Parameters.FEATURE_DIMENSION))
    
    ####### go vocal segment by vocal segment
    i = 0
    while(i<len(ref_voicing)):
        while i<len(ref_voicing) and not ref_voicing[i] : # seek for a first vocal frame
            i+=1
        
        if i==len(ref_voicing): # no segment found
                break
        beginTs = ts[i] # set begin ts
        
        while i<len(ref_voicing) and ref_voicing[i]:
            i+=1
        endTs = ts[i-1] # set end ts
        
        if endTs - beginTs < 2 * Parameters.varianceLength: #  too small voiced segment so dont use for training
            continue
    # for each forVocal segment extract feaures
        audioChunkSamples = audioSamples[beginTs*sampleRate : endTs*sampleRate]
        features = extract_features(audioChunkSamples)
        
#         monoWriter = essentia.standard.MonoWriter(filename='/tmp/test.wav')
#         monoWriter(audioChunkSamples)
    # put them in a bag 
        feature_bag = np.append(feature_bag, features, axis = 0) 
    
    # train
    return feature_bag

 
    
def train_classifier(vocal_frames, non_vocal_frames): 
    
    frames = np.append(vocal_frames, non_vocal_frames, axis = 0)
    
    labels_vocal = np.ones(vocal_frames.shape[0])
    labels_non_vocal = np.zeros(non_vocal_frames.shape[0])
    
    labels = np.append(labels_vocal, labels_non_vocal, axis = 0)

    
    rfc= RandomForestClassifier(n_estimators=100, max_depth=None)
    rfc.fit(frames, labels)
    
    return rfc

def midi_to_freq(midis):
    # mus20.py
# Midi numbers and frequencies

    g = 2.0**(1./12.)

    a = [440.*g**(midi-69.0) for midi in midis]
    return np.array( a)

def resynth_vocal(_filename, ref_MIDI  ):
    '''   do resyntheis with reference vocal frequencies. 
    used to resynthesize vocal part for training
    '''        

    sampleRate = 44100
    downmix = 'right' # vocal
    
    _filename_base = os.path.basename(_filename)[:-4]
    loader = essentia.standard.MonoLoader(filename = _filename, sampleRate = sampleRate, downmix = downmix)
    audioSamples = loader()
    _filename = os.path.join(iKalaURI + '/Wavfile/mono/vocal/', _filename_base + '.wav')
    monoWriter = essentia.standard.MonoWriter(filename=_filename)
    monoWriter(audioSamples)
    
    freq_ref =  midi_to_freq(ref_MIDI)
    ts = getTimeStamps(audioSamples, freq_ref)
    freq_ref_and_ts = zip(ts, freq_ref)
    
    fileWavURI_resynth = os.path.join(iKalaURI + '/Wavfile/mono/ref_vocal_resynth', _filename_base + '.wav')
    if not os.path.isfile(fileWavURI_resynth):
            hfreq, hmag, hphase, fs, hopSizeMelodia, inputAudioFromTsToTs = extractHarmSpec(_filename, freq_ref_and_ts)
            resynthesize(hfreq, hmag, hphase, fs, hopSizeMelodia, fileWavURI_resynth)
    return fileWavURI_resynth
    
if __name__ == '__main__':
    
    iKalaURI = '/home/georgid/Documents/iKala/'

    rfc = train(iKalaURI)
    for tree in rfc.estimators_:
        importances = tree.feature_importances_
        print np.argmax(importances)
    

    
    pickle.dump(rfc, open( Parameters.model_name, "wb" ) )
        
    