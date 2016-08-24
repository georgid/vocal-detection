'''
Created on Aug 23, 2016

@author: joro
'''
import os
import sys
# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../smstools/software/models_interface/dftModel_function'))

from main import readCsv, writeCsv
import tempfile
from Parameters import Parameters
import numpy as np
from smstools.workspace.harmonicModel_function import extractHarmSpec
import essentia.standard

if __name__ == '__main__':
    
    ref_MIDI_URI = '10161_chorus.pv'
    curdir = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(curdir , 'output')
    _filename = '10161_chorus.wav'
    
    
    _filename_base = os.path.basename(_filename)[:-4]
    ##### 0.  stereo-to-mono        
    loader = essentia.standard.MonoLoader(filename= _filename    )
    audioSamples = loader()
    _filename = os.path.join(tempfile.mkdtemp() , _filename_base + '_mono.wav')
    monoWriter = essentia.standard.MonoWriter(filename=_filename)
    monoWriter(audioSamples)
    
    
    ####### read instead from melodia
    melodiaFileURI = os.path.join(curdir , _filename_base + '.melodia.csv')
    est_freq_and_ts = readCsv(melodiaFileURI)
    
    
    ####### simulate non-vocal for file  10161_chorus
    est_freq_and_ts_only_vocal =  []
    for i in range(len(est_freq_and_ts)):
        if est_freq_and_ts[i][0] < 0.7 or  3.82 < est_freq_and_ts[i][0] < 4.97:
            est_freq_and_ts[i] = (est_freq_and_ts[i][0],0)
        est_freq_and_ts_only_vocal.append(est_freq_and_ts[i])
        
        
    fileWavURI_resynth = os.path.join(tempfile.mkdtemp() , _filename_base + '_tmp.wav')
    if not os.path.isfile(fileWavURI_resynth):
        hfreq, hmag, hphase, fs, hopSizeMelodia, x, w, N = extractHarmSpec(_filename, est_freq_and_ts_only_vocal, nH=Parameters.nHarmonics)
         
        # compensate for last missing, why missing? 
#         hfreq = np.vstack((hfreq,np.zeros(Parameters.nHarmonics)))
#         hmag = np.vstack((hmag,np.zeros(Parameters.nHarmonics)))
#         hphase = np.vstack((hphase,np.zeros(Parameters.nHarmonics)))
           
        timestamps = est_freq_and_ts[:,0]
        for i in range(5):
            harm_series = hfreq[:,i]
            if len(timestamps) != len(harm_series):
                sys.exit('not equal size harm series and pitch')
            est_partial_and_ts  = zip(timestamps, harm_series)
            outFileURI = os.path.join(output_path , _filename_base + '._' +  str(i) + '_pitch_onlyVocal.csv')
            writeCsv(outFileURI, est_partial_and_ts)        