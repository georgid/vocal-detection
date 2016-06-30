'''
Created on Jul 4, 2016

@author: georgid
'''
from Parameters import Parameters
import os
import essentia.standard
import numpy
import mir_eval
from numpy.linalg.linalg import norm
from os.path import basename
from numpy import vstack

def eval_all(iKala_URI, filenames_selected=None):
    wavfiles_URI = iKala_URI + '/Wavfile/'
    _filenames = getWavFiles(filenames_selected, wavfiles_URI)
    
    NSDR_total = 0
    SIR_total = 0
    SAR_total = 0
    
    for filename in _filenames:
        filename = basename(filename)
        [NSDR, SIR, SAR ] = eval_svs_main_(filename)
        NSDR_total += NSDR
        SIR_total += SIR
        SAR_total += SAR


def eval_svs_main_(filename):
    
    
    ######## reference
    ref_iKalaURI= Parameters.iKalaURI + '/Wavfile/'
    ref_fileURI=os.path.join(ref_iKalaURI, filename)
    
    loader = essentia.standard.MonoLoader(filename= ref_fileURI, downmix='left'    )
    trueKaraoke = loader()
    
    loader = essentia.standard.MonoLoader(filename= ref_fileURI, downmix='right'    )
    trueVoice = loader()
    
    ref_iKalaURI= Parameters.iKalaURI + '/Wavfile/mono/'
    ref_file_mono_URI=os.path.join(ref_iKalaURI, filename)
    
    loader = essentia.standard.MonoLoader(filename= ref_file_mono_URI )
    trueMixed = loader()

    
    ######## estimated
    estim_iKalaURI= Parameters.iKalaURI + '/Wavfile_resynth/'
    
    estim_fileURI=os.path.join(estim_iKalaURI,filename)
    
    loader = essentia.standard.MonoLoader(filename= estim_fileURI  )
    estimatedVoice  = loader()
    
    estimatedVoice_resized= numpy.zeros((trueMixed.shape))
    
    diff_length=len(trueMixed) - len(estimatedVoice)
    estimatedVoice_resized[diff_length:]=estimatedVoice
    
    estimatedKaraoke = trueMixed - estimatedVoice_resized
    norm_est = norm(estimatedVoice_resized + estimatedKaraoke)
    norm_est = 229.0
    norm_true = norm(trueVoice + trueKaraoke)
    a = vstack((estimatedVoice_resized,estimatedKaraoke)) 
    b = vstack((trueVoice,trueKaraoke))
    (SDR,SIR,SAR, perm) = mir_eval.separation.bss_eval_sources( a / norm_est,  b / norm_true  )
    (NSDR,NSIR,NSAR, perm) = mir_eval.separation.bss_eval_sources( \
        [trueMixed,trueMixed] / norm(trueMixed + trueMixed),\
        [trueVoice,trueKaraoke] / norm(trueVoice + trueKaraoke))
    NSDR=SDR - NSDR
    
    return NSDR,SIR,SAR


def getWavFiles(filenames_selected, monoURI):
    _filenames = []
    if filenames_selected == None:
        for file in os.listdir(monoURI):
            if file.endswith(".wav"):
                _filenames.append(os.path.join(monoURI, file))
    
    else:
        for file in filenames_selected:
            _filenames.append(os.path.join(monoURI, file)) # pre-given set of files
    
    return _filenames