'''
Created on Jun 23, 2016

feature vocal variance

@author: georgid
'''


from essentia.standard import *
import math
import numpy as np
from matplotlib.pyplot import imshow, show
from Parameters import Parameters
# from cante.extrBarkBands import extrBarkBands

def extract_features(audioChunkSamples):
    
    mfccs_array = extractMFCCs(audioChunkSamples, Parameters.wSize, Parameters.hSize)
    
    vocal_var_array = extractVocalVar(mfccs_array, Parameters.hSize)
    
            #### extract voicing features
    # 1. bark bands
#     bb = extrBarkBands(audioChunkSamples, Parameters.wSize, Parameters.hSize)
#     bb = essentia.array(bb)
    
    # and plot
#         bb_T = bb.T
#         imshow(bb_T, aspect = 'auto', interpolation='none') 
    
    
    
    features = vocal_var_array
    if Parameters.WITH_MFCCS:
        features = np.hstack((mfccs_array, vocal_var_array))
        
    return features 

def extractSpecFluct():
    # step 1 getTribank spectrum
    
    # step 2 
    import numpy as np
    num_semitones = 5
    spectrogram = np.array([[1, 2, 0, 3, 4,2,1 ], [0, 1, 2, 0, 3,4,2], [2, 2, 1, 2, 0,3,4]])
    
    mid_idx = spectrogram.shape[1] - 1 / 2
    f_shift_infices = np.zeros(len(spectrogram))
    for i in range(len(spectrogram) - 1):
        corr_spec = np.correlate(spectrogram[i], spectrogram[i+1], 'same') # full computes reduntant values for us
           
        subsemitone_corr_spec = corr_spec[mid_idx-num_semitones : mid_idx+num_semitones + 1]
        
        idx_max_corr = np.argmax(corr_spec )
        idx_max_corr =  idx_max_corr  - (num_semitones +1) # normalize no_shifts = 0
        f_shift_infices[i+1] = idx_max_corr
    
    # step 3: extractFeatureBlockVariance
    

def extractMFCCs(audioSamples, _frameSize, _hopSize):

    # as suggested by BErnhard Lehner
    num_mfccs = 30
    
    ######## compute MFCCs
    w = Windowing(type = 'hann')
    spectrum = Spectrum()
    mfcc = MFCC(inputSize=_frameSize / 2 + 1, numberCoefficients=num_mfccs, numberBands=30)
    
    frames = list(FrameGenerator(audioSamples, frameSize = _frameSize, hopSize = _hopSize))
    mfccs_array = np.zeros( (len(frames), num_mfccs) )
    
    for i,frame in enumerate(frames):
        spec = spectrum(w(frame))
    
        mfcc_bands, mfcc_coeffs = mfcc( spec )
    #     highFrequencyBound=22100
        # take first 5 coeffs. no energy
        mfccs_array[i] = mfcc_coeffs
       
    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    
#     mfccs_T = essentia.array(mfccs_array).T
#     # and plot
#     imshow(mfccs_T, aspect = 'auto', interpolation='none')
#     show() # unnecessary if you started "ipython --pylab" 

    return mfccs_array


def extractVocalVar(mfccs_array, _frameSize ):
    '''
    vocal variance
    
    '''
    
    num_mfccs_var = 5
        
    
    mfccs_array = mfccs_array[:, 1:num_mfccs_var+1]
    
    fs = 44100
    num_mfccs= mfccs_array.shape[1]       
    num_frames = len(mfccs_array)
    
   
    # variance num frames
    numFrVar = int(math.floor(fs * Parameters.varianceLength / _frameSize))
    vocal_var_array = np.zeros(mfccs_array.shape)
    
    for i in range(0, num_frames):
        startIdx = max(0,i-numFrVar)
        endIdx = min( num_frames-1, i + numFrVar )
    
        # iterate over mfccs
        for coeff in range(num_mfccs):
            mfcc_slice = mfccs_array [ startIdx : endIdx + 1, coeff ]
            vocal_var_array[i, coeff] = np.var( mfcc_slice )
       
    # transpose to have it in a better shape
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    vocal_var_T = essentia.array(vocal_var_array).T
     
    # and plot
    imshow(vocal_var_T, aspect = 'auto', interpolation='none')  
    
    return vocal_var_array