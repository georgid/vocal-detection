'''
Created on Jul 1, 2016

@author: georgid
'''

class Parameters(object):
    '''
    classdocs
    '''
    
    iKalaURI = '/home/georgid/Documents/iKala/'
    
    # suggested by B. Lehner
    varianceLength = 1 # in sec
    
    wSize = 2048
    hSizePitch = 128
    hSize = wSize 
    
    WITH_VOCAL_DETECTION = 0
    visualize = 0
    
    nHarmonics = 30
    fs = 44100
    harmonicTreshold = -70
    
    WITH_MFCCS = 1
    
    if WITH_MFCCS:
        FEATURE_DIMENSION = 30 + 5
    else:
        FEATURE_DIMENSION = 5
    
    
    #### store classifier
    model_name  = 'rfc_vv'
    if WITH_MFCCS:
        model_name += '_mfcc'
    model_name += '.pkl'
    