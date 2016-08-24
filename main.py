'''
Created on Jun 29, 2016

@author: georgid
'''
import os
import essentia.standard
from vocalVariance import extractMFCCs, extractVocalVar
from matplotlib.pyplot import imshow, show
import numpy
from vocal import resampleSeries, extractMelody, detect_vocal, eval_voicing
from matplotlib import pyplot

from smstools.workspace.harmonicModel_function import extractHarmSpec, resynthesize,\
    hprModel_2
from Parameters import Parameters
import pickle
from math import floor
import numpy as np
import sys
import utilFunctions as UF
import tempfile
from eval_svs import getWavFiles
from smstools.software.models.hprModel import hprModel

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'smstools/software/models/'))
                


pathModel = os.path.join(os.path.dirname(os.path.realpath(__file__)), Parameters.model_name)


def writeCsv(fileURI, list_, withListOfRows=1):
    '''
    TODO: move to utilsLyrics
    '''
    from csv import writer
    fout = open(fileURI, 'wb')
    w = writer(fout)
    print 'writing to csv file {}...'.format(fileURI)
    for row in list_:
        if withListOfRows:
            w.writerow(row)
        else:
            tuple_note = [row.onsetTime, row.noteDuration]
            w.writerow(tuple_note)
    
    fout.close()

def readCsv(onsets_URI):
    from csv import reader
    onsets_ts = [] 
    with open(onsets_URI) as f:
            r = reader(f)
            for row in r:
                currTs = float( "{0:.6f}".format(float(row[0].strip())) )
#                 currDur = float( "{0:.2f}".format(float(row[1].strip())) )
                currMIDI = float(row[1].strip())
                onsets_ts.append((currTs, currMIDI))
    
    onsets_ts = numpy.array(onsets_ts)
    return onsets_ts

def doit(argv):
    '''
    Singing Voice Separation by Harmonic Modeling 
    https://drive.google.com/drive/u/0/folders/0B4bIMgQlCAuqR1RidkNUUXotaGs
    
    '''
    
    if len(argv) != 3:
            print ("usage: {}  <pathToAudio> <pathToOutputFolder>".format(argv[0]) )
            sys.exit();
     
    recall = 0; false_alarm = 0
            
    # load pre-trained classifier 
    rfc = pickle.load(open(pathModel,'rb' ) )   
    
    _filename = argv[1]
    output_path = argv[2]
    _filename_base = os.path.basename(_filename)[:-4]
#         fileWavURI = os.path.join(iKalaURI + '/Wavfile/mono/', _filename + '.wav')

    ##### 0.  stereo-to-mono        
    loader = essentia.standard.MonoLoader(filename= _filename    )
    audioSamples = loader()
    _filename = os.path.join(tempfile.mkdtemp() , _filename_base + '_mono.wav')
    monoWriter = essentia.standard.MonoWriter(filename=_filename)
    monoWriter(audioSamples)
    
    #### 0.5. extract f0
    timestamps, est_f0 = extractMelody(audioSamples, Parameters.wSize, Parameters.hSizePitch)
    est_freq_and_ts = zip(timestamps, est_f0)
    
    ####### read instead from melodia
    melodiaFileURI = os.path.join(output_path , _filename_base + '.melodia.csv')
    est_freq_and_ts = readCsv(melodiaFileURI)
    
        
    
#     outFileURI = os.path.join(output_path , _filename_base + '.pitch.csv')
#     writeCsv(outFileURI, est_freq_and_ts)
    
    #### 1. harmonic modeling + resynthesis
    outVocalURI = os.path.join(output_path , _filename_base + '_voice.wav')
    outbackGrURI = os.path.join(output_path, _filename_base + '_instr.wav')
    
    fileWavURI_resynth = os.path.join(tempfile.mkdtemp() , _filename_base + '_tmp.wav')
    if not os.path.isfile(fileWavURI_resynth):
        hfreq, hmag, hphase, fs, hopSizeMelodia, x, w, N = extractHarmSpec(_filename, est_freq_and_ts, nH=Parameters.nHarmonics)
        
        # compensate for last missing, why missing? 
        hfreq = np.vstack((hfreq,np.zeros(Parameters.nHarmonics)))
        hmag = np.vstack((hmag,np.zeros(Parameters.nHarmonics)))
        hphase = np.vstack((hphase,np.zeros(Parameters.nHarmonics)))       
              
        resynthesize(hfreq, hmag, hphase, fs, hopSizeMelodia, fileWavURI_resynth)

     
    ###### 2. detect vocal, extracting features from harmonic components
    voiced  = est_f0
    if Parameters.WITH_VOCAL_DETECTION:

        loader = essentia.standard.MonoLoader(filename= fileWavURI_resynth    )
        audioSamples = loader()
        voiced, est_f0 = detect_vocal(audioSamples, rfc, est_f0)
    
#     recall, false_alarm = eval_one_file_voicing(audioSamples, _filename_base, est_f0, voiced)

    
    
# #     # mark non-vocal segments as empty in harmonic spectrum
# ##  TODO: replace multipying by zero hmag, hfreq, hphase by est_f0 output of detect_voiced
#     if Parameters.WITH_VOCAL_DETECTION:
#         resFactor = Parameters.hSize/Parameters.hSizePitch 
#      
#         for i in range(1,len(voiced)-1): # skip fisrt, workaround for midIdx
#               
#             if not voiced[i]:
#                 midIdx = i*resFactor
#                 hfreq[midIdx - resFactor/2 -1: midIdx + resFactor/2+1] = np.zeros(Parameters.nHarmonics)
#                 hmag[midIdx - resFactor/2 -1: midIdx + resFactor/2+1] = np.zeros(Parameters.nHarmonics)
#                 hphase[midIdx - resFactor/2 -1: midIdx + resFactor/2+1] = np.zeros(Parameters.nHarmonics)
#      
#              
#  

#      3. resynthesize final result, with background masking
    yh, xr = hprModel_2(x, fs, w, N, Parameters.harmonicTreshold, Parameters.nHarmonics, hfreq, hmag, hphase, outVocalURI, outbackGrURI)
    return recall, false_alarm
    
    
def eval_one_file_voicing(audioSamples, _filename_base, est_f0, voiced):
    
    ##### eval voicing

    ref_MIDI_URI = os.path.join(Parameters.iKalaURI + '/PitchLabel/', _filename_base + '.pv')
    ref_MIDI = numpy.loadtxt(ref_MIDI_URI)
     
    # visualize contour with features
    ref_MIDI_res, voicing_res, times_res = resampleSeries(audioSamples, ref_MIDI, est_f0, ref_MIDI)
    pyplot.plot(times_res, ref_MIDI_res)
    pyplot.plot(times_res, est_f0)


#         
    recall, false_alarm  = eval_voicing(audioSamples, ref_MIDI, voiced)
    print 'recall total: ' + str(recall)
    print 'f. alarm total: ' + str(false_alarm)
    
    return recall, false_alarm
    


def doit_all(iKalaTestURI, output_URI, filenames_selected=None):
    
    # load pre-trained classifier 
    rfc = pickle.load(open(pathModel,'rb' ) )   
    
    monoURI  = iKalaTestURI + '/Wavfile/mono/'

    _filenames = getWavFiles(filenames_selected, monoURI)
            
    
    voicing_recall_total = 0
    voicing_false_alarm_total = 0
    
    for _filename in _filenames:
        recall, false_alarm =  doit(['dummy', _filename,  iKalaTestURI + output_URI])
        voicing_recall_total += recall
        voicing_false_alarm_total += false_alarm
    

    print 'recall total: ' + str(voicing_recall_total/len(_filenames))
    print 'f. alarm total: ' + str(voicing_false_alarm_total/len(_filenames))
        
    
    

if __name__ == '__main__':
    
#     doit(sys.argv)
#     
#     sys.exit()
    audio_URI = '/Users/joro/Documents/Phd/UPF/voxforge/myScripts/vocal-detection/smstools/sounds/vignesh.wav'
    audio_URI = '/Users/joro/Documents/Phd/UPF/voxforge/myScripts/vocal-detection/10161_chorus.wav'
    output_URI = '/Users/joro/Documents/Phd/UPF/voxforge/myScripts/vocal-detection/'
    
    doit(['dummy,', audio_URI, output_URI])
  
  #############################################################   doit all
    _filenames_selected = ['45416_verse.wav', '45412_chorus.wav', '54247_verse.wav', '10161_chorus.wav', '10161_verse.wav', '10170_chorus.wav',  '31113_chorus.wav',   '45412_verse.wav']

    _filenames_selected = ['54247_verse.wav',  '10170_chorus.wav']

    output_URI = '/Wavfile_resynth_norm'
    doit_all(Parameters.iKalaURI, output_URI, _filenames_selected)
    
#     eval_all(Parameters.iKalaURI, _filenames)
    
    
    
    
    