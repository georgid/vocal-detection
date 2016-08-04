# function to call the extractHarmSpec analysis/synthesis functions in software/models/harmonicModel.py

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import get_window
import logging
from Parameters import Parameters
import math

from scipy.signal import blackmanharris, triang
from scipy.fftpack import fft, ifft

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import utilFunctions as UF
import sineModel as SM
import harmonicModel as HM

# import software.models.utilFunctions as UF
# import software.models.sineModel as SM
# import software.models.harmonicModel as HM


# inputFile = '../sounds/vignesh.wav'

# def extractHarmSpec(inputFile='../sounds/vignesh.wav', window='blackman', M=1201, N=2048, t=-90, 
# 	minSineDur=0.1, nH=100, minf0=130, maxf0=300, f0et=7, harmDevSlope=0.01):

# increasing the threshold means discarding more  peaks and selecting less 	
def extractHarmSpec( inputFile, f0FreqsRaw, fromTs=-1, toTs=-1, t=Parameters.harmonicTreshold, window='blackman',  M=Parameters.wSize - 1, N=2048 , 
	minSineDur=0.0, nH=Parameters.nHarmonics, harmDevSlope=0.02):
	"""
	Analysis and synthesis using the harmonic models_makam
	inputFile: input sound file (monophonic with sampling rate of 44100)
	window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
	M: analysis window size; N: fft size (power of two, bigger or equal than M)
	t: magnitude threshold of spectral peaks; minSineDur: minimum duration of sinusoidal tracks
	nH: maximum number of harmonics; minf0: minimum fundamental frequency in sound
	maxf0: maximum fundamental frequency in sound; f0et: maximum error accepted in f0 detection algorithm                                                                                            
	harmDevSlope: allowed deviation of harmonic tracks, higher harmonics could have higher allowed deviation
	"""
	
	# if not monophonic, convert to monophonic
	
	
	# read input sound
	(fs, x) = UF.wavread(inputFile)
	

# 	hopsize
	hopSizeMelodia = int( round( (float(f0FreqsRaw[1][0])  - float(f0FreqsRaw[0][0]) ) * fs ) )

	### get indices in melodia
	toTs = float(toTs)
 	if fromTs==-1 and toTs==-1:
 		logging.debug("fromTs and toTs not defined. extracting whole recording")
 		fromTs=0; toTs=f0FreqsRaw[-1][0]
    		
	
	
	finalPitchSeriesTs = f0FreqsRaw[-1][0]
	if finalPitchSeriesTs < fromTs or finalPitchSeriesTs < toTs:
		sys.exit('pitch series have final time= {} and requested fromTs= {} and toTs={}'.format(finalPitchSeriesTs, fromTs, toTs) )
	idx = 0
	while fromTs > float(f0FreqsRaw[idx][0]):
		idx += 1
	
	firstTs = float(f0FreqsRaw[idx][0])
	pinFirst  = round (firstTs * fs)
		
	f0Series = []
	while  idx < len(f0FreqsRaw) and float(f0FreqsRaw[idx][0]) <= toTs:
		f0Series.append(float(f0FreqsRaw[idx][1])) 
		idx += 1
	lastTs = float(f0FreqsRaw[idx-1][0])
	pinLast = round (lastTs * fs)
	

	
	# discard ts-s
# 	for foFreqRaw in range() f0FreqsRaw:
# 		f0Series.append(float(foFreqRaw[1])) 
	
	# size of fft used in synthesis

	
	# hop size (has to be 1/4 of Ns)
# 	H = 128


	# compute analysis window
	w = get_window(window, M)

	# detect harmonics of input sound
	hfreq, hmag, hphase = HM.harmonicModelAnal_2(x, fs, w, N, hopSizeMelodia, pinFirst, t, nH, f0Series, harmDevSlope, minSineDur)

# w/o melodia and with resynthesis 
# 	minf0=130
# 	maxf0=300
# 	f0et=7
# 	HM.harmonicModel(x, fs, w, N, t, nH, minf0, maxf0, f0et)

	
# 	return hfreq, hmag, hphase, fs, hopSizeMelodia, x[pinFirst:pinLast]
	return hfreq, hmag, hphase, fs, hopSizeMelodia, x, w, N
	
	
def resynthesize(hfreq, hmag, hphase, fs, hopSizeMelodia, URIOutputFile):
	''' synthesize the harmonics
	'''
# 	Ns = 512
	Ns = 4 * hopSizeMelodia

	y = SM.sineModelSynth(hfreq, hmag, hphase, Ns, hopSizeMelodia, fs)  

	# output sound file (monophonic with sampling rate of 44100)
# 	URIOutputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_harmonicModel.wav'
	
	
	# write the sound resulting from harmonic analysis
	UF.wavwrite(y, fs, URIOutputFile)
	print 'written file ' + URIOutputFile
	
	return y


def hprModel_2(x, fs, w, N, t, nH, hfreq, hmag, hphase, outVocalURI, outbackGrURI):
	"""
	Analysis/synthesis of a sound using the harmonic plus residual model
	x: input sound, fs: sampling rate, w: analysis window, 
	N: FFT size (minimum 512), t: threshold in negative dB, 
	nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
	maxf0: maximim f0 frequency in Hz, 
	f0et: error threshold in the f0 detection (ex: 5),
	maxhd: max. relative deviation in harmonic detection (ex: .2)
	returns y: output sound, yh: harmonic component, xr: residual component
	"""

	hN = N/2                                                      # size of positive spectrum
	hM1 = int(math.floor((w.size+1)/2))                           # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                               # half analysis window size by floor
	Ns = 512                                                      # FFT size for synthesis (even)
	H = Ns/4                                                      # Hop size used for analysis and synthesis
	hNs = Ns/2      
	pin = max(hNs, hM1)                                           # initialize sound pointer in middle of analysis window          
	pend = x.size - max(hNs, hM1)                                 # last sample to start a frame
	fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
	yhw = np.zeros(Ns)                                            # initialize output sound frame
	xrw = np.zeros(Ns)                                            # initialize output sound frame
	yh = np.zeros(x.size)                                         # initialize output array
	xr = np.zeros(x.size)                                         # initialize output array
	w = w / sum(w)                                                # normalize analysis window
	sw = np.zeros(Ns)     
	ow = triang(2*H)                                              # overlapping window
	sw[hNs-H:hNs+H] = ow      
	bh = blackmanharris(Ns)                                       # synthesis window
	bh = bh / sum(bh)                                             # normalize synthesis window
	wr = bh                                                       # window for residual
	sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
	hfreqp = []
	f0t = 0
	i = 0
	while pin<pend and i < len(hfreq):  
	#-----analysis-----             

		hfreqp = hfreq
		ri = pin-hNs-1                                             # input sound pointer for residual analysis
		xw2 = x[ri:ri+Ns]*wr                                       # window the input sound                     
		fftbuffer = np.zeros(Ns)                                   # reset buffer
		fftbuffer[:hNs] = xw2[hNs:]                                # zero-phase window in fftbuffer
		fftbuffer[hNs:] = xw2[:hNs]                     
		X2 = fft(fftbuffer)                                        # compute FFT of input signal for residual analysis
		
		#-----synthesis-----
		Yh = UF.genSpecSines(hfreq[i,:], hmag[i,:], hphase[i,:], Ns, fs)          # generate sines
		
		# soft masking
		Yh, Xr = softMask(X2, Yh, i)
		
		fftbuffer = np.zeros(Ns)
		fftbuffer = np.real(ifft(Yh))                              # inverse FFT of harmonic spectrum
		yhw[:hNs-1] = fftbuffer[hNs+1:]                            # undo zero-phase window
		yhw[hNs-1:] = fftbuffer[:hNs+1] 
		
		fftbuffer = np.real(ifft(Xr))                              # inverse FFT of residual spectrum
		xrw[:hNs-1] = fftbuffer[hNs+1:]                            # undo zero-phase window
		xrw[hNs-1:] = fftbuffer[:hNs+1]
		yh[ri:ri+Ns] += sw*yhw                                     # overlap-add for sines
		xr[ri:ri+Ns] += sw*xrw                                     # overlap-add for residual
		pin += H
		i += 1                                                    # advance sound pointer
                                                  # sum of harmonic and residual components
	UF.wavwrite(yh, fs, outVocalURI)
	print 'written file ' + outVocalURI 
		
	UF.wavwrite(xr, fs, outbackGrURI)
	print 'written file ' + outbackGrURI 
	
	return yh, xr


def softMask(X, V,i): 
	''' 
	X - original spectrum 
	V - vocal estim 
	'''
	
	div = np.divide(abs(V), abs(X))
	indices_clipping = np.where(div>1)[0]
	if len(indices_clipping) > 1:
		
		print 'it happens {} times at time {}'.format(len(indices_clipping),i)
 	maskVocal  = np.minimum(div,1)
	V_est = np.multiply(X, maskVocal )
	K_est = np.multiply(X, 1-maskVocal)
	return V_est, K_est
	
	
# 	##########################
# 	## plotting of harmonic spectrum
# 	
def visualizeHarmSp(x, y, hopSizeMelodia ):
	# create figure to show plots
	plt.figure(figsize=(12, 9))
 
	# frequency range to plot
	maxplotfreq = 10000.0
 
	# plot the input sound
	plt.subplot(3,1,1)
	plt.plot(np.arange(x.size)/float(fs), x)
	plt.axis([0, x.size/float(fs), min(x), max(x)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('input sound: x')
 
	# plot the harmonic frequencies
	plt.subplot(3,1,2)
	if (hfreq.shape[1] > 0):
		numFrames = hfreq.shape[0]
		frmTime = hopSizeMelodia * np.arange(numFrames)/float(fs)
		hfreq[hfreq<=0] = np.nan
		plt.plot(frmTime, hfreq)
		plt.axis([0, x.size/float(fs), 0, maxplotfreq])
		plt.title('frequencies of harmonic tracks')
 
	# plot the output sound
	plt.subplot(3,1,3)
	plt.plot(np.arange(y.size)/float(fs), y)
	plt.axis([0, y.size/float(fs), min(y), max(y)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('output sound: y')
 
	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	
# 	inputFile = 'example_data/dan-erhuang_01_1.wav'
# 	melodiaInput = 'example_data/dan-erhuang_01.txt'

	inputFile = '/Users/joro/Documents/Phd/UPF/arias/laosheng-erhuang_04.wav'
	melodiaInput = '/Users/joro/Documents/Phd/UPF/arias/laosheng-erhuang_04.melodia'
	fromTs = 49.85
	toTs = 55.00
	
	fromTs = 0
	toTs = 858
		
	inputFile = '../sounds/vignesh.wav'
	melodiaInput = '../sounds/vignesh.melodia'
	fromTs = 0
	toTs = 2

	# exatract spectrum
	hfreq, hmag, hphase, fs, hopSizeMelodia, inputAudioFromTsToTs = extractHarmSpec(inputFile, melodiaInput, fromTs, toTs)
	np.savetxt('hfreq_2', hfreq)
	np.savetxt('hmag_2', hmag)
	np.savetxt('hphase_2', hphase)
	
	hfreq = np.loadtxt('hfreq_2')
	hmag = np.loadtxt('hmag_2')
	hphase = np.loadtxt('hphase_2')
	
	# resynthesize
	URIOutputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_harmonicModel.wav'
	y = resynthesize(hfreq, hmag, hphase, fs, hopSizeMelodia, URIOutputFile)
	
	visualizeHarmSp(inputAudioFromTsToTs, y, hopSizeMelodia )
	
