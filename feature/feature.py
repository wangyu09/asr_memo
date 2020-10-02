# Wang Yu, the University of Yamanashi, Japan
# Sep 27, 2020

import numpy as np
from scipy.fftpack import dct

#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

def pre_emphasis(singal, alpha=0.97):
  '''
  Pre-emphasis the singal of high-frequency.

  Args:
    singal: a 1-d numpy array, (samplePoints,)
    alpha: a float value within [0,1]
  
  Return:
    a 1-d array whose length will be: original length - 1
  '''
  assert isinstance(singal,np.ndarray) and len(singal.shape) == 1, "<singal> must be a 1-d array."
  assert 0 <= alpha <= 1, "<alpha> must be a value with in [0,1]."

  return singal[1:] - alpha*singal[:-1]

def hamming_window(length):
  '''
  Get a hamming window filter.

  Args:
    length: an int value.

  Return:
    a 1-d array whose length will be <length>.
  '''
  assert isinstance(length,int) and length > 3, "<length> is unresonable." 

  window = np.zeros([length,],dtype="float32")
  for n in range(length):
      window[n] = 0.54 - 0.46*np.cos(2*np.pi*n/length)
  
  return window

def frame(singal,sampleRate=16000,windowLength=25,windowShift=10):
  '''
  Frame the singal.

  Args:
    singal: 1-d array.
    sampleRate: frequence of sampling.
    windowLength: milliseconds of wondow's width.
    windowLength: milliseconds of wondow's shift.
  
  Return:
    a 2-d array, (frames, sample points of one frame).
  '''
  assert isinstance(singal,np.ndarray) and len(singal.shape) == 1, "<singal> must be a 1-d array."

  width = int(sampleRate*windowLength*0.001)
  shift = int(sampleRate*windowShift*0.001)
  
  numSamples = len(singal)
  numFrames = int(np.floor((numSamples-width)/shift) + 1)
  
  window = hamming_window(width)
  
  result = np.zeros((numFrames,width),dtype="float32")
  
  for n in range(numFrames):
      result[n] = singal[n*shift:(n*shift+width)] * window
  
  return result

def naive_dft(frameSingal):
  '''
  Naive dft. Time complexity O(n*n).

  Args:
    frameSingal: a 2-d array.(frames, sample points of one frame)
  
  Return:
    a 2-d array. (frames, (sample points of one frame)//2 + 1)
  '''
  assert isinstance(frameSingal,np.ndarray) and len(frameSingal.shape) == 2, "<frameSingal> must be a 2-d array."

  N = frameSingal.shape[1]
  
  transMat = np.zeros((N,N//2+1),dtype="complex128")
  for m in range(N//2+1):
      for n in range(N):
          transMat[n][m] = complex(np.cos(2*np.pi*n*m/N),-np.sin(2*np.pi*n*m/N))
  
  return np.abs(np.dot(frameSingal,transMat))

def compute_spectrogram(frameSingal,power=True):
  '''
  Compute spetrogram feature.

  Args:
    frameSingal: a 2-d array, (frames, sample points of one frame).
    power: bool value. If true, return power spectrogram.
  
  Return:
    a 2-d array, (frames, (sample points of one frame)//2 + 1).
  '''
  assert isinstance(frameSingal,np.ndarray) and len(frameSingal.shape) == 2, "<frameSingal> must be a 2-d array."

  frames = frameSingal.shape[0]
  points = frameSingal.shape[1]
  
  fftLen = 1
  while points:
      fftLen <<= 1
      points >>= 1
  
  cFFT = np.fft.fft(frameSingal,n=fftLen)
  spectrogram = np.abs(cFFT[:,0:int(fftLen/2+1)]) 
  if power:
      spectrogram **= 2
      
  return spectrogram

def compute_fbank(spectrogram,sampleRate=16000,melBanks=24):
  '''
  Compute fbank feature.

  Args:
    spectrogram: a 2-d array, spectrogram feature, (frames, points).
    sampleRate: frequence of sampling.
    melBanks: int value, the number of mel banks.
  
  Return:
    a 2-d array, (frames, mel banks).
  '''
  assert isinstance(spectrogram,np.ndarray) and len(spectrogram.shape) == 2, "<spectrogram> must be a 2-d array."

  # create mel filter banks
  melMin = 0
  melMax = 2595 * np.log10( 1 + (sampleRate/2)/700 )
  melFiltersPoints = np.linspace(melMin,melMax,melBanks+2)
  
  freqFiltersPoints = (700*(np.power(10,(melFiltersPoints/2595))-1))
  span = (sampleRate/2)/spectrogram.shape[1]
  
  freqBin = np.floor(freqFiltersPoints/span)
  
  banks = np.zeros((spectrogram.shape[1],melBanks))
  
  for m in range(1,melBanks+1):
      low = int(freqBin[m-1])
      medium = int(freqBin[m])
      high = int(freqBin[m+1])
      for k in range(low,medium):
          banks[k,m-1] = (k-low)/(medium-low)
      for k in range(medium,high):
          banks[k,m-1] = (high-k)/(high-medium)
  
  return np.log10( np.dot(spectrogram,banks) )

def compute_mfcc(fbank,dim=12):
    '''
    Compute MFCC feature. the first dim will be discarded.

    Args:
      fbank: a 2-d array,fbank feature,(frames,melBanks).
      dim: int value, the output dims of mfcc feature.
    
    Return:
      a 2-d array,(frames,dim).
    '''     
    assert isinstance(fbank,np.ndarray) and len(fbank.shape) == 2, "<fbank> must be a 2-d array."

    dctFeat = dct(fbank)
    return dctFeat[:,1:dim+1]

def compute_energy(frameSingal):
  '''
  Compute energy feature.

  Args:
    singal: a 2-d array, (frames, sample points of one frame).

  Return:
    a 2-d array, (frames,1).
  '''
  assert isinstance(frameSingal,np.ndarray) and len(frameSingal.shape) == 2, "<frameSingal> must be a 2-d array."

  return np.sum(frameSingal*frameSingal,axis=1,keepdims=True)

def add_delta(feature,order=2):
  '''
  Add delta to feature.

  Args:
    feature: a 2-d array, (frames,feature dim)
    order: the order, an int value.
  
  Return:
    a 2-d array, (frames,dim*(order+1)).
  '''
  assert isinstance(feature,np.ndarray) and len(feature.shape) == 2, "<feature> must be a 2-d array."
  
  result = [feature,]
  target = feature
  
  for o in range(order):
    delta = np.zeros_like(target)
    for f in range(target.shape[0]):
      if f == 0:
        left = target[0,:]
      else:
        left = target[f-1,:]
      if f == target.shape[0]-1:
        right = target[-1,:]
      else:
        right = target[f+1,:]
      delta[f] = (right-left)/2
    result.append(delta)
    target = delta

  return np.concatenate(result,axis=1) 

def plot_feature(feature,fileName=None):
  '''
  Plot a feature figure.

  Args:
    feature: a 2-d array, (frames,feature dims).
    fileName: If not None, expected a png file name. Save figure to file.
  '''
  fig = plt.figure(figsize=(20,5))
  heatMap = plt.pcolor(feature.T)
  fig.colorbar(mappable=heatMap)
  plt.xlabel("Frames")
  plt.ylabel("Feature Dims")
  if fileName:
    assert isinstance(fileName,str) and fileName.endswith(".png")
    plt.savefig(fileName)

def plot_wave(singal,fileName=None):
  '''
  Plot a wave figure.

  Args:
    singal: a 1-d array.
    fileName: If not None, expected a png file name. Save figure to file.
  '''
  plt.figure(figsize=(20,5))
  plt.plot(singal)
  plt.xlabel("Time / Sampling Points")
  plt.ylabel("Amplitude")
  if fileName:
    assert isinstance(fileName,str) and fileName.endswith(".png")
    plt.savefig(fileName)

def plot_window(window,fileName=None):
  '''
  Plot a window figure.

  Args:
    singal: a 1-d array.
    fileName: If not None, expected a png file name. Save figure to file.
  '''
  plt.figure(figsize=(20,5))
  plt.plot(window)
  plt.xlabel("Window Points")
  plt.ylabel("Value")
  if fileName:
    assert isinstance(fileName,str) and fileName.endswith(".png")
    plt.savefig(fileName)