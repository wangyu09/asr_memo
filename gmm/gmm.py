# Wang Yu, the University of Yamanashi, Japan
# Sep 27, 2020

import numpy as np
import scipy.cluster.vq as vq
import librosa
import threading
import time

import os,sys

DIR=os.path.dirname(os.path.dirname(__file__))
sys.path.append(DIR)

from feature import feature

def compute_mfcc_feat(filePath,sampleRate=16000,windowLength=25,
                      windowShift=10,power=False,melBanks=24,mfccDim=12,
                      energy=True,deltaOrder=2):
  '''
  Compute MFCC feature from wave file.

  Args:
    filePath: wave file path.
    sampleRate: sampling rate.
    windowLength: milliseconds of window's length.
    windowShift: milliseconds of window's shift.
    power: If True, use power instead of amplitude.
    melBanks: the number os mel filters.
    mfccDim: the dim of raw mfcc.
    energy: If True, add energy to MFCC feature.
    deltaOrder: If > 0, add delta to feature.
  
  Return:
  a 2-d array, (frames, feature dim).
  '''
  wave, sr = librosa.load(filePath,sr=sampleRate)
  wave = feature.pre_emphasis(wave)
  frameSingal = feature.frame(wave,sampleRate=sr,windowLength=windowLength,windowShift=windowShift)
  spectrogram = feature.compute_spectrogram(frameSingal,power=power)
  fbank = feature.compute_fbank(spectrogram,sampleRate=sampleRate,melBanks=melBanks)
  rawMfcc = feature.compute_mfcc(fbank)
  if deltaOrder > 0:
    assert isinstance(deltaOrder,int)
    rawMfcc = feature.add_delta(rawMfcc,order=deltaOrder)
  if energy:
    energy = feature.compute_energy(frameSingal)
    if deltaOrder > 0:
      energy = feature.add_delta(energy,order=deltaOrder)
    
    return np.column_stack([rawMfcc,energy])
  else:
    return rawMfcc

def kmeans(data,k,iters=100,threshold=1e-6):
  '''
  K-means cluster.

  Args:
    data: a 2-d array, (N, dim).
    k: a int value.
    iters: iteration times.
    thread: early stop threshold.
  
  Return:
    a tuple of 4 members:
      1. cluster center, a 2-d array, (k,dim).
      2. covariance matrix of each cluster, (k,dim,dim).
      3. percentage of the number of each cluster, (k,).
      4. labels, (N,).
  '''

  assert len(data.shape) == 2
  N,D = data.shape
  
  assert N >= k
  
  center = np.zeros((k,D),dtype="float32")
  
  # initialize
  cSize = int(N/k)
  for i in range(k):
      center[i] = np.mean( data[i*cSize:(i+1)*cSize], axis=0 )
  
  # train
  lastCenter = center.copy()
  for i in range(iters):
      
    # compute rnk
    rnk = np.zeros((N,k))
    for j in range(k):
        rnk[:,j] = np.sum((data-center[j])**2, axis=1)

    minIndex = np.argmin(rnk,axis=1)
    rnk *= 0
    rnk[ [d for d in range(N)], minIndex ] = 1
    
    # update center
    for j in range(k):
        index = rnk[:,j]
        center[j] = np.mean( data[index==1], axis=0 )
    
    if np.mean(np.sum((center-lastCenter)**2,axis=1)) < threshold:
        break
        
    lastCenter = center.copy()
  
  cov = np.zeros((k,D,D),dtype="float32")
  pi = np.zeros((k,),dtype="float32")
  for j in range(k):
      index = rnk[:,j]
      cov[j] = np.cov(data[index==1],rowvar=False)
      
      pi[j] = np.sum(index)/N
  
  return center, cov, pi, minIndex

def kmeans_scipy(data,k,iters=100):
  '''
  K-means cluster.

  Args:
    data: a 2-d array, (N, dim).
    k: a int value.
    iters: iteration times.
    thread: early stop threshold.
  
  Return:
    a tuple of 4 members:
      1. cluster center, a 2-d array, (k,dim).
      2. covariance matrix of each cluster, (k,dim,dim).
      3. percentage of the number of each cluster, (k,).
      4. labels, (N,).
  '''

  assert len(data.shape) == 2
  N,D = data.shape
  
  assert N >= k
  
  mu = []
  sigma = []
  
  (centroids, labels) = vq.kmeans2(data, k, minit="points", iter=iters)
  clusters = [[] for i in range(k)]
  for (l,d) in zip(labels,data):
    clusters[l].append(d)

  for cluster in clusters:
    mu.append(np.mean(cluster, axis=0))
    sigma.append(np.cov(cluster, rowvar=0))
  pi = np.array([len(c)*1.0 / len(data) for c in clusters])

  return np.array(mu), np.array(sigma), pi, labels

class GMM:
  '''
  Gaussian Mixture Model.
  '''
  def __init__(self, k):
    '''
    Args:
      k: a positive int value within.
    '''
    assert isinstance(k,int) and k > 0, f"<k> is an invalid value: {k}."

    self.K = k
    self.D,self.mu,self.sigma,self.pi = None,None,None,None
  
  def initialize(self,data):
    '''
    Initialize the GMM parameters.

    Args:
      data: a 2-d array.
    
    Return:
      itself.
    '''
    assert isinstance(data,np.ndarray) and len(data.shape) == 2

    self.mu, self.sigma, self.pi, _ = kmeans(data,self.K,50)
    self.D = data.shape[1]

    return self

  def gaussian(self,x,mu,sigma,epsilon=1e-80):
    '''
    Compute a probability density function value.

    Args:
      x: a 1-d array, (dim,).
      mu: a 1-d array, (dim,).
      sigma: a 2-d array, (dim,dim).
      epsilon: an extremely small value.

    Return:
      a float value.
    '''
    if not self.D:
      raise Exception("Please initialize the GMM before using.")

    det_sigma = np.linalg.det(sigma) + epsilon
    inv_sigma = np.linalg.inv(sigma + np.identity(self.D) * epsilon)
    expo = np.dot(np.transpose(x - mu), inv_sigma)
    expo = np.dot(expo, (x-mu))
    const = 1/((2*np.pi)**(self.D/2))
    return const * (det_sigma)**(-0.5) * np.exp(-0.5 * expo)
  
  def calc_log_likelihood(self,data):
    '''
    Compute log likelihood of a dataset.

    Args:
      data: a 2-d array, (N,dim).
    
    Return:
      a float value.
    '''
    if not self.D:
      raise Exception("Please initialize the GMM before using.")

    N = len(data)
    llh = 0
    for n in range(N):
      prob = 0
      for k in range(self.K):
        prob += float(self.pi[k] * self.gaussian(data[n],self.mu[k],self.sigma[k]))
      llh += np.log(prob)
    
    return llh/N

  def estimate(self,data):
    '''
    Estimate the parameters with EM algorithm.

    Args:
      data: a 2-d array, (N,dim).
    
    Return:
      a float value.
    '''
    if not self.D:
      raise Exception("Please initialize the GMM before using.")

    N = len(data)

    # E step
    gamma = np.zeros([N, self.K])
    for n in range(N):

      for k in range(self.K):
        gamma[n,k] = self.pi[k] * self.gaussian(data[n],self.mu[k],self.sigma[k])
      
      gamma[n] = gamma[n]/np.sum(gamma[n])
  
    # M step
    for k in range(self.K):

      # 1, estimate Pi
      Nk = np.sum(gamma[:,k])
      self.pi[k] = Nk/N

      # 2, get the statistics for Mu of Sigma
      self.mu[k] = np.sum(gamma[:,k][:,None]*data,axis=0) / Nk

      statsForSigma = np.zeros([N,self.D,self.D])
      for n in range(N):
          temp = (data[n]-self.mu[k])[:,None] 
          statsForSigma[n] = gamma[n,k] * np.dot(temp, temp.T)
  
      # 3, estimate Mu and Sigma
      self.sigma[k] = np.sum(statsForSigma,axis=0)/Nk
        
    # evaluate
    return self.calc_log_likelihood(data)

  def calc_log_likelihood_parallel(self,data,threads=3):
    '''
    Compute log likelihood of a dataset with multiple threads.

    Args:
      data: a 2-d array, (N,dim).
    
    Return:
      a float value.
    '''    
    if not self.D:
      raise Exception("Please initialize the GMM before using.")

    threadManager = {}

    # The core function to compute log likelihood of a chunk data.
    def core(tid,chunkData):
      '''
      The core function to compute the likelihood.
      '''
      N = len(chunkData)
      llh = 0
      for n in range(N):
        prob = 0
        for k in range(self.K):
          prob += float(self.pi[k] * self.gaussian(chunkData[n],self.mu[k],self.sigma[k]))
        llh += np.log(prob)
      threadManager[tid] = llh
    
    # Split data into "threads" chunks and run threads
    chunkLength = int(np.ceil(len(data)//threads))
    for tid in range(threads):
      if tid == threads - 1:
        chunkData = data[ tid*chunkLength: ]
      else:
        chunkData = data[ tid*chunkLength:(tid+1)*chunkLength ]
      if len(chunkData) > 0:
        threadManager[tid] = threading.Thread(target=core,args=(tid,chunkData,))
        threadManager[tid].start()
    
    # Process the results
    totalLlh = 0
    for tid,obj in threadManager.items():
      if not isinstance(obj,float) and obj.is_alive():
        obj.join()
      totalLlh += threadManager[tid]
    
    return totalLlh/len(data)

  def estimate_parallel(self,data,threads=3):
    '''
    Estimate the parameters with EM algorithm with mutiple threads.

    Args:
      data: a 2-d array, (N,dim).
    
    Return:
      a float value.
    '''
    if not self.D:
      raise Exception("Please initialize the GMM before using.")

    threadManager = {}

    def accumulate_stats(tid,chunkData):
      
      occuStats = np.zeros([self.K,],dtype="float32")
      fOrderStats = np.zeros([self.K,self.D,],dtype="float32")
      sOrderStats = np.zeros([self.K,self.D,self.D],dtype="float32")

      count = 0
      for nd in chunkData:

        # statstics for pi
        gamma = np.zeros([self.K,],dtype="float32")
        for k in range(self.K):
          gamma[k] = self.pi[k] * self.gaussian(nd,self.mu[k],self.sigma[k])
        
        if (np.sum(gamma)) == 0:
          continue  

        gamma = gamma/np.sum(gamma)
        occuStats += gamma

        for k in range(self.K):
          # statstics for mu
          fOrderStats[k] += gamma[k]*nd
          # statstics for cov
          sOrderStats[k] += gamma[k]*( np.dot(nd[:,None],nd[None,:]) )
        
        count += 1
      
      threadManager[tid] = (occuStats,fOrderStats,sOrderStats,count)

    # Split data into "threads" chunks and run threads
    chunkLength = int(np.ceil(len(data)//threads))
    for tid in range(threads):
      if tid == threads-1:
        chunkData = data[tid*chunkLength:]
      else:
        chunkData = data[tid*chunkLength:(tid+1)*chunkLength]
      if len(chunkData) > 0:
        threadManager[tid] = threading.Thread(target=accumulate_stats,args=(tid,chunkData,))
        threadManager[tid].start()
    
    # Process the results
    N = 0
    sumOccuStats = np.zeros([self.K,],dtype="float32")
    sum1OrderStats = np.zeros([self.K,self.D,],dtype="float32")
    sum2OrderStats = np.zeros([self.K,self.D,self.D],dtype="float32")

    for tid,obj in threadManager.items():
      if not isinstance(obj,tuple) and obj.is_alive():
        obj.join()
      sumOccuStats += threadManager[tid][0]
      sum1OrderStats += threadManager[tid][1]
      sum2OrderStats += threadManager[tid][2]
      N += threadManager[tid][3]
    
    # update pi
    self.pi = sumOccuStats/N
    # update mu
    self.mu = sum1OrderStats/sumOccuStats[:,None]
    # update sigma
    mean2OrderStats = sum2OrderStats/sumOccuStats[:,None,None]
    for k in range(self.K):
        temp2 = np.dot( self.mu[k][:,None], self.mu[k][None,:] )
        self.sigma[k] = mean2OrderStats[k] - temp2

    return self.calc_log_likelihood_parallel(data,threads=threads)

  def save(self,fileName):
    '''
    Save the GMM model to file.

    Args:
      fileName: a string with suffix ".gmm".
    
    Return:
      the name of saved file.
    '''
    if not self.D:
      raise Exception("Please initialize the GMM before saving.")

    assert isinstance(fileName,str)
    fileName = fileName.strip()
    if not fileName.endswith(".gmm"):
      fileName += ".gmm"

    if os.path.isfile(fileName):
      os.remove(fileName)

    with open(fileName,"a") as fa:

      fa.write( f"K {self.K}\n" )
      fa.write( f"D {self.D}\n\n" )

      for k in range(self.K):
        fa.write( f"Component {k}\n" )
        fa.write( f"{self.pi[k]}\n" )
        fa.write( " ".join( map(str,self.mu[k].tolist())) + "\n" )
        for d in range(self.D):
          fa.write( " ".join( map(str,self.sigma[k][d].tolist())) + "\n" )
        fa.write("\n")
    
      fa.write("End\n")
  
def load_GMM(filePath):
  '''
  Load a GMM from a .gmm file.

  Args:
    filePath: a .gmm file path.
  
  Return:
    a GMM object.
  '''
  assert os.path.isfile(filePath), f"No such file: {filePath} ."

  gmm = GMM(1)

  with open(filePath,"r",encoding="utf-8") as fr:
    
    headerK = fr.readline().strip().split()
    assert len(headerK) == 2 and headerK[0] == "K", "Wrong info: K (in the first line)."
    gmm.K = int(headerK[1])
    
    headerD = fr.readline().strip().split()
    assert len(headerD) == 2 and headerD[0] == "D", "Wrong info: D (in the second line)."
    gmm.D = int(headerD[1])

    fr.readline()

    gmm.pi = np.zeros([gmm.K,])
    gmm.mu = np.zeros([gmm.K,gmm.D])
    gmm.sigma = np.zeros([gmm.K,gmm.D,gmm.D])

    for k in range(gmm.K):

      title = fr.readline().strip()
      assert title == f"Component {k}", f"Found wrong info when read title of {k} component."

      kv = fr.readline().strip()
      assert len(kv) > 0, f"Found wrong info when read pi of {k}th component."
      gmm.pi[k] = float(kv)

      kmu = fr.readline().strip().split()
      assert len(kmu) == gmm.D, f"Found wrong info when read mu of {k}th component."
      gmm.mu[k] = np.array(kmu,dtype="float32")

      ksigma = np.zeros([gmm.D,gmm.D])
      for d in range(gmm.D):
        temp = fr.readline().strip().split()
        assert len(temp) == gmm.D, f"Found wrong info when read sigma of {k}th component."
        ksigma[d] = np.array(temp,dtype="float32")
      gmm.sigma[k] = ksigma

      fr.readline()
    
    assert fr.readline().strip() == "End", "Missed end flag in file."

  return gmm

      






