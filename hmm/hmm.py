# Wang Yu, the University of Yamanashi, Japan
# Oct 2, 2020

import numpy as np
import os,sys

DIR=os.path.dirname(os.path.dirname(__file__))
sys.path.append(DIR)

from gmm import gmm
from collections import namedtuple
import copy

class NaiveDiscreteHMM:
  '''
  A naive HMM with discrete observation probability.
  '''

  def __init__(self,initProbs,transProbs,obserProbs):
    
    assert isinstance(initProbs,np.ndarray) and len(initProbs.shape) == 1 
    assert isinstance(transProbs,np.ndarray) and len(transProbs.shape) == 2 and transProbs.shape[0] == transProbs.shape[1] == initProbs.shape[0]
    assert isinstance(obserProbs,np.ndarray) and len(obserProbs.shape) == 2 and obserProbs.shape[0] == transProbs.shape[0]

    self.I = initProbs
    self.A = transProbs
    self.B = obserProbs

    self.states = obserProbs.shape[0]
    self.classes = obserProbs.shape[1]
  
  def forward(self,observation):
    '''
    Compute the forward probability of an observed sequence.
    Time: O(T*(N**2)), Space: O(N).

    Args:
      observation: an 1-d array.
    
    Return:
      a float value, the forward probability of the observed sequence.
    '''
    curProbs = np.zeros([self.states]) # record probability of current frame.
    lastProbs = np.zeros([self.states]) # record probability of last frame.

    for t,o in enumerate(observation):
      obs = int(o)
      assert  0 <= o < self.classes

      if t == 0:
        for i in range(self.states):
          curProbs[i] = self.I[i]*self.B[i,obs]

      else:
        for i in range(self.states):
          sumProb = 0
          for j in range(self.states):
            sumProb += lastProbs[j]*self.A[j,i]
          curProbs[i] = sumProb*self.B[i,obs]
      lastProbs = curProbs.copy()
    
    return float( np.sum(curProbs) )
  
  def backward(self,observation):
    '''
    Compute the backward probability of an observed sequence.
    Time: O(T*(N**2)), Space: O(N).

    Args:
      observation: an 1-d array.
    
    Return:
      a float value, the backward probability of the observed sequence.
    '''
    curProbs = np.zeros([self.states]) # record probability of current frame.
    lastProbs = np.ones([self.states]) # record probability of last frame.
    T = len(observation)

    for t in range(T-1,-1,-1):
      obs = int(observation[t])
      assert  0 <= obs < self.classes

      if t == 0:
        for i in range(self.states):
          curProbs[i] = self.I[i]*self.B[i,obs]*lastProbs[i]

      else:
        for i in range(self.states):
          sumProb = 0
          for j in range(self.states):
            sumProb += self.A[i,j]*self.B[j,obs]*lastProbs[j]
          curProbs[i] = sumProb

      lastProbs = curProbs.copy()
    
    return float( np.sum(curProbs) )

  def viterbi_decode(self,observation):
    '''
    Compute the best path by viterbi search algorithm.
    Time: O(T*(N**2)), Space: O(T*N).

    Args:
      observation: an 1-d array.
    
    Return:
      a tuple with two members:
        1. an 1-d array, the best path.
        2. a float value, the probability of best path.
    '''
    T = len(observation)
    curProbs = np.zeros([self.states,]) # record probability of current frame.
    lastProbs = np.zeros([self.states,]) # record probability of last frame.

    pathMemory = np.zeros([T,self.states,],dtype="int32") # record history path.

    for t,o in enumerate(observation):
      assert  0 <= o < self.classes
      obs = int(observation[t])
      if t == 0:
        for i in range(self.states):
          curProbs[i] = self.I[i]*self.B[i,obs]
      else:
        for i in range(self.states):
          sumProb = np.zeros([self.states,])
          for j in range(self.states):
            sumProb[j] = lastProbs[j]*self.A[j,i]
          pathMemory[t,i] = np.argmax(sumProb)
          curProbs[i] = sumProb[int(pathMemory[t,i])]*self.B[i,obs]
      lastProbs = curProbs.copy()

    bestPath = []
    bestPath.append( int(np.argmax(curProbs)) )
    for t in range(T-1,0,-1):
      bestPath.append( int(pathMemory[t,bestPath[-1]]) )

    return np.array( bestPath[::-1] ), np.around(np.max(curProbs[-1]),4)

class State:
  '''
  Create a state token to record:
    1. Which HMM it belongs to.
    2. Which state it is of this HMM.
    3. Whether it is the termination state.
    4. All arcs of this state. A transforming arc: (origin HMM, origin state) -> (target HMM, target state), with probability: weight.
  '''
  def __init__(self,hmmID,stateID,terminate=False):
    self.hID = hmmID
    self.sID = stateID
    self.is_termination = terminate

    self.arcs = []
    self.arcIDCount = 0
  
  @property
  def ArcSpec(self):
    return namedtuple("Arc",["aID","start","end","weight"])
  
  @property
  def NodeSpec(self):
    return namedtuple("Node",["hID","sID"])

  def add_arc(self,endHmm,endState,weight):
    '''
    Add a arc
    '''
    if endHmm != self.hID:
      assert endState == 0, "If this arc goes to other HMM, it can only skip to state 0."
      assert self.is_termination, "Only termination state can goes to other HMM."
      for a in self.arcs:
        if a.end.hID == endHmm:
          raise Exception("Cannot add arc to skip to the same target HMM twice.")
    else:
      for a in self.arcs:
        if a.end.sID == endState:
          raise Exception("Cannot add arc to skip to the same target state twice.")
    
    self.arcs.append( self.ArcSpec(self.arcIDCount, self.NodeSpec(self.hID,self.sID), self.NodeSpec(endHmm,endState), weight) )
    self.arcIDCount += 1
  
  def remove_arc(self,arcID):
    for i,a in enumerate(self.arcs):
      if a.aID == arcID:
        self.arcs.pop(i)

  def reset_arc_weight(self,arcID,weight):
    for i,a in enumerate(self.arcs):
      if a.aID == arcID:
        self.arcs[i] = a._replace(weight=weight)

class ViterbiToken:

  def __init__(self,hID,sID,weight):
    self.history = [ self.NodeSpec(hID,sID), ]
    self.p = weight

  @property
  def NodeSpec(self):
    return namedtuple("Node",["hID","sID"])

  def copy(self):
    return copy.deepcopy(self)
  
  def passing(self,arc):
    self.history.append( arc.end )
    self.p *= arc.weight

  def get_path(self):

    path = np.zeros([len(self.history),2])
    for i,h in enumerate(self.history):
      path[i][0],path[i][1] = h.hID,h.sID
    
    return path, self.p

class HMM:

  def __init__(self,nums,hmmID,initArcs=True):

    assert isinstance(nums,int) and nums > 0

    self.hID = hmmID

    # Generate states
    self.states = []
    for n in range(nums):
      self.states.append( State(self.hID,n) )
    self.states[-1].is_termination = True

    if initArcs:
      # Add arcs
      for i in range(nums):
        self.states[i].add_arc(endHmm=self.hID,endState=i,weight=0.5)
        if i != nums-1:
          self.states[i].add_arc(endHmm=self.hID,endState=i+1,weight=0.5)
    
  def __view(self):

    contents = f"HMM ID: {self.hID}\n"
    contents += f"States: {len(self.states)}\n"
    contents += f"Termination State ID: {len(self.states)-1}\n"
    contents += f"Arcs ( start HMM , start state -> end HMM, end state P: transform weight ):\n"
    for s in self.states:
      for a in s.arcs:
        contents += f"{a.start.hID},{a.start.sID} -> {a.end.hID},{a.end.sID} P:{a.weight}  \n"
    
    return contents
  
  def view(self):
    print(self.__view())
  
  def save(self,fileName):

    assert isinstance(fileName,str)
    if not fileName.strip().endswith(".hmm"):
      fileName += ".hmm"
    
    contents = self.__view()
    with open(fileName,"w") as fw:
      fw.write(contents + "End\n")
    
    return fileName

  def forward(self,obserProbs,initProb=1.0):
    '''
    Compute the forward probability of a observation.

    Args:
      obserProbs: a 2-d array, (T,states). In each frame, each was observed with this probability.
      initProb: a float value within [0,1].
    
    Return:
      a float value.
    '''
    assert isinstance(obserProbs,np.ndarray) and len(obserProbs.shape)==2
    assert obserProbs.shape[1] == len(self.states)

    curProbs = {}
    lastProbs = {}
    T = obserProbs.shape[0]

    for t in range(T):
      if t == 0:
        curProbs[0] = initProb*obserProbs[t,0]
      else:
        for sID,forwardWeight in lastProbs.items():
          for arc in self.states[sID].arcs:
            if arc.end.sID not in curProbs.keys():
              curProbs[arc.end.sID] = forwardWeight*arc.weight
            else:
              curProbs[arc.end.sID] += forwardWeight*arc.weight
        for sID in curProbs.keys():
          curProbs[sID] *= obserProbs[t,sID]
      
      lastProbs.clear()
      lastProbs.update(curProbs)
      curProbs.clear()

    sumProb = 0
    for finalStateID,finalWeight in lastProbs.items():
      if self.states[finalStateID].is_termination:
        sumProb += finalWeight
    
    return sumProb

  def cherry_pick(self, tokens, obserProb=1.0):
    '''
    1. Choose the best token arrived the same state.
    2. Discard other tokens.

    Args:
      tokens: a list or tuple of viterbi tokens.
    
    Return:
      a ViterbiToken object.
    '''
    bestToken = tokens[0]
    for i in range(1,len(tokens)):
      if tokens[i].p > bestToken.p:
        del bestToken
        bestToken = tokens[i]
      del tokens[i]
    bestToken.p *= obserProb
  
    return bestToken

  def viterbi_decode(self,obserProbs,initProb=1.0):
    '''
    Search the best path with viter.

    Args:
      obserProbs: a 2-d array, (T,states). In each frame, each was observed with this probability.
      initProb: a float value within [0,1].

    Return:
      a tuple with two values:
      1. an 2-d array standing for the best alignment.
      2. a float value, the probability.
    '''
    assert isinstance(obserProbs,np.ndarray) and len(obserProbs.shape)==2
    assert obserProbs.shape[1] == len(self.states)

    curProbs = {}
    lastProbs = {}
    T = obserProbs.shape[0]

    for t in range(T):
      if t == 0:
        token = ViterbiToken( self.hID, 0, initProb*obserProbs[t,0] )
        curProbs[0] = token

      else:
        for sID,token in lastProbs.items():
          for arc in self.states[sID].arcs:

            tempToken = token.copy()
            tempToken.passing(arc)

            if arc.end.sID not in curProbs.keys():
              curProbs[arc.end.sID] = [ tempToken, ]
            else:
              curProbs[arc.end.sID].append(tempToken)

        for sID in curProbs.keys():
          curProbs[sID] = self.cherry_pick( curProbs[sID], obserProbs[t,sID] )
      
      lastProbs.clear()
      lastProbs.update(curProbs)
      curProbs.clear()

    finalTokens = []
    for finalStateID,finalToken in lastProbs.items():
      if self.states[finalStateID].is_termination:
        finalTokens.append(finalToken)
    bestToken = self.cherry_pick(finalTokens)
    
    return bestToken.get_path()

def load_HMM(filePath):
  '''
  Load a HMM from a .hmm file.

  Args:
    filePath: a .hmm file path.
  
  Return:
    a HMM object.
  '''
  assert os.path.isfile(filePath), f"No such file: {filePath} ."

  with open(filePath,"r",encoding="utf-8") as fr:
    
    headerID = fr.readline().strip().split()
    assert len(headerID) == 3 and headerID[0] == "HMM" and headerID[1] == "ID:", "Wrong info: HMM ID (in the first line)."
    hmmID = int(headerID[2])
    
    headerS = fr.readline().strip().split()
    assert len(headerS) == 2 and headerS[0] == "States:", "Wrong info: states (in the second line)."
    nums = int(headerS[1])

    hmm = HMM(nums=nums,hmmID=hmmID,initArcs=False)

    fr.readline()
    fr.readline()

    while True:
      arc = fr.readline().strip()
      if arc == "End":
        break
      elif len(arc) == 0:
        raise Exception("Missed end flag in file.")
      else:
        arc = arc.split(",",maxsplit=1)
        startHmm = int(arc[0])
        arc = arc[1].split("->",maxsplit=1)
        startState = int(arc[0])
        arc = arc[1].split(",",maxsplit=1)
        endHmm = int(arc[0])
        arc = arc[1].split("P:",maxsplit=1)
        endState = int(arc[0])
        weight = float(arc[1])

        hmm.states[startState].add_arc(endHmm,endState,weight)

  return hmm
        
        
        
        
        

        
        
        
        
        


