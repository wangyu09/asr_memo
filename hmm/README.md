# Hiden Markov Model

## Algorithms 

### 1. Forward Algorithm

Initial step:  
$$ \alpha_{1}(i) = \pi_{i}B_{i}(O_{1}), i=1,2,,...,N $$

Loop steps:  
$$\alpha_{t+1}(i) = [\sum_{j=1}^{N}\alpha_{t}(j)A_{ji}]B_{i}(O_{t+1}),i=1,2,...,N$$

Final step:
$$P(O|\lambda) = \sum_{i=1}^{N}\alpha_{T}(i)$$

The Time Complexity: $O(TN^{2})$, The Space Complexity: $O(N)$.

### 2. Backward Algorithm

Initial step:  
$$ \beta_{T}(i) = 1, i=1,2,,...,N $$

Loop steps:  
$$\beta_{t}(i) = \sum_{j=1}^{N}A_{ij}B_{j}(o_{t+1})\beta_{t+1}(j),i=1,2,...,N$$

Final step:
$$P(O|\lambda) = \sum_{i=1}^{N}\pi_{i}B_{i}(o_{1})\beta_{1}(i)$$

The Time Complexity: $O(TN^{2})$, The Space Complexity: $O(N)$.

### 3. Viterbi Algorithm

Initial step:  
$$ \delta_{1}(i) = \pi_{i}B_{i}(O_{1}), i=1,2,,...,N $$
$$ \phi_{1}(i) = 0 $$

Loop steps:  
$$\alpha_{t+1}(i) = \max_{1 \le j \le N}[\delta_{t}(j)A_{ji}]B_{i}(O_{t+1}),i=1,2,...,N$$
$$ \phi_{t+1}(i) = arg\,\max_{1 \le j \le N}[\delta_{t-1}(j)A_{ji}] $$

Final step:
$$P(O|\lambda) = \max_{1 \le j \le N}\delta_{T}(i)$$
$$ i_{T}^{\ast} = arg\,\max_{1 \le j \le N}\delta_{T}(i)$$

Get The Best Path:

$$ i_{t}^{\ast} = \phi_{t+1}(i_{t+1}^{\ast}) $$

The Time Complexity: $O(TN^{2})$, The Space Complexity: $O(N)$.

## APIs

### Naive Discrete HMM
```python

Pi = np.array(
    [0.2,0.4,0.4],
  )
A = np.array([
    [0.5,0.2,0.3],
    [0.3,0.5,0.2],
    [0.2,0.3,0.5]]
  )
B = np.array([
    [0.5,0.5],
    [0.4,0.6],
    [0.7,0.3]]
  )

hmm = NaiveDiscreteHMM(initProbs=Pi,transProbs=A,obserProbs=B)

obs = np.array([0,1,0])
print( hmm.forward(obs) )
print( hmm.backward(obs) )
print( hmm.viterbi_decode(obs) )
```

### Advanced HMM

Initial a 3 states HMM, ID 0:
```python
hmm = HMM(nums=3,hmmID=0)
```
View the topology of this HMM:
```python
hmm.view()
```
Adjust the transform weight of some arcs:
```python
hmm.states[1].reset_arc_weight(arcID=0,weight=0.3)
hmm.states[1].reset_arc_weight(arcID=1,weight=0.7)
hmm.states[2].reset_arc_weight(arcID=0,weight=0.8)
```
View these arcs:
```python
print(hmm.states[0].arcs)
print(hmm.states[1].arcs)
print(hmm.states[2].arcs)
```
Compute forward probability and decode with Token Passing Viterbi for an observed probability matrix whose shape is (frames, states).
```python
observation = np.array([
    [0.6,0.2,0.2],
    [0.4,0.4,0.2],
    [0.5,0.2,0.3],
    [0.1,0.8,0.1],
    [0.3,0.3,0.4]
])

print(hmm.forward(obserProbs=observation,initProb=1.0))
print(hmm.viter_decode(obserProbs=observation,initProb=1.0))
```
Save this HMM model to file.
```python
hmm.save("test.hmm")
```
