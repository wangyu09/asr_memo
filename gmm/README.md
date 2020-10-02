# GMM training functions

You need to decompress the data file after downloading.

## Compute MFCC feature

This function is a combination of feature extraction functions. It will do the followings:  
1. read a wave file from file path.  
2. pre-emphasis.  
3. cut frames.  
4. compute spectrogram feature from 3.  
5. compute fbank feature from 4.  
6. compute MFCC feature 5.  

```python
feat = compute_mfcc_feat("test.wav",sampleRate=16000,windowLength=25,windowShift=10,power=False,melBanks=24,mfccDim=12,energy=True,deltaOrder=2)
```

## K-Means cluster

Estimate the initial Pi, Mu and Sigma for a GMM model from sample data.

```Python
mu, sigma, pi, label = kmeans(data, k=3, iters=100, threshold=1e-6)
```
If use Scipy backend:
```Python
mu, sigma, pi, label = kmeans_scipy(data, k=3, iters=100)
```

## Gaussian Mixture Model

```python
gmm = GMM(k=3).initialize(data=data)
gmm.save("final.gmm")
```

### PDF value

Compute mixtured gaussian probability density function (PDF) value.

$$N(x|\mu,\Sigma) = \frac{1}{(2\pi)^{\frac{D}{2}}} \frac{1}{|\Sigma|^{\frac{1}{2}}}exp\{-\frac{1}{2}(x-\mu)^{T}\Sigma^{-1}(x-\mu)\}$$

```python
gmm = load_GMM("pretrained.gmm")

for k in range(gmm.K):
  v = gmm.gaussian(oneFrame,gmm.mu[k],gmm.sigma[k])
  print(f"GMM component: {k} PDF value: {v}")
```

### GMM likelihood

$$ P(x) = \sum_{k=1}^{K}\pi_{k}N(x|\mu_{k},\Sigma_{k})$$

```python
gmm = load_GMM("pretrained.gmm")

print(gmm.calc_log_likelihood(data))
print(gmm.calc_log_likelihood_parallel(data,threads=3))
```

### Estimate the GMM parameters

Compute the occupation of each gaussian PDF of a GMM:
$$ \gamma_{n,k}= \frac{\pi_{k}N(x_{n}|\mu_{k},\Sigma_{k})}{\sum_{j=1}^{J}\pi_{j}N(x_{n}|\mu_{j},\Sigma_{j})}$$

Estimate parameters of GMM:
$$ N_{k} = \sum_{n=1}^{N}\gamma_{n,k}$$
$$ \pi_{k}^{new} = \frac{N_{k}}{N}$$
$$ \mu_{k}^{new} = \frac{1}{N_{k}}\sum_{n=1}^{N}\gamma_{n,k}x_{n}$$
$$ \Sigma_{k}^{new} = \frac{1}{N_{k}}\sum_{n=1}^{N}\gamma_{n,k}(x_{n}-\mu_{k}^{new})(x_{n}-\mu_{k}^{new})^{T}$$

```python
gmm = load_GMM("pretrained.gmm")

print(gmm.estimate(data))
```

### Trick for Mutiple threads
$$ \mu(x) = \frac{\sum_{n=1}^{N}x_{n}}{N}$$
$$ \Sigma(x) = \mu(x^{2}) - \mu(x)^{2}$$

```python
gmm = load_GMM("pretrained.gmm")

print(gmm.estimate_parallel(data,threads=3))
```