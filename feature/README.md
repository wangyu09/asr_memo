# Feature extraction functions
1. spectrogram feature  
2. fbank feature  
3. mfcc feature  

## Pre-emhpasis
 
$$x(n) = x(n) - \alpha*x(n-1),0<\alpha<1$$

```python
import librosa

wave, sampleRate = librosa.load("test.wav",sr=16000)
wave = pre_emphasis(wave)
```

## Hamming windows
$$ w(n) = 
\begin{cases}
0.54-0.46*cos(\frac{2 \pi n}{L} ),0<n<L\\
0,otherwise\\
\end{cases}
$$

## Framing
$$ N_{frame}=floor(\frac{N_{singal}-width}{shift})+1$$

```python
frameSingal = frame(wave,sampleRate=16000,windowLength=25,windowShift=10) 
```

## Fourier transform
### DFT:
$$X(n) = \sum_{n=0}^{N-1} x(n) e^{\frac{-j2 \pi nm}{N}}=\sum_{n=0}^{N-1} x(n)[cos(\frac{2 \pi n m}{N})-jsin(\frac{2 \pi n m}{N})], m=(0,1...N-1)$$ 

### IDFT:
$$c(k) = \sum_{m=0}^{N-1} X(m) e^{\frac{j2 \pi m k}{N}}=\sum_{m=0}^{N-1} X(m)[cos(\frac{2 \pi m k}{N})+jsin(\frac{2 \pi m k}{N})], k=(0,1...N-1)$$ 

### DCT:
$$c(i) = \sqrt{\frac{2}{N}}\sum_{m=0}^{N-1} X(m)*cos((m-\frac{1}{2})\frac{i\pi}{N}), i=(0,1...N-1)$$ 

```python
dftSingal = naive_dft(frameSingal) 
```

## Spectrogram feature

```python
powerSpectrogram = compute_spectrogram(frameSingal,power=True) 
```

## Fbank feature
Mel Freuence:
$$mel(f)=2595log_{10}(1+\frac{f}{700})$$

```python
fbank = compute_fbank(spectrogram,sampleRate=16000,melBanks=24) 
```

## Mfcc feature

```python
rawMfcc = compute_mfcc(fbank,dim=12) 
```

## Energy

$$E=\sum x(n)$$

```python
energy = compute_energy(frameSingal) 
```

## Add delta

$$\delta(n) = \frac{x(n+1)-x(n-1)}{2}$$

```python
mfccTemp = add_delta(rawMfcc,order=2)
energyTemp = add_delta(energy,order=2)

mfcc = np.column_stack([mfccTemp,energyTemp])
```

## Plot feature figure
```python
plot_feature(mfcc,fileName="mfcc.png")
```