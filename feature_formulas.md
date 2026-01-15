# EMG Feature Extraction Formulas

Let a windowed EMG signal be $x = [x_1, x_2, \dots, x_N]$ of length $N$, and let $\bar{x}$ be its mean.

## Time-Domain Features

### 1. Root Mean Square (RMS)
$$
\text{RMS} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}
$$

### 2. Waveform Length (WL)
$$
\text{WL} = \sum_{i=1}^{N-1} \big|x_{i+1} - x_i\big|
$$

### 3. Variance (VAR)
$$
\text{VAR} = \frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2
$$

### 4. Mean Absolute Value (MAV)
$$
\text{MAV} = \frac{1}{N} \sum_{i=1}^{N} |x_i|
$$

### 5. Maximum Absolute Value
$$
\text{MAX} = \max_{1 \le i \le N} |x_i|
$$

### 6. Median Absolute Value
$$
\text{MEDIAN} = \operatorname{median}(|x_1|, |x_2|, \dots, |x_N|)
$$

### 7. Zero Crossings (ZC)
Let the threshold be
$$
\text{threshold} = \operatorname{mean}(|x_i|).
$$
Then the zero-crossing count (around the threshold) is
$$
\text{ZC} = \sum_{i=1}^{N-1} \Big|\,[x_i > \text{threshold}] - [x_{i+1} > \text{threshold}]\,\Big|
$$
where $[\cdot]$ is the indicator function (1 if the condition is true, 0 otherwise).

### 8. Slope Sign Changes (SSC)
$$
\text{SSC} = \sum_{i=2}^{N-1} \begin{cases}
1, & (x_i - x_{i-1})(x_i - x_{i+1}) > 0 \\
0, & \text{otherwise}
\end{cases}
$$

### 9. Standard Deviation (STD)
$$
\text{STD} = \sqrt{\frac{1}{N-1} \sum_{i=1}^{N} (x_i - \bar{x})^2}
$$

### 10. Skewness
$$
\text{SKEW} =
\frac{\dfrac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^3}
     {\left(\dfrac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2\right)^{3/2}}
$$

### 11. Kurtosis
$$
\text{KURT} =
\frac{\dfrac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^4}
     {\left(\dfrac{1}{N} \sum_{i=1}^{N} (x_i - \bar{x})^2\right)^2}
$$

### 12. Integrated EMG (IEMG)
$$
\text{IEMG} = \sum_{i=1}^{N} |x_i|
$$

### 13. Willison Amplitude (WAMP)
With a fixed threshold $\varepsilon$ (in the code: $\varepsilon = 0.05$):
$$
\text{WAMP} = \sum_{i=1}^{N-1}
\begin{cases}
1, & |x_{i+1} - x_i| > \varepsilon \\
0, & \text{otherwise}
\end{cases}
$$

### 14. Log Detector
$$
\text{LOG} = \log\left( \sum_{i=1}^{N} |x_i| \right)
$$

### 15. Signal Energy
$$
E = \sum_{i=1}^{N} x_i^2
$$

### 16. Waveform Length Ratio (WL Ratio)
$$
\text{WLR} = \frac{1}{N} \sum_{i=1}^{N-1} |x_{i+1} - x_i|
$$

### 17. Autocorrelation (lag 0, normalized)
Let $R_{xx}(k)$ be the autocorrelation and $R_{xx}(0)$ its value at lag 0. With MATLAB’s `xcorr(sig, 'coeff')`,
$$
\text{AC} = R_{xx}(0) = 1
$$
(because it is normalized by the zero-lag energy). In the code, the stored value is the center element of the normalized autocorrelation vector.

### 18. Energy Ratio (First Half vs Total)
$$
\text{ER} =
\frac{\sum_{i=1}^{\lfloor N/2 \rfloor} x_i^2}
     {\sum_{i=1}^{N} x_i^2}
$$

## Frequency-Domain Features

Compute the discrete Fourier transform (DFT) of the signal:
$$
Y(k) = \sum_{n=0}^{N-1} x_{n+1} e^{-j 2\pi kn / N}, \quad k = 0, 1, \dots, N-1.
$$
Define the two-sided spectrum magnitude
$$
P_2(k) = \frac{|Y(k)|}{N}
$$
and the one-sided spectrum
$$
P_1(k) = \begin{cases}
P_2(k), & k = 0, \tfrac{N}{2} \\
2P_2(k), & 1 \le k \le \tfrac{N}{2}-1
\end{cases}
$$
for $k = 0,1,\dots,\tfrac{N}{2}$.

Let $K = \tfrac{N}{2}+1$ be the number of one-sided bins, and index them as $k = 0,1,\dots,\tfrac{N}{2}$.

### 19. Mean Frequency Magnitude
$$
\text{MFM} = \frac{1}{K} \sum_{k=0}^{N/2} P_1(k)
$$

### 20. Median Frequency Magnitude
$$
\text{MEDFM} = \operatorname{median}\big(P_1(0), P_1(1), \dots, P_1(N/2)\big)
$$

### 21. Maximum Frequency Magnitude
$$
\text{MAXFM} = \max_{0 \le k \le N/2} P_1(k)
$$

### 22. Peak Frequency Index
$$
\text{PFI} = \arg\max_{0 \le k \le N/2} P_1(k)
$$

### 23. Mean Frequency (MNF)
Let the (normalized) frequency values be
$$
 f_k = \frac{k}{N}, \quad k = 0, 1, \dots, N/2.
$$
Normalize the spectrum:
$$
P_\text{norm}(k) = \frac{P_1(k)}{\sum_{m=0}^{N/2} P_1(m)}.
$$
Then
$$
\text{MNF} = \sum_{k=0}^{N/2} f_k \, P_\text{norm}(k)
$$

### 24. Frequency Standard Deviation
$$
\text{FSTD} = \sqrt{\sum_{k=0}^{N/2} (f_k - \text{MNF})^2 \, P_\text{norm}(k)}
$$
