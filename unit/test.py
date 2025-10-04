import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, get_window

# ---------------------------------------------------
# Simulate a noisy tremor-like signal
# ---------------------------------------------------
fs = 100  # sample rate (Hz)
t = np.arange(0, 10, 1/fs)  # 10 seconds of data

# Tremor ~6 Hz plus some noise
tremor_freq = 6
signal = 0.5 * np.sin(2 * np.pi * tremor_freq * t) + 0.2 * np.random.randn(len(t))

# ---------------------------------------------------
# Regular FFT (single estimate)
# ---------------------------------------------------
fft_vals = np.fft.rfft(signal)
fft_freqs = np.fft.rfftfreq(len(signal), 1/fs)
fft_power = np.abs(fft_vals)**2 / len(signal)

# ---------------------------------------------------
# Welch’s method
# ---------------------------------------------------
f_welch, pxx_welch = welch(signal, fs=fs, window=get_window('hann', 256), nperseg=256, noverlap=128)

# ---------------------------------------------------
# Plot comparison
# ---------------------------------------------------
plt.figure(figsize=(12,6))

# Raw FFT
plt.subplot(1,2,1)
plt.plot(fft_freqs, fft_power, color='gray')
plt.title("Raw FFT (noisy)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power")
plt.xlim(0, 20)

# Welch PSD
plt.subplot(1,2,2)
plt.semilogy(f_welch, pxx_welch, color='blue')
plt.title("Welch’s Method (smoothed PSD)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (log scale)")
plt.xlim(0, 20)

plt.tight_layout()
plt.show()
