import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 2000
duration = 1.0
frequency1 = int(input("Enter 1st wave frequency:\n"))
frequency2 = int(input("Enter 2nd wave frequency:\n"))
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

sine_wave1 = np.sin(2 * np.pi * frequency1 * t)
sine_wave2 = np.sin(2 * np.pi * frequency2 * t)

# Ensure that the length of the input signal is a power of 2
length = 2**int(np.log2(len(t)))
sine_wave1 = sine_wave1[:length]
sine_wave2 = sine_wave2[:length]
sum_of_signal = sine_wave1 + sine_wave2

def fft(x):
    N = len(x)
    
    if N <= 1:
        return x
    
    even = fft(x[::2])
    odd = fft(x[1::2])

    # Zero-pad the shorter component to match the length of the longer one
    if len(even) < len(odd):
        even = np.concatenate((even, np.zeros(len(odd) - len(even))))
    else:
        odd = np.concatenate((odd, np.zeros(len(even) - len(odd))))

    T = [np.exp(-2j * np.pi * k / N) * odd[k] for k in range(N // 2)]

    return [even[k] + T[k] for k in range(N // 2)] + [even[k] - T[k] for k in range(N // 2)]

# Calculate the FFT of the combined signal
X_fft = fft(sum_of_signal)

# Create a frequency axis
N = len(X_fft)
frequency_axis = np.arange(N) * sampling_rate / N

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t[:length], sine_wave1)
plt.title('Sine Wave 1')
plt.xlabel('Time (s)')

plt.subplot(2, 2, 2)
plt.plot(t[:length], sine_wave2)
plt.title('Sine Wave 2')
plt.xlabel('Time (s)')

plt.subplot(2, 2, 3)
plt.plot(frequency_axis, np.abs(X_fft))
plt.title('FFT of Combined Signal')
plt.xlabel('Frequency (Hz)')

plt.xlim(0, sampling_rate / 2)
plt.tight_layout()
plt.show()
