import numpy as np
import matplotlib.pyplot as plt

sampling_rate = 44100
duration = 1.0
frequency1 = int(input("Enter 1st wave frequency:\n"))
frequency2 = int(input("Enter 2nd wave frequency:\n"))

t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

sine_wave1 = np.sin(2 * np.pi * frequency1 * t)
sine_wave2 = np.sin(2 * np.pi * frequency2 * t)

sum_of_signal = sine_wave1 + sine_wave2

def DFT(x):
    N = len(x)
    X = np.zeros(N, dtype=complex)
    for k in range(N):
        X[k] = sum(x[n] * np.exp(-2j * np.pi * k * n / N) for n in range(N))
    return X

X = DFT(sum_of_signal)  # Calculate the DFT of the combined signal

# Create a manual frequency axis
N = len(X)
frequency_axis = np.arange(N) * sampling_rate / N

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, sine_wave1)
plt.title('Sine Wave 1')
plt.xlabel('Time (s)')

plt.subplot(2, 2, 2)
plt.plot(t, sine_wave2)
plt.title('Sine Wave 2')
plt.xlabel('Time (s)')

plt.subplot(2, 1, 2)
plt.plot(frequency_axis, np.abs(X))
plt.title('DFT of Combined Signal')
plt.xlabel('Frequency (Hz)')
plt.xlim(0, 1000)
plt.tight_layout()
plt.show()

