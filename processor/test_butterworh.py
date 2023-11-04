import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, freqz

# Parameters
fs = 44100  # Sampling frequency
f_signal = 1000  # Frequency of the sinusoid
duration = 0.01  # Duration in seconds (for the time axis from 0 to 0.01 seconds)

# Time array for the updated duration
t = np.linspace(0, duration, int(fs * duration), endpoint=False)

# Generate a clean sinusoidal signal for the updated duration
clean_signal = np.sin(2 * np.pi * f_signal * t)

# Add some noise to the sinusoidal signal to create a distorted signal for the updated duration
distorted_signal = clean_signal + 0.5 * np.random.normal(size=t.shape)

# Filter design: Butterworth Low-Pass Filter
cutoff_frequency = 1500  # Cutoff frequency for the low-pass filter
order = 6  # Order of the filter

# Get the filter coefficients
b, a = butter(order, cutoff_frequency / (0.5 * fs), btype='low')

# Apply the zero-phase filter to the distorted signal to get the zero-phase output signal
zero_phase_output_signal = filtfilt(b, a, distorted_signal)

# Compute the error signal for the zero-phase output
zero_phase_error_signal = distorted_signal - zero_phase_output_signal

# Set the style to dark background for plotting
plt.style.use('dark_background')

# Plot everything with zero-phase filtering
fig, ax = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

# Input signal
ax[0].plot(t, distorted_signal, label='Input Signal')
ax[0].set_title('Input Signal (Distorted Sinusoid)')
ax[0].set_ylabel('Amplitude')
ax[0].grid(True)
ax[0].legend()

# Zero-phase output signal
ax[1].plot(t, zero_phase_output_signal, label='Zero-phase Output Signal', color='orange')
ax[1].set_title('Zero-phase Output Signal (Filtered Sinusoid)')
ax[1].set_ylabel('Amplitude')
ax[1].grid(True)
ax[1].legend()

# Zero-phase error signal
ax[2].plot(t, zero_phase_error_signal, label='Zero-phase Error Signal', color='green')
ax[2].set_title('Zero-phase Error Signal (Difference Signal)')
ax[2].set_xlabel('Time [s]')
ax[2].set_ylabel('Amplitude')
ax[2].grid(True)
ax[2].legend()

plt.tight_layout()
plt.show()