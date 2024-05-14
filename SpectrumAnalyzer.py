import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter1d
import time
import sounddevice as sd
from scipy.fftpack import fft


def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    snr = np.where(sd == 0, 0, m/sd)
    snr = snr.item() # converts from numpy array to a float
    return snr

def thd(abs_data):
    sq_sum=0.0
    for r in range( len(abs_data)):
       sq_sum = sq_sum + (abs_data[r])**2

    sq_harmonics = sq_sum -(max(abs_data))**2.0
    thd = 100*sq_harmonics**0.5 / max(abs_data)

    return thd

# Parameters
CHUNK = 1024
RATE = 44100
FORMAT = pyaudio.paInt16 # Defines the audio format as 16-bit integers.
CHANNELS = 1  #  Defines the number of audio channels (mono sound).

p = pyaudio.PyAudio()  # Creates a PyAudio object, which serves as the main interface for accessing audio I/O functionalities.
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
      
    
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# Initialize plot
fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(25,8))
line, = ax.plot([], [])
line_fft, = ax2.semilogx([], [])
ax.set_title('AUDIO WAVEFORM')
ax.set_xlabel('samples')
ax.set_ylabel('volume')
ax.set_ylim(-250, 250)
ax.set_xlim(0, CHUNK)
plt.setp(ax, xticks=[0, 2 * CHUNK, CHUNK], yticks=[-500,-250,0, 250, 500])
ax2.set_xlim(20, RATE / 2)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('SNR')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('THD')
snr_vals = []
thd_vals = []

# Function to update plots
def animate(i, stream):
    data, _ = stream.read(CHUNK)
    data = np.frombuffer(data, dtype='int16')
    
    # Calculate SNR
    snr = np.mean(data) / np.std(data)
    snr_vals.append(snr)


    # Compute FFT
    y_fft = fft(data)
    fft_scaled = np.abs(y_fft[:CHUNK]) * 2 / (256 * CHUNK)

    thd_val = thd(np.abs(y_fft))
    thd_vals.append(thd_val)
    
    # Update plots
    line.set_ydata(data)
    line_fft.set_ydata(fft_scaled)

    # Update SNR and THD plots
    ax3.clear()
    ax4.clear()
    ax3.set_title(f'Signal to noise ratio: {snr:.2f} dB')
    ax4.set_title(f'Total Harmonic Distortion: {thd_val:.2f} dB')
    ax3.plot(range(len(snr_vals)), snr_vals, color='blue')
    ax4.plot(range(len(thd_vals)), thd_vals, color='red')

    return line, line_fft, ax3, ax4

stream = sd.InputStream(channels=1, samplerate=RATE, blocksize=CHUNK, dtype='int16')
stream.start()

ani = animation.FuncAnimation(fig, animate, fargs=(stream,), blit=False, interval=50)

plt.show()

stream.stop()
stream.close()
