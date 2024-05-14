import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import gaussian_filter1d
import time
import sounddevice as sd

from scipy.fftpack import fft

def compute_fft_bins_center_frequency(sampling_rate, num_data_points):
    # Generate FFT bin indices
    fft_bin_indices = np.arange(num_data_points)

    # Compute center frequency of each FFT bin
    center_frequency = fft_bin_indices * sampling_rate / num_data_points

    return center_frequency

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


CHUNK = 1024 #* 2  # Defines the chunk size, i.e., the number of audio samples per frame that will be displayed.
FORMAT = pyaudio.paInt16 # Defines the audio format as 16-bit integers.
CHANNELS = 1  #  Defines the number of audio channels (mono sound).
RATE = 44100 #44100  #  Defines the sampling rate, i.e., the number of samples per second.

p = pyaudio.PyAudio()  # Creates a PyAudio object, which serves as the main interface for accessing audio I/O functionalities.
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
        
# Opens an audio stream for both input and output, specifying the format, channels, 
# sampling rate, and chunk size.
stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

# Creates a figure with two subplots (ax and ax2) for plotting the audio waveform and its FFT. 
# The figsize parameter sets the size of the figure.
fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(25,8))

x = np.arange(0, 2 * CHUNK, 2) # Creates an array of x-values for plotting the audio waveform.
x_fft = np.linspace(0, RATE, CHUNK) # Creates an array of frequency values for plotting the FFT.

line, = ax.plot(x, np.random.rand(CHUNK)) # Initializes a line object for plotting the audio waveform in the first subplot (ax).
line_fft, = ax2.semilogx(x_fft, np.random.rand(CHUNK), '-', lw=2) # Initializes a line object for plotting the FFT in the second subplot (ax2).

ax.set_title('AUDIO WAVEFORM')
ax.set_xlabel('samples')
ax.set_ylabel('volume')
ax.set_ylim(-250, 250) #Sets the y-axis limits for the first subplot.
ax.set_xlim(0, 2 * CHUNK) #Sets the x-axis limits for the first subplot.
plt.setp(ax, xticks=[0, 2 * CHUNK, CHUNK], yticks=[-5000,0, 5000]) #Sets the x and y-axis tick marks for the first subplot.

ax2.set_xlim(20, RATE / 2) # Sets the x-axis limits for the second subplot (FFT plot).

ax3.set_xlabel('Time (s)')
ax3.set_ylabel('SNR')
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('THD')

snr_vals = []
thd_vals = []

def animate(i):

    data = stream.read(CHUNK) #Reads a chunk of audio data from the audio stream (stream) with a size defined by CHUNK.
    #data = sd.rec(frames=CHUNK, channels=1, dtype='int16', blocking=False)
    # Unpacks the binary data (data) into an array of 8-bit unsigned integers (dtype=np.uint8). 
    # The ::2 slicing selects every other element, effectively downsampling the data to half its
    # original size. It then adds 127 to each element, likely to shift the signal to be centered 
    # around zero.
    # Unpacks the binary data into a tuple of unsigned bytes 'B' by specifying the format string as '2*CHUNK' concatenated with 'B'.
    # This indicate that the data is being unpacked in chunks of 2 bytes per value.
    #data_int = np.array(struct.unpack(str(CHUNK) + 'h', data), dtype=np.uint16)[::2] 
    if len(data) > 0:
        data_int = np.array(struct.unpack(str(CHUNK) + 'h', data), dtype=np.int16)

        # Calculate SNR
        snr = signaltonoise(data_int)
        snr_vals.append(snr)

        signal_mean = np.mean(data_int)
        data_int_centered = data_int - signal_mean


        # Computes the Fast Fourier Transform (FFT) of the downsampled audio data (data_int), 
        # resulting in the frequency domain representation of the signal.
        y_fft = fft(data_int)

        # Calculate THD (Assuming distortion power as the difference between signal power and FFT power)
        thd_val = thd(np.abs(y_fft))
        thd_vals.append(thd_val)

        
        # Updates the y-data of the line object (line) representing the waveform plot to the 
        # downsampled audio data (data_int)
        line.set_ydata(data_int)#data_int)

        # Normalize FFT data appropriately for plotting
        fft_scaled = np.abs(y_fft[:CHUNK]) * 2 / (256 * CHUNK)

        # Calculate the center frequency
        max_index = np.argmax(np.abs(y_fft))
        frequency = max_index * RATE / CHUNK
        # #max_index = np.argmax(fft_scaled)
        # frequency = compute_fft_bins_center_frequency(RATE, CHUNK)#max_index * 44100 / CHUNK


        # Updates the y-data of the line object (line_fft) representing the FFT plot to the 
        # absolute values of the FFT of the first CHUNK elements of y_fft,
        # multiplied by 2 / (256 * CHUNK). 
        # This line is performing scaling and normalization of the FFT data.
        line_fft.set_ydata(fft_scaled)

        # Dynamically adjust y-axis limits of FFT plot
        # Computes the maximum value of the scaled and normalized 
        # FFT data for the first CHUNK elements of y_fft.
        max_fft_value = np.max(np.abs(y_fft[:CHUNK]) * 2 / (256 * CHUNK))


        # Dynamically adjusts the y-axis limits of the FFT plot (ax2) to 
        # ensure that the highest peak of the FFT is still visible, with some additional padding.
        ax2.set_ylim(0, max_fft_value * 1.2)  # Add some padding

        ax.set_title(f'AUDIO WAVEFORM')
        ax2.set_title(f'FFT: {frequency:.2f} Hz ')
        ax3.clear()
        ax4.clear()
        ax3.set_title(f'Signal to noise ratio: {snr:.2f} dB')  
        ax4.set_title(f'Total Harmonic Distortion: {thd_val:.2f} dB')      
        ax3.plot(range(len(snr_vals)), snr_vals, color='blue')
        ax4.plot(range(len(thd_vals)), thd_vals, color='red')


        return line, line_fft, ax3, ax4


ani = animation.FuncAnimation(fig, animate, blit=False, interval=250)

plt.show()

stream.stop_stream()
stream.close()
p.terminate()

plt.plot(snr_vals)
plt.title('SNR Values')

