import pyaudio
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.fftpack import fft

CHUNK = 1024 * 4  # how much audio sample per frame are we gonna display
FORMAT = pyaudio.paInt16
CHANNELS = 1  # mono sound
RATE = 44100  # samples /sec

p = pyaudio.PyAudio()  # main pyaudio object

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    output=True,
    frames_per_buffer=CHUNK
)

fig, ax = plt.subplots()

x = np.arange(0, 2 * CHUNK, 2)
line, = ax.plot(x, np.random.rand(CHUNK))
ax.set_ylim(0,255)
ax.set_xlim(0,CHUNK)


def animate(i):
    data = stream.read(CHUNK)  # data is in bytes not getting integers convert to integers for matplotlib
    data_int = np.array(struct.unpack(str(2 * CHUNK) + 'B', data), dtype=np.uint8)[::2] + 127
    line.set_ydata(data_int)
    return line,


ani = animation.FuncAnimation(fig, animate, blit=True, interval=25)

plt.show()

stream.stop_stream()
stream.close()
p.terminate()
