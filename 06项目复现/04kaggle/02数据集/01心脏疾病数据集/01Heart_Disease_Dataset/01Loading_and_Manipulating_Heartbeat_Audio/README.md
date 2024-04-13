# Loading and Manipulating Heartbeat Audio

This notebook will show you how to load and manipulate audio data in Python.

The Heartbeat Sounds dataset is primarily audio-based: all of the heartbeat sounds are stored as WAV files that record either normal or abnormal heartbeats. So let's learn how to load and play with WAVs in Python.

In general, uncompressed audio is stored as a sequence of numbers that indicate the amplitude of the recorded sound pressure at each time point. In the WAV standard, these numbers are packed into a bytestring. The interpretation of this byestring depends primarily on two factors: first, the sampling rate, usually given in Hertz, which indicates how many number samples comprise a second's worth of data; and second, the bit depth (or sample width), which indicates how many bits comprise a single number.

These parameters, along with other parameters like the number of channels (e.g., is the audio mono or stereo) are stored in the header of the WAV file.

The wave library handles the parsing of WAV file headers, which include the parameters mentioned above. Let's load the wave library and use it to open a sound file.




下面这段代码是用于读取WAV格式的音频文件，并打印出音频帧的前20个字节的内容。下面是对每行代码的详细中文注释：

```python
# 导入Python的wave模块，它提供了读取和写入WAV文件的功能。
import wave

# 定义一个变量FNAME，存储音频文件的路径。这个路径是一个示例，指向'../input/set_a/normal__201101070538.wav'。
# 这个路径可能需要根据实际文件的位置进行调整。
FNAME = '../input/set_a/normal__201101070538.wav'

# 使用wave模块的open函数打开指定路径的WAV文件，返回一个wave对象。
# 这个对象提供了许多方法来操作和访问音频文件的数据。
f = wave.open(FNAME)

# 从打开的WAV文件中读取所有的音频帧，并将它们作为一个字节串存储在变量frames中。
# readframes函数的参数-1表示读取文件中的所有帧。
frames = f.readframes(-1)

# 打印出frames字节串的前20个字节。这通常用于快速查看音频数据的开始部分。
# 对于音频分析或处理，这些数据通常是二进制的，需要进一步的处理才能转换为可听的音频。
print(frames[:20])
```

这段代码的主要作用是读取WAV格式的音频文件，并快速查看文件的音频数据。在实际应用中，这些数据可以用于音频的特征提取、分类、分析等任务。打印出的前20个字节对于了解音频数据的基本结构和内容有一定的帮助，但由于音频数据通常是大量的二进制数据，所以这里的打印输出仅用于演示目的。


结果：
```python
b'\x0e\x00\x11\x00\x0e\x00\x00\x00\x02\x00\x0b\x00\xfe\xff\x05\x00\x05\x00\x04\x00'
```


So frames now holds the entire bytestring representing all the audio samples in the sound file. We need to unpack this bytestring into an array of numbers that we can actually work with.

The first question is: how many bytes represent a single observation? In my experience in voice recording, 16-bit and 24-bit are the most common sample widths, but you can find a whole collection on Wikipedia.

Powers of 2 tend to be the easiest to work with, and luckily for us the heartbeat audio seems to be 16-bit. We can check this by using the getsamplewidth() method on the wave file:



下面这行代码是用于输出通过`wave`模块打开的WAV文件的每个采样的宽度（以字节为单位）。下面是对这行代码的详细中文注释：

```python
# 调用已经打开的WAV文件对象f的getsampwidth方法。
# getsampwidth方法用于获取音频文件中每个采样值的宽度，即每个采样值占用的字节数。
# 采样宽度是音频数据的重要属性之一，它影响音频的质量和大小。
# 例如，一个采样宽度为2的音频文件意味着每个采样值使用2个字节存储，通常是16位的PCM编码音频。
sampwidth = f.getsampwidth()

# 打印出每个采样的宽度。这个值有助于了解音频文件的编码格式和存储密度。
print(sampwidth)
```

这行代码通常用于音频文件的处理和分析中，因为它提供了音频数据采样精度的信息。例如，采样宽度为1通常表示8位音频，而采样宽度为2则表示16位音频。了解采样宽度对于正确解析和处理音频数据至关重要，尤其是在进行音频数据转换或压缩时。


结果：

```python
2
```

代码`print(f.getsampwidth())`的输出结果为`2`，这表示该WAV文件中每个采样值的宽度是2个字节。这个信息对于理解音频文件的采样精度和数据表示方式非常重要。下面是对这个结果的详细解读：

1. **采样宽度（Sample Width）**: 采样宽度指的是每个音频采样值所占用的字节数。在这个例子中，`2`表示每个采样值使用2个字节（16位）来存储。

2. **音频质量**: 16位的采样宽度意味着音频信号的每个采样值可以表示从-32768到32767的整数值（如果是有符号整数的话）。这提供了相对较好的音频质量，能够捕捉到音频信号的细节和动态范围。

3. **数据表示**: 16位的采样宽度通常采用有符号整数（signed integer）来表示，这样可以表示正负值，适用于模拟音频信号的振幅表示。

4. **文件大小**: 使用16位采样宽度的音频文件相比于8位或其他更低位数的采样宽度会占用更多的存储空间。然而，它提供了更高的动态范围和更好的音质。

5. **兼容性**: 16位采样宽度是一种非常常见的音频格式，被广泛支持和使用在各种音频播放和编辑软件中。

了解这些信息对于后续的音频处理和分析非常重要，例如在进行音频信号的数字化处理、滤波、特征提取或格式转换时，都需要考虑到采样宽度这一参数。



The result of getsamplewidth() is in bytes, so multiply it by 8 to get the bit depth. Since the result from the call is 2, that means we're looking at a 16-bit file.

We'll unpack the bytestring by using the `struct` library in Python. struct requires a format string based on C format characters, which you can take a look at on the documentation page for Python's struct library.

We're in luck with the 16-bit depth, since the struct library prefers powers of 2. 16 bits corresponds to 2 bytes, so we'll use the signed format that corresponds to 2 bytes; according to the C format characters, we should use the format character 'h'.

A slight trick in the `struct` library is that it wants its format string to exactly match the expected size, so we have to multiply the format character 'h' by the number of frames in the bytestring:










































































