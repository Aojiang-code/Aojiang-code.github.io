#!/usr/bin/env python
import PCF8591 as ADC
import scipy.io.wavfile as wavf
import RPi.GPIO as GPIO
import time
import numpy as np
import tensorflow as tf
from python_speech_features import mfcc,delta
from LCDnew import Adafruit_CharLCD as LCD
from time import sleep


GPIO.setmode(GPIO.BOARD)
DO = 16


def setup():
    ADC.setup(0x48)
    GPIO.setup(DO, GPIO.IN)


def loop():
    status = 1
    wavfile=[]
    while len(wavfile)<20000:
        wavfile.append(ADC.read(0))
        tmp = GPIO.input(DO)
        if tmp != status:
            print('读取失败！')
            status = tmp
        time.sleep(5e-4)
    GPIO.setup(DO, GPIO.IN)
    return wavfile


def getMFCCMap(y, sr=2000):
    mfcc0 = mfcc(y, sr, numcep=13)
    mf1=delta(mfcc0,1)
    mf2 = delta(mfcc0, 2)
    mfcc_all=np.hstack((mfcc0,mf1,mf2))
    mfcc_all1=mfcc_all.reshape(499,39,1)
    return mfcc_all1

lcd=LCD()
lcd.clear()
lcd.message('Testing...')
GPIO.cleanup()
setup()
wav=loop()
wav=np.array(wav)
GPIO.cleanup()
lcd=LCD()
lcd.clear()
lcd.message('Finish Testing!')
sleep(1)
lcd.clear()
lcd.message('Analyzing...')
fs = 2000
out_f = 'hs.wav'
wavf.write(out_f, fs, wav)

#signal,f=lib.load('./hs.wav',sr=None)
sr, signal = wavf.read('a0001.wav')
Map=getMFCCMap(signal[:10000],2000)
Map=Map.reshape(1,499,39,1)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="./Lite/tfl.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on random input data.
input_shape = input_details[0]['shape']

# 修改下面这行为自己的数据(要求是499*39*1)
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
#input_data=Map
input_data=np.array(Map,dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
result=np.argmax(output_data[0])
if result==0:
    lcd.clear()
    lcd.message('Abnormal!\nBe careful!')
elif result==1:
    lcd.clear()
    lcd.message('Normal!')
elif result==2:
    lcd.clear()
    lcd.message('Noisy!\n Do again!')