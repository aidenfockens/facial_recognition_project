
import tensorflow as tf
import numpy as np

print("fjdsla;fjkdlas")
from PIL import Image
print("fjdsla;fjkdlas")

import cv2
print("fjdsla;fjkdlas")

print("hi")


def recognizeRace(file):
    race = tf.keras.models.load_model('RaceRecognition')
    img = cv2.imread(file)
    img = cv2.resize(img,dsize = (100,100), interpolation = cv2.INTER_CUBIC) 
    img = img.astype('float32')
    img /= 255
    img = np.array(img)
    img = img.reshape(1,100,100,3)
    print("middle")

    return race.predict(img)
print("hello")
def recognizeAge(file):
    age = tf.keras.models.load_model('AgeRecognition')

    img = cv2.imread(file)
    img = cv2.resize(img,dsize = (100,100), interpolation = cv2.INTER_CUBIC) 
    img = img.astype('float32')
    img /= 255
    img = np.array(img)
    img = img.reshape(1,100,100,3)

    return age.predict(img)

def recognizeGender(file):
    gender = tf.keras.models.load_model('GenderRecognition')

    img = cv2.imread(file)
    img = cv2.resize(img,dsize = (100,100), interpolation = cv2.INTER_CUBIC) 
    img = img.astype('float32')
    img /= 255
    img = np.array(img)
    img = img.reshape(1,100,100,3)

    return gender.predict(img)

def recognizeEmotion(file):
    emotion = tf.keras.models.load_model('EmotionRecognition')

    img = cv2.imread(file)
    img = cv2.resize(img,dsize = (48,48), interpolation = cv2.INTER_CUBIC) 
    img = img.astype('float32')
    img /= 255
    img = np.array(img)
    img = img.reshape(1,48,48,3)

    return emotion.predict(img)
    
print(recognizeAge("test1.jpg"))


print("start")


print("done")