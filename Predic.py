import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
import keras.utils as image
import cv2 as cv


def prediccion(img):

    # Cargar el modelo
    new_model = keras.models.load_model('model.keras')
    img = cv.resize(img, (64, 64))  # Ajustar el tamaño de la imagen
    img = np.expand_dims(img, axis=0)
    result = new_model.predict(img)  # Realizar la prediccion

    # training_set.class_indices
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'

    return prediction


if __name__ == '__main__':
    pass

    # prediccion()
