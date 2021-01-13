import random

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Dense, Flatten, concatenate, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from data_storage import read_files, ImageData

input_shape = (64, 64, 3)


def create_model():
    first_img_input = Input(shape=input_shape)
    first_img_model = __create_input_layers(first_img_input)

    second_img_input = Input(shape=input_shape)
    second_img_model = __create_input_layers(second_img_input)

    conv = concatenate([first_img_model, second_img_model])
    conv = Flatten()(conv)

    dense = Dense(256)(conv)
    dense = LeakyReLU(alpha=0.1)(dense)
    dense = Dropout(0.5)(dense)

    output = Dense(1, activation="sigmoid")(dense)
    model = Model(inputs=[first_img_input, second_img_input], outputs=[output])
    opt = SGD(lr=0.01)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    training_data, test_data = __get_data()
    x_training_first_img, x_training_second_img, y_training = __split_to_task_and_result(training_data)
    x_test_first_img, x_test_second_img, y_test = __split_to_task_and_result(test_data)

    best_weights_file = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(best_weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks = [checkpoint]

    model.fit([x_training_first_img, x_training_second_img], y_training,
              batch_size=32, epochs=20,
              callbacks=callbacks, verbose=1,
              validation_data=([x_test_first_img, x_test_second_img], y_test),
              shuffle=True)

    model.evaluate([x_test_first_img, x_test_second_img], y_test, verbose=1)

    model.save("my_model")


def __create_input_layers(input_img):
    model = Conv2D(32, (3, 3), padding='same', input_shape=input_shape)(input_img)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPool2D((2, 2), padding='same')(model)
    model = Dropout(0.25)(model)

    model = Conv2D(64, (3, 3), padding='same')(model)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPool2D(pool_size=(2, 2), padding='same')(model)
    model = Dropout(0.25)(model)

    model = Conv2D(128, (3, 3), padding='same')(model)
    model = LeakyReLU(alpha=0.1)(model)
    model = MaxPool2D(pool_size=(2, 2), padding='same')(model)
    model = Dropout(0.4)(model)

    return model


def __get_data():
    training_data, test_data = read_files()
    random.shuffle(training_data)
    random.shuffle(test_data)
    return training_data, test_data


def __split_to_task_and_result(data):
    x_first = []
    x_second = []
    y = []

    for item in data:
        x_first.append(item.get_first_img() / 255.0)
        x_second.append(item.get_second_img() / 255.0)
        y.append(item.get_result())

    return np.array(x_first), np.array(x_second), np.array(y)

create_model()