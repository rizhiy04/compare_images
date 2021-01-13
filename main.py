from tensorflow.keras.models import load_model
import cv2
import numpy as np
from data_storage import read_files, ImageData
from cnn import __split_to_task_and_result
import cnn


def compare_to_images_by_path():
    first_img = cnn.__prepare_image_before_predict("./images/1.jpeg")
    second_img = cnn.__prepare_image_before_predict("./images/2.jpeg")

    __predict(first_img, second_img)


def compare_from_training_data(image_num=0):
    first_imgs = []
    second_imgs = []
    training_data, test_data = read_files()
    x_training_first_img, x_training_second_img, y_training = __split_to_task_and_result(training_data)

    first_imgs.append(x_training_first_img[image_num])
    second_imgs.append(x_training_second_img[image_num])

    first_imgs = np.array(first_imgs)
    second_imgs = np.array(second_imgs)

    __predict(first_imgs, second_imgs)


def __predict(first_imgs, second_imgs):
    model = load_model("my_model")
    prediction = model.predict(x=[first_imgs, second_imgs])
    print(prediction)


# compare_from_training_data(35)
compare_to_images_by_path()
