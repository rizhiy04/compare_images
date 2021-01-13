from tensorflow.keras.models import load_model
import cv2
import numpy as np
from data_storage import read_files, ImageData
from cnn import __split_to_task_and_result


def compare_to_images_by_path():
    first_imgs = []
    second_imgs = []
    first_img = cv2.imread("./images/1.jpeg")
    second_img = cv2.imread("./images/2.jpeg")

    first_img = cv2.resize(first_img, (64, 64), interpolation=cv2.INTER_AREA)
    second_img = cv2.resize(second_img, (64, 64), interpolation=cv2.INTER_AREA)

    first_img = first_img / 255.0
    second_img = second_img / 255.0

    first_imgs.append(first_img)
    second_imgs.append(second_img)

    __predict(first_imgs, second_imgs)


def compare_from_training_data(image_num=0):
    first_imgs = []
    second_imgs = []
    training_data, test_data = read_files()
    x_training_first_img, x_training_second_img, y_training = __split_to_task_and_result(training_data)

    first_imgs.append(x_training_first_img[image_num])
    second_imgs.append(x_training_second_img[image_num])

    __predict(first_imgs, second_imgs)


def __predict(first_imgs, second_imgs):
    first_imgs = np.array(first_imgs)
    second_imgs = np.array(second_imgs)
    model = load_model("my_model")
    prediction = model.predict(x=[first_imgs, second_imgs])
    print(prediction)


# compare_from_training_data(35)
compare_to_images_by_path()
