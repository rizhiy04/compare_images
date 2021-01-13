from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
from modifyImage import modify
import random
import data_storage
from data_storage import ImageData

image_folder = './images/downloaded'
image_size = 64
image_duplicate = 3


def __generate_data(image_names):
    image_datas = []
    count = 1

    for img_name in image_names:
        print(str(count) + "/" + str(len(image_names)))
        count = count + 1

        image = cv2.imread(image_folder + '/' + img_name)
        if type(image) is np.ndarray:
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            for x in range(0, image_duplicate):
                modified_image = modify(image)
                image_data = ImageData(image, modified_image, True)
                image_datas.append(image_data)

            for index in range(0, 4):
                second_image_name = image_names[random.randint(0, (len(image_names) - 1))]
                if img_name != second_image_name:
                    second_image = cv2.imread(image_folder + '/' + second_image_name)
                    second_image = cv2.resize(second_image, (image_size, image_size), interpolation=cv2.INTER_AREA)
                    image_data = ImageData(image, second_image, False)
                    image_datas.append(image_data)

    return image_datas


def create_data():
    images = [f for f in listdir(image_folder) if isfile(join(image_folder, f))]
    test_images_count = int(len(images) * 0.7)

    training_images = images[:test_images_count]
    test_images = images[test_images_count:]

    training_data = __generate_data(training_images)
    test_data = __generate_data(test_images)

    data_storage.save_files(training_data, test_data)
