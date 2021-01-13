from tensorflow.keras.models import load_model
import cv2
import numpy as np

first_img = cv2.imread("./images/11.jpeg")
second_img = cv2.imread("./images/12.jpeg")

first_imgs = []
second_imgs = []

first_img = cv2.resize(first_img, (64, 64), interpolation=cv2.INTER_AREA)
second_img = cv2.resize(second_img, (64, 64), interpolation=cv2.INTER_AREA)

first_imgs.append(first_img)
second_imgs.append(second_img)

first_imgs = np.array(first_imgs)
second_imgs = np.array(second_imgs)

model = load_model("my_model")
prediction = model.predict(x=[first_imgs, second_imgs])
print(prediction)
