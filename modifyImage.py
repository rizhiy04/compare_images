import random
import cv2
import numpy as np


def modify(img):
    img = __rotation(img)
    img = __vertical_shift(img)
    img = __horizontal_shift(img)
    img = __zoom(img)
    img = __brightness(img)
    return img


def __fill(img, h, w):
    img = cv2.resize(img, (h, w), cv2.INTER_CUBIC)
    return img


def __horizontal_shift(img, ratio=0.15):
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = w * ratio
    if ratio > 0:
        img = img[:, :int(w - to_shift), :]
    if ratio < 0:
        img = img[:, int(-1 * to_shift):, :]
    img = __fill(img, h, w)
    return img


def __vertical_shift(img, ratio=0.15):
    ratio = random.uniform(-ratio, ratio)
    h, w = img.shape[:2]
    to_shift = h * ratio
    if ratio > 0:
        img = img[:int(h - to_shift), :, :]
    if ratio < 0:
        img = img[int(-1 * to_shift):, :, :]
    img = __fill(img, h, w)
    return img


def __brightness(img, low=0.5, high=1.5):
    value = random.uniform(low, high)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def __zoom(img, value=0.8):
    value = random.uniform(value, 1)
    h, w = img.shape[:2]
    h_taken = int(value * h)
    w_taken = int(value * w)
    h_start = random.randint(0, h - h_taken)
    w_start = random.randint(0, w - w_taken)
    img = img[h_start:h_start + h_taken, w_start:w_start + w_taken, :]
    img = __fill(img, h, w)
    return img


def __rotation(img, angle=13):
    angle = int(random.uniform(-angle, angle))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w / 2), int(h / 2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h))
    return img
