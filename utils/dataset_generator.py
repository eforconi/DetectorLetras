import h5py
import cv2
import numpy as np


def read_image(path):
    raw_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = np.reshape(raw_image, (32 * 32))
    return image


def create_dataset(filelist):
    f = h5py.File("../dataset/train.h5", "w")
    n = len(filelist)
    f.create_dataset("train_images", (n, 32*32), np.uint8)
    f.create_dataset("train_tags", (n, 1), np.uint8)

    for i in range(n):
        path = filelist[i]
        img = read_image(path)
        f["train_images"][i] = img
        f["train_tags"][i] = 1 if "/a/" in path else 0

    f.close()





if __name__ == '__main__':
    filelist = [
        "../images/a/img0.png",
        "../images/a/img1.png",
        "../images/a/img2.png",
        "../images/b/img0.png",
        "../images/b/img1.png",
        "../images/b/img2.png"
    ]
    create_dataset(filelist)
