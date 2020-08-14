import h5py
import cv2
import numpy as np
import os
import random


def get_images_list():
    list = []
    add_images("../images/", list)
    return list


def add_images(dir, list):
    for entry in os.scandir(dir):
        if entry.is_file():
            list.append(entry.path)
        elif entry.is_dir():
            add_images(entry.path, list)


def read_image(path):
    raw_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = np.reshape(raw_image, (32 * 32))
    return image


def create_dataset(filelist, name):
    f = h5py.File(f"../dataset/{name}.h5", "w")
    # Create train dataset
    n = len(filelist)
    f.create_dataset(f"{name}_images", (n, 32*32), np.uint8)
    f.create_dataset(f"{name}_tags", (n, 1), np.uint8)

    print(f"Adding {n} images to {name} dataset")
    for i in range(n):
        path = filelist[i]
        img = read_image(path)
        f[f"{name}_images"][i] = img
        f[f"{name}_tags"][i] = 1 if "/a/" in path else 0

    f.close()

def create_datasets(training_set, test_set):
    create_dataset(training_set, "train")
    create_dataset(test_set, "test")


def split_sets(images, split):
    n = len(images)
    training_size = int(n * split / 100)
    training_set = []
    for i in range(training_size):
        image = random.choice(images)
        training_set.append(image)
        images.remove(image)

    return training_set, images


if __name__ == '__main__':
    images = get_images_list()
    training_set, test_set = split_sets(images, 80)
    # print(len(training_set))
    # print(len(test_set))
    create_datasets(training_set, test_set)
