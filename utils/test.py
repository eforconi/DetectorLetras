import h5py
import cv2
import numpy as np

with h5py.File("../dataset/train.h5", "r") as f:
    print("train_images: ", f["train_images"].shape)
    print("train_tags: ", f["train_tags"].shape)

with h5py.File("../dataset/test.h5", "r") as f:
    print("test_images: ", f["test_images"].shape)
    print("test_tags: ", f["test_tags"].shape)

