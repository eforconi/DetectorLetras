import h5py
import cv2
import numpy as np

f = h5py.File("../dataset/train.h5", "r")
print(f["train_images"].shape)
print(f["train_tags"].shape)
print(np.array(f["train_tags"]))

f.close()
