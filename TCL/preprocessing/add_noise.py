import numpy as np
import os
from utils import *

def gauss_noise(img, sigma):
    temp_img = np.float64(np.copy(img))
    c, h, w = temp_img.shape
    noisy_img = np.zeros(temp_img.shape, np.float64)
    for idx in range(c):
        noise = np.random.randn(h, w) * sigma
        noisy_img[idx, :, :] = temp_img[idx, :, :] + noise
    return noisy_img.astype(np.uint8)

def add_noise(origin_path, dest_name, sigma):
    i = 0
    for root, dirs, files in os.walk(origin_path):
        dst_pathname = root.split('/')
        dst_pathname[-3] = dest_name
        for file in files:
            data = np.load(os.path.join(root, file))['data']
            noisy_data = gauss_noise(data, sigma=sigma)
            dst_abs_path = os.path.join('/'.join(dst_pathname), file)
            save2npz(dst_abs_path, data=noisy_data)
            i += 1
    print("Done! Totally %d images" % i)

if __name__ == "__main__":
    dest_name = 'LRW_crop_gray_noise_51'
    add_noise('/home/sunlichao/TCL_LipReading/datasets/LRW_h96w96_mouth_crop_gray', dest_name, sigma = 51)
    # for sigma in range(5, 24, 5):
    #     dest_name = 'LRW_crop_gray_noise_' + str(sigma)
    #     add_noise('/home/sunlichao/TCL_LipReading/datasets/LRW_h96w96_mouth_crop_gray', dest_name, sigma = sigma)