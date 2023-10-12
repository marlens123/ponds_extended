import numpy as np
from train import train_wrapper, final_train
import cv2
import os

images = np.load('./data/ds_np_extended_16/480_im_extended_16.npy')
masks = np.load('./data/ds_np_extended_16/480_ma_extended_16.npy')

images_norm = np.load('./data/ds_np_extended/480_im_norm_extended.npy')

test_images = np.load('./data/test_ds_np/480_im.npy')
test_masks = np.load('./data/test_ds_np/480_ma.npy')

#time = final_train(images, masks, test_images, test_masks, loss='focal_dice', weight_classes=True, epochs=500, im_size=480, base_pref='extended', augmentation='on_fly', mode=4, use_dropout=True, train_transfer='imagenet', batch_size=2)
time = final_train(images, masks, test_images, test_masks, loss='focal_dice', weight_classes=True, epochs=500, im_size=480, base_pref='test02', augmentation='on_fly', mode=4, use_dropout=True, train_transfer='imagenet', batch_size=2)
