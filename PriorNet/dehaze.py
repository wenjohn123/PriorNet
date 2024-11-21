import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import argparse
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

parser = argparse.ArgumentParser(description='Performance')
parser.add_argument('--input_dir', default='results')
parser.add_argument('--reference_dir', default='original')

opt = parser.parse_args()
print(opt)


def dehaze_image(image_path):
	data_hazy = Image.open(image_path)
	data_hazy = (np.asarray(data_hazy) / 255.0)

	data_hazy = torch.from_numpy(data_hazy).float()
	data_hazy = data_hazy.permute(2, 0, 1)
	data_hazy = data_hazy.cuda().unsqueeze(0)

	dehaze_net = net.dehaze_net().cuda()
	dehaze_net.load_state_dict(torch.load(f'snapshots/dehazer.pth'))

	clean_image = dehaze_net(data_hazy)
	# print(image_path.split("/")[-1].split("_")[0])
	torchvision.utils.save_image(clean_image, "results/" + image_path.split("\\")[-1])


if __name__ == '__main__':
	im_path = opt.input_dir
	re_path = opt.reference_dir
	avg_psnr = 0
	avg_ssim = 0
	n = 0
	test_list = glob.glob("test_images/*")

	for image in test_list:
		dehaze_image(image)
	# print(image, "done!")
	for filename in os.listdir(im_path):
		# print(im_path + '/' + filename)
		n = n + 1
		im1 = cv2.imread(im_path + '\\' + filename)
		im2 = cv2.imread(re_path + '\\' + filename)

		(h, w, c) = im2.shape
		im1 = cv2.resize(im1, (w, h))  # reference size

		score_psnr = psnr(im1, im2)
		score_ssim = ssim(im1, im2, channel_axis=2, data_range=255,multichannel=True)

		avg_psnr += score_psnr
		avg_ssim += score_ssim

	avg_psnr = avg_psnr / n
	avg_ssim = avg_ssim / n
	print("===> Avg.PSNR: {:.4f} dB ".format(avg_psnr))
	print("===> Avg.SSIM: {:.4f} ".format(avg_ssim))

