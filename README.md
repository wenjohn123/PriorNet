# PriorNet: A Lightweight and Generalizable Dehazing Network

### Environment
python 3.8, 
pytorch 1.7.1+cuda110, 
torchvision 0.8.2+cuda110, 
scikit-image 0.21.0, 
opencv-python 4.9.0.80, 
numpy 1.24.4
### Dataset
trainset: HAZE4k
testset: part of NYU_Depth_V2 (NYU2_1400.jpg-NYU2_1449.jpg)
### Document Illustration
The doucument structures are as follows:
- PriorNet
  - data
    - data
    - image
  - original
  - results
  - samples
  - snapshots
	  - dehazer.pth 
  - test_images
  - dataloader.py
  - dehaze.py
  - net.py
  - ssim_psnr.py
  - train.py

The .pth file is in the snapshots

The net structure is in the net.py

Train code and test code is in the train.py and test.py separately. 
