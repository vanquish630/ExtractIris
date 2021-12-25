import sys
sys.path.append('../')

import numpy as np
import torch
from torchvision import transforms
import cv2
from skimage import exposure
from PIL import Image
import colorsys
from sklearn.cluster import KMeans
from models.model import BiSeNet

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def segment_image(image,modelpath = './pretrained_models/79999_iter.pth', size = 256):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(torch.load(modelpath, map_location=DEVICE) )
    net.to(DEVICE)
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
    image = cv2.resize(image, (512,512),interpolation = cv2.INTER_NEAREST)

    #size = 256,256
    with torch.no_grad():
      img = to_tensor(image)
      img = torch.unsqueeze(img, 0)
      img = img.to(DEVICE)
      out = net(img)[0]
      img = inv_normalize(img)
      output = (np.transpose(np.array(out.squeeze(0).cpu()),(1,2,0)).argmax(2).astype(np.uint8))
    return output


def segment_image_aspect_ratio(image,modelpath = './pretrained_models/79999_iter.pth', size = 256):
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(torch.load(modelpath, map_location=DEVICE) )
    net.to(DEVICE)
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)
    #image = cv2.resize(image, (512,512),interpolation = cv2.INTER_NEAREST)

    #size = 256,256 0-255
    with torch.no_grad():
      #print(image.shape , image.max())
      img =  (image)
      basewidth = max(512,np.array(img).shape[0])
      wpercent = (basewidth/float(img.size[0]))
      hsize = int((float(img.size[1])*float(wpercent)))
      img = img.resize((basewidth,hsize), Image.ANTIALIAS,)
      image = img
      img = to_tensor(image)
      img = torch.unsqueeze(img, 0)
      img = img.to(DEVICE)
      out = net(img)[0]
      img = inv_normalize(img)
      output = (np.transpose(np.array(out.squeeze(0).cpu()),(1,2,0)).argmax(2).astype(np.uint8))
    return output, np.transpose(np.array(img.squeeze(0).cpu()),(1,2,0))

def return_hair_mask(image):
  out = segment_image(image)
  hairmask =  np.where(out==17,1,0)  #index corresponding to hair segmentation
  return hairmask

def maskIris(out):
  mask = []
  for parsing in out:
    base = np.zeros((parsing.shape[0],parsing.shape[1]))
    base = np.where(parsing == 4, 1,0)

    mask.append(base)

  return mask


def return_face_mask(image):
  out = segment_image(image)
  hairmask =  np.where(out==1,1,0) 
  im_th = np.array(hairmask*255 , dtype=np.uint8)
  im_floodfill = im_th.copy()

  # Mask used to flood filling.
  # Notice the size needs to be 2 pixels than the image.
  h, w = im_th.shape[:2]
  mask = np.zeros((h+2, w+2), np.uint8)

  # Floodfill from point (0, 0)
  cv2.floodFill(im_floodfill, mask, (0,0), 255);

  # Invert floodfilled image
  im_floodfill_inv = cv2.bitwise_not(im_floodfill)

  # Combine the two images to get the foreground.
  im_out = im_th | im_floodfill_inv

  return im_out


