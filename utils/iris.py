import os
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
import math
import time
import matplotlib.pyplot as plt
from utils.util import calcHist , hsvDist , returnIrisTemplate
from utils.segment import segment_image_aspect_ratio



def returnEye(image , eye_ind=4):

  ## eye_index: 5 -> left eye, 4 -> right eye

  out,input = segment_image_aspect_ratio(image)
  mask = np.zeros((out.shape[0],out.shape[1]))
  mask = np.where(out == eye_ind, 255,0).astype(np.uint8)

  

  kernel = np.ones((5,5),np.uint8)
  mask_dil = cv2.dilate(mask,kernel,iterations = 1)
  masked = cv2.bitwise_and(np.array(input),np.array(input), mask = mask_dil)

  contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)



  contours_poly = [None]*len(contours)
  boundRect = [None]*len(contours)
  centers = [None]*len(contours)
  radius = [None]*len(contours)

  if len(contours) == 0:
    return None , None, input

    
  for i, c in enumerate(contours):
      contours_poly[i] = cv2.approxPolyDP(c, 3, True)
      boundRect[i] = cv2.boundingRect(contours_poly[i])
      centers[i], radius[i] = cv2.minEnclosingCircle(contours_poly[i])

  start = [int(boundRect[i][0]), int(boundRect[i][1])]
  end =  [int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3]) ]


  eye_crop = (np.array(input)[start[1] + int((end[1] -start[1])*0.1)  : end[1] + int((end[1] -start[1])*0.1) , start[0] : end[0]])

  eye_crop = cv2.blur(eye_crop , (3,3))

  eyeCenter = eye_crop[ int(eye_crop.shape[0]/2) - int(eye_crop.shape[0]*0.2):int(eye_crop.shape[0]/2) + int(eye_crop.shape[0]*0.2) , int(eye_crop.shape[1]/2) - int(eye_crop.shape[1]*0.1):int(eye_crop.shape[1]/2) + int(eye_crop.shape[1]*0.1) , :]
  #eyeCenter = eye[ : , int(eye.shape[1]/2) - int(eye.shape[1]*0.1):int(eye.shape[1]/2) + int(eye.shape[1]*0.1) , :]


  return eye_crop , eyeCenter, input


def histMatchIris(image,iris_template_folder = '/ExtractIris/utils/iris_templates',eye_left_right = 4):

  iris_green,iris_brown,iris_blue,iris_black,iris_mask = returnIrisTemplate()
  


  base = [iris_blue , iris_brown, iris_green, iris_black]

  eyeWhole,eye, inp = returnEye(image , eye_left_right)

  if eye is None:
    return None, None, None

  elif eye.shape[0]*eye.shape[1]<4:
    return None, None, None
  

  eye = np.array( eye*255, dtype = np.uint8) 
  ref = eye

  method = cv2.HISTCMP_BHATTACHARYYA
  dbl = cv2.compareHist(calcHist(ref), calcHist(iris_blue,mask = cv2.cvtColor(iris_mask , cv2.COLOR_RGB2GRAY)), method)
  dbr= cv2.compareHist(calcHist(ref), calcHist(iris_brown,mask = cv2.cvtColor(iris_mask , cv2.COLOR_RGB2GRAY)), method)
  dg = cv2.compareHist(calcHist(ref), calcHist(iris_green,mask = cv2.cvtColor(iris_mask , cv2.COLOR_RGB2GRAY)), method)
  dblck = cv2.compareHist(calcHist(ref), calcHist(iris_black,mask = cv2.cvtColor(iris_mask , cv2.COLOR_RGB2GRAY)), method)

    #print(np.argmax([dbl,dbr,dg]))
  src = base[np.argmax([dbl,dbr,dg,dblck])]
  


  matched = exposure.match_histograms(src, ref, multichannel=True)
  matched = matched*(iris_mask/255)
  matched = cv2.blur(matched , (3,3)).astype(np.uint8)


  # (fig, axs) =  plt.subplots(nrows=1, ncols=4, figsize=(18, 6))

  # axs[0].imshow(inp)
  # axs[0].axis('off')

  # axs[1].imshow(eyeWhole)
  # axs[1].axis('off')

  # axs[2].imshow(eye)
  # axs[2].axis('off')

  # axs[3].imshow(matched/255)
  # axs[3].axis('off')

  # plt.show()
  # time.sleep(1)


  #print(dbl , dbr , dg)

  return eyeWhole , ref , matched


def makeIris(colour,predIris , iris_brown , iris_blue, iris_green , iris_black):
  
  hsv_g = [95.62,
  44.44,
  28.23]

  hsv_bl = [205.06,
  40,
  53.72]

  hsv_br = [27.18,
  78.05,
  32.15]

  hsv_blck = [0.0,
  14.28,
  2.74]

  distances = [hsvDist(colour,hsv_br) , hsvDist(colour ,hsv_bl), hsvDist(colour ,hsv_g ) , hsvDist(colour ,hsv_blck )]
  majorColorIndex = np.argmin(distances)

  if majorColorIndex == 0:
    #print("brown")
    iris = cv2.addWeighted(iris_brown, 0.65, predIris, 0.35, 0)
  
  elif majorColorIndex == 1:
        #print("blue")
        iris = cv2.addWeighted(iris_blue, 0.2, predIris, 0.8, 0)

  elif majorColorIndex == 2:
        #print("green")
        iris = cv2.addWeighted(iris_green, 0.2, predIris, 0.8, 0)

  elif majorColorIndex == 3:
        #print("black")
        iris = cv2.addWeighted(iris_black, 0.65, predIris, 0.35, 0)


  return iris


def combineIris(irisL,irisR):
  if irisL is None:
    iris = irisR
  elif irisR is None:
    iris = irisL
  elif irisR is None and irisL is None:
    iris = None
  else:
    iris = cv2.addWeighted(irisR, 0.5, irisL, 0.5, 0)

  return iris





