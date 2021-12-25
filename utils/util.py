import os
import sys
sys.path.append('../')

import io
import numpy as np
from PIL import Image
import cv2
import IPython.display
from natsort import natsorted
import matplotlib.pyplot as plt
import face_alignment
import colorsys
from sklearn.cluster import KMeans
import math




def calcHist(image , mask = None):
  hist = cv2.calcHist([image], [0, 1, 2], mask, [8, 8, 8],[0, 256, 0, 256, 0, 256])
  hist = cv2.normalize(hist, hist).flatten()
  return hist


def majorColors(image , number_color = 2):

  # read image into range 0 to 1
  img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) / 255

  # set number of colors
  number = number_color

  # quantize to 16 colors using kmeans
  h, w, c = img.shape
  img2 = img.reshape(h*w, c)
  kmeans_cluster = KMeans(n_clusters=number)
  kmeans_cluster.fit(img2)
  cluster_centers = kmeans_cluster.cluster_centers_
  cluster_labels = kmeans_cluster.labels_

  # need to scale back to range 0-255 and reshape
  img3 = cluster_centers[cluster_labels].reshape(h, w, c)*255.0
  img3 = img3.astype('uint8')



  # reshape img to 1 column of 3 colors
  # -1 means figure out how big it needs to be for that dimension
  img4 = img3.reshape(-1,3)

  # get the unique colors
  colors, counts = np.unique(img4, return_counts=True, axis=0)

  # compute HSV Value equals max(r,g,b)
  values = []
  for color in colors:
      b=color[0]
      g=color[1]
      r=color[2]
      v=max(b,g,r)
      values.append(v)

  # zip colors, counts, values together
  unique = zip(colors,counts,values)

  # make list of color, count, value
  ccv_list = []
  for color, count, value in unique:
      ccv_list.append((color, count, value))
      
  # function to define key as third element
  def takeThird(elem):
      return elem[1]

  # sort ccv_list by Value (brightness)
  ccv_list.sort(key=takeThird)

  # plot each color sorted by increasing Value (brightness)
  # pyplot uses normalized r,g,b in range 0 to 1
  #fig = plt.figure()
  gray = None
  length = len(ccv_list)
  colours = []

  for i in range(length):
      item = ccv_list[i]
      color = item[0]
      b = color[0]/255
      g = color[1]/255
      r = color[2]/255
      h, s, v = colorsys.rgb_to_hsv(r, g, b)
      colours.append([h*360,s*100,v*100])
      # print(f"color {i}")
      # print(h*360)
      # print(s*100)
      # print(v*100)
      count = item[1]
      #plt.bar(i, count, color=((r,g,b)))

  return colours

def hsvDist(hsv1 , hsv2):
  dh = min(abs(hsv1[0]-hsv2[0]), 360-abs(hsv1[0]-hsv2[0])) / 180.0
  ds = abs(hsv1[1]-hsv2[1])/100
  dv = abs(hsv1[2]-hsv2[2]) / 100
  distance = math.sqrt(dh*dh+ds*ds+dv*dv)

  return distance

def returnIrisTemplate(iris_template_folder ='./utils/iris_templates'):
  iris_green = plt.imread(os.path.join(iris_template_folder,"iris_green.jpg"))
  iris_brown = plt.imread(os.path.join(iris_template_folder,"iris_brown.jpg"))
  iris_blue =  plt.imread(os.path.join(iris_template_folder,"iris_blue.jpg"))
  iris_black = plt.imread(os.path.join(iris_template_folder,"iris_black.jpg"))
  iris_mask =  plt.imread(os.path.join(iris_template_folder, "iris_mask.png"))

  iris_blue = cv2.resize(iris_blue,(400,400))
  iris_brown = cv2.resize(iris_brown,(400,400))
  iris_black = cv2.resize(iris_black,(400,400))
  iris_green = cv2.resize(iris_green,(400,400))
  iris_mask = cv2.resize(iris_mask,(400,400))
  iris_mask = (cv2.cvtColor(iris_mask , cv2.COLOR_RGBA2RGB)*255).astype(np.uint8)


  return iris_green,iris_brown,iris_blue,iris_black,iris_mask

