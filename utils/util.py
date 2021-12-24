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
from .face_landmark_detector import FaceLandmarkDetector




# def align_face(image_path, align_size=256):
#   """Aligns a given face."""
#   model = FaceLandmarkDetector(align_size)
#   face_infos = model.detect(image_path)
#   imgs = []
#   if len(face_infos) != 0:
#     face_infos = face_infos
#     for info in face_infos:
#       imgs.append(model.align(info))

#   else:
#     return None
#   return imgs


def align_face(image_path, align_size=256 , use_hog = False):
  """Aligns a given face."""
  model = FaceLandmarkDetector(align_size , use_hog= use_hog)
  face_infos = model.detect(image_path)

  aligned_images = []
  aligned_images_names = []


  if len(face_infos) != 0:

    if len(face_infos) == 1:
      info = face_infos[0]
      aligned_image = model.align(info)
      aligned_image_name = info['image_path'].split('/')[-1].split('.')[0]+".jpg"
      return aligned_image , aligned_image_name

    elif len(face_infos) > 1:
      for b,info in enumerate(face_infos):
        aligned_image = model.align(info)
        aligned_image_name = info['image_path'].split('/')[-1].split('.')[0]+f'_{b}.jpg'
        
        aligned_images.append(aligned_image)
        aligned_images_names.append(aligned_image_name)

      print(len(aligned_images) , len(aligned_images_names))
      return aligned_images,aligned_images_names
  else:
    return None, None


def align(image_path):
  """Aligns an unloaded image."""
  aligned_images , aligned_images_names  = align_face(image_path,
                             align_size=256)
  return aligned_images , aligned_images_names





def load_image(path):
  """Loads an image from disk.

  NOTE: This function will always return an image with `RGB` channel order for
  color image and pixel range [0, 255].

  Args:
    path: Path to load the image from.

  Returns:
    An image with dtype `np.ndarray` or `None` if input `path` does not exist.
  """
  if not os.path.isfile(path):
    return None

  image = Image.open(path)
  return image

def flatten(t):
    return [item for sublist in t for item in sublist]

    
def load_images_from_dir(dspth,align_size = 256, need_align = False , use_hog = False):

  images = []
  image_names =  natsorted(os.listdir(dspth))
  aligned_images_names = []


  for image_name in natsorted(os.listdir(dspth)):
    if image_name.split('.')[-1].lower() is 'jpg' or 'png' or 'jpeg' :
      if need_align:
        aligned_image , aligned_name  = align_face((os.path.join(dspth,image_name)),align_size=align_size , use_hog = use_hog )

      else:
        aligned_image = plt.imread(os.path.join(dspth,image_name))
        aligned_image = cv2.resize(aligned_image , (align_size,align_size))
        aligned_name = image_name


      images.append(aligned_image)
      if type(aligned_image) == list :
        images = flatten(images)

      aligned_images_names.append(aligned_name)
      if type(aligned_image) == list:
        aligned_images_names = flatten(aligned_images_names)

      

  return images,aligned_images_names


def imshow(images, col, viz_size=256):
  """Shows images in one figure."""
  num, height, width, channels = images.shape
  assert num % col == 0
  row = num // col

  fused_image = np.zeros((viz_size * row, viz_size * col, channels), dtype=np.uint8)

  for idx, image in enumerate(images):
    i, j = divmod(idx, col)
    y = i * viz_size
    x = j * viz_size
    if height != viz_size or width != viz_size:
      image = cv2.resize(image, (viz_size, viz_size))
    fused_image[y:y + viz_size, x:x + viz_size] = image

  fused_image = np.asarray(fused_image, dtype=np.uint8)
  data = io.BytesIO()
  if channels == 4:
    Image.fromarray(fused_image).save(data, 'png')
  elif channels == 3:
    Image.fromarray(fused_image).save(data, 'jpeg')
  else:
    raise ValueError('Image channel error')
  im_data = data.getvalue()
  disp = IPython.display.display(IPython.display.Image(im_data))
  return disp


def get_landmarks(images):
  landmarks = []
  fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
  for img in images:
    pred = fa.get_landmarks(img)
    pred = np.array(pred)
    pred.resize((68,2))
    landmarks.append(pred)
    
  return landmarks


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
  fig = plt.figure()
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
      plt.bar(i, count, color=((r,g,b)))

  return colours

def hsvDist(hsv1 , hsv2):
  dh = min(abs(hsv1[0]-hsv2[0]), 360-abs(hsv1[0]-hsv2[0])) / 180.0
  ds = abs(hsv1[1]-hsv2[1])/100
  dv = abs(hsv1[2]-hsv2[2]) / 100
  distance = math.sqrt(dh*dh+ds*ds+dv*dv)

  return distance

def returnIrisTemplate(iris_template_folder ='./iris_template'):
  iris_green = plt.imread(os.path.join(iris_template_folder,"iris_green.jpg"))
  iris_brown = plt.imread(os.path.join(iris_template_folder,"iris_brown.jpg"))
  iris_blue =  plt.imread(os.path.join(iris_template_folder,"iris_blue.jpg"))
  iris_black = plt.imread(os.path.join(iris_template_folder,"iris_black.jpg"))
  iris_mask =  plt.imread(os.path.join(iris_template_folder, "iris_mask.png"))
  return iris_green,iris_brown,iris_blue,iris_black,iris_mask

