import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
import cv2
import matplotlib.pyplot as plt
import pickle
from pylab import *

import utils.segment
from utils.logger import setup_logger
from utils import util
from utils.iris import histMatchIris,makeIris,combineIris


def parse_args():
  """Parses arguments."""
  
  parser = argparse.ArgumentParser()

  parser.add_argument('--test_dir', type=str, default = './test_data',
                      help='directory of images to invert.')

  parser.add_argument('--need_align', dest='need_align', action='store_true',
                      help='need alignment and crop of input images.')

  parser.add_argument('--use_hog', dest='use_hog', action='store_true',
                      help='Use HOG + SVM in face detection instead of MMOD CNN.')

  parser.add_argument('-o', '--output_dir', type=str, default='./results',
                      help='Directory to save the results. If not specified, '
                           '`./results/'
                           'will be used by default.')

  parser.add_argument('--pretrained_dir', type=str, default = './pretrained_models',
                      help='Directory tof pretraied models. If not specified, '
                           '`./pretrained_models/'
                           'will be used by default.')
  
  parser.add_argument('--gpu_id', type=str, default='0',
                      help='Which GPU(s) to use. (default: `0`)')
  
  return parser.parse_args()



def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    align_save_path = os.path.join(output_dir,'aligned_images/')
    
    if not os.path.exists(align_save_path) and args.need_align:
         os.makedirs(align_save_path)

    images,aligned_image_names = util.load_images_from_dir(args.test_dir,need_align = args.need_align , use_hog = args.use_hog)
    if args.need_align:
      for i,img in enumerate(images):
         plt.imsave(os.path.join(align_save_path,aligned_image_names[i]),img)
      args.test_dir = align_save_path
      images,aligned_image_names = util.load_images_from_dir(args.test_dir)
     
     iris_green,iris_brown,iris_blue,iris_black,iris_mask = util.returnIrisTemplate()

     for b , image in enumerate(images):

          eyeWholeL, eyeCenterL,predIrisL = histMatchIris(image,eye_left_right=5)

          if eyeCenterL is None:
               irisL = None
          else:
               coloursL = util.majorColors(eyeCenterL)
               irisL = makeIris(coloursL[1],predIrisL, iris_brown , iris_blue, iris_green,iris_black)
               
          eyeWholeR, eyeCenterR,predIrisR = histMatchIris(image,eye_left_right=4)

          if eyeCenterR is None:
               irisR = None
          else:
               coloursL = util.majorColors(eyeCenterR)
               irisR = makeIris(coloursL[1],predIrisR, iris_brown , iris_blue, iris_green,iris_black)

          iris = combineIris(irisL,irisR)

          plt.imwrite(aligned_image_names[b],iris)

          print(f"Completed generating iris of {len(images)} images")




               
     
if __name__ == '__main__':
  main()













    



  