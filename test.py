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
import time
import utils.segment
from utils.logger import setup_logger
from utils import util
from utils.iris import histMatchIris,makeIris,combineIris


def parse_args():
  """Parses arguments."""
  
  parser = argparse.ArgumentParser()

  parser.add_argument('--test_dir', type=str, default = './test_data',
                      help='directory of images to invert.')

 
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
     
    iris_green,iris_brown,iris_blue,iris_black,iris_mask = util.returnIrisTemplate()

    images = os.listdir(args.test_dir)

    for b , img in enumerate(images):


          print(img)
          image = Image.open(os.path.join(args.test_dir,img))


          eyeWholeL, eyeCenterL,predIrisL = histMatchIris(image,eye_left_right=5)

          if eyeCenterL is None:
               irisL = None
          else:
               coloursL = util.majorColors(eyeCenterL)
              #  print(coloursL[1].shape)
              #  print(predIrisL.shape)
               irisL = makeIris(coloursL[1],predIrisL, iris_brown , iris_blue, iris_green,iris_black)
               
          eyeWholeR, eyeCenterR,predIrisR = histMatchIris(image,eye_left_right=4)

          if eyeCenterR is None:
               irisR = None
          else:
               coloursR = util.majorColors(eyeCenterR)
               irisR = makeIris(coloursR[1],predIrisR, iris_brown , iris_blue, iris_green,iris_black)

          #print(irisL,irisR)

          iris = combineIris(irisL,irisR)
          #print(iris.shape)

          plt.imsave(os.path.join('results',img),iris)

    print(f"Completed generating iris of {len(images)} images")




               
     
if __name__ == '__main__':
  main()













    



  