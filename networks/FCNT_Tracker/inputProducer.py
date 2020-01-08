import os
import skimage
import numpy as np
from scipy.misc import imread, imresize
from utils import gauss2d

class InputProducer: 
    def __init__(self, imgs_path, gt_path, live=False):
        """
        """
        self.imgs_path_list = [os.path.join(imgs_path, fn) for fn in sorted(os.listdir(imgs_path))]
        self.gts_list = self.gen_gts(gt_path)
        self.gen_img  = self.get_image()

    def get_gts(self, gt_path):
        """
        Each row in the ground-truth files represents the bounding box of the target in that frame (tl_x, tl_y, box_width, box_height)
        """

        f = open(gt_path, 'r')
        lines = f.readlines()
        
        try:
			gts_list = [[int(p) for p in i[:-1].split(',')] 
			                   for i in lines
        except Exception as e:
			gts_list = [[int(p) for p in i[:-1].split('\t')] 
			                   for i in lines]
		return gts_list
    
    def get_image(self):
        idx = -1
        for image_path, gt in zip(self.imgs_path_list, self.gts_list):
            img = imread(image_path, mode = 'RGB')
            assert min(img.shape[:2]) >= 224
            assert len(img.shape) == 3

            idx += 1
            if idx == 0:
                self.first_gt = gt
                self.first_img = img
            yield img, gt, idx
    
    def gen_mask(self, fea_sz):
        """
        Generates 2D Gaussian masked canvas with shape same as fea_sz. This method should only called on the first frame. 
        
        Args: 
            img_sz : input image size. 
            fea_sz : feature size to be identical to the output of sel-CNN net

        Return: 
            convas: fea_sz shape with 1 channel, the central region is an 2D gaussian. 
        """
        im_sz = self.first_img.shape
        x, y, w, h = self.first_gt
        convas = np.zeros(im_sz[:2])

        # Generates 2D gaussian mask
        scale = min([w,h])/3 # 
        mask  = gauss2d([h,w], sigma = scale)
        print(mask.max(), 'max of mask')
        

        