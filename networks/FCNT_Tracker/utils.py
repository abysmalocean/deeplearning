import numpy as np
import skimage
from scipy.misc import imresize


def gauss2d(shape=(6,6), sigma = 0.5):
    """
    2-D gaussian mask
    """
    m, n = [(ss-1.)*2. for ss in shape]
    y,x  = np.ogrid[-m:m+1, -n:n+1]
    h    = np.exp( - (x*x + y*y) / (2.*sigma*sigma))
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

# draw on img
def img_with_bbox(img_original, gt_1):
    img =np.copy(img_original)
	gt_1 = [int(i) for i in gt_1]
	w, h = gt_1[2:]
	tl_x, tl_y = gt_1[:2]
	tr_x, tr_y = tl_x + w, tl_y 
	dl_x, dl_y = tl_x, tl_y + h
	dr_x, dr_y = tl_x + w, tl_y +h

	rr1, cc1 = skimage.draw.line( tl_y,tl_x, tr_y, tr_x)
	rr2, cc2 = skimage.draw.line( tl_y,tl_x, dl_y, dl_x)
	rr3, cc3 = skimage.draw.line( dr_y,dr_x, tr_y, tr_x)
	rr4, cc4 = skimage.draw.line( dr_y,dr_x, dl_y, dl_x)
	img[rr1, cc1, :] = 1
	img[rr2, cc2, :] = 1
	img[rr3, cc3, :] = 1
	img[rr4, cc4, :] = 1
    return img