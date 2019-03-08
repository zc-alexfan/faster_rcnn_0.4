import gzip
import pickle
from easydict import EasyDict as edict
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import random
import numpy as np

class visual_genome_driver(): 
    def __init__(self, _datasplit, _seed=2): 
        random.seed(_seed)
        self._datasplit = _datasplit
        self._feature_path = os.path.join('../data/features/%s/'%(_datasplit))
        
        meta_path = os.path.join(self._feature_path, 'meta.pkl')
        self._meta = pickle.load(open(meta_path, 'rb'))

        self._class_labels = self._meta.imdb_classes
        self._images_index = self._meta.imdb_image_index  # image name of JPEG files
        print("Constructed: visual_genome_driver(%s)"%(_datasplit))

    def get_meta(self):
        return self._meta
   

    """
    i.e. self._image_index[idx]
    """
    def get_image_by_idx(self, idx): 
        return self.get_image_by_id(self._images_index[idx])
    
    """
    i.e. self._image_index
    """
    def get_image_by_id(self, image_id): 
        curr_im_path = str(image_id) + ".pkl"
        with open(os.path.join(self._feature_path, curr_im_path), 'rb') as f:
            im = pickle.load(f)
        return im

    def show_image(self, _idx,  _k, _k_gt): 
        print("Image index is: %d" %self._images_index[_idx])
        curr_im_path = str(self._images_index[_idx]) + ".pkl"
        im_summary = pickle.load(open(os.path.join(self._feature_path, curr_im_path), 'rb'))
        curr_gt = im_summary.gt

        im = visual_genome_driver._normalize_image(im_summary) # image in RGB with [0, 1] range
        boxes = im_summary.pred.bbox_nms
        _scale = im_summary.info.dim_scale[0][2]

        _k = min(_k, len(boxes)) # num predictions to show
        _k_gt = min(_k_gt, len(curr_gt.boxes)) # num gt to show
        
        visual_genome_driver._visualize_regions_with_gt(self._class_labels, im, boxes[:_k], curr_gt, _k_gt, _scale)
  
    def show_random_image(self, _k, _k_gt): 
        idx = random.randint(0, len(self._images_index)-1)
        self.show_image(idx, _k, _k_gt)

    @staticmethod
    def _normalize_image(_im_summary): 
        """
        Swap axis of im_data from (channel, width, height) to (height, width, channel); 
        Convert im_data's color channels from GBR to RGB (fasterRCNN uses GBR).  
        Normalize each pixel to the range [0, 1]. 

        Input: im_summary.info.im_data from an image
        Output: regular image normalized to [0, 1]
        """

        _im_data = _im_summary.info.im_data 

        _im_data = _im_data.squeeze()
        _im_data = np.swapaxes(_im_data, 0, 1)
        _im_data = np.swapaxes(_im_data, 1, 2)

        _im_data = _im_data - _im_data.min()
        _im_data = _im_data/_im_data.max()

        _im = _im_data # need to adjust color channel 
        _im[:, :, 0] = _im_data[:, :, 1]
        _im[:, :, 1] = _im_data[:, :, 2]
        _im[:, :, 2] = _im_data[:, :, 0]

        return _im

    @staticmethod 
    def _visualize_regions_with_gt(_class_labels, _im, _boxes, _curr_gt, _k_gt, _scale, _xlim=15, _ylim=7):
        plt.rcParams['figure.figsize'] = [_xlim, _ylim]
        plt.imshow(_im)
        ax = plt.gca()

        _k_gt = min(_k_gt, len(_curr_gt.boxes))
        gt_boxes = _curr_gt.boxes[:_k_gt]
        gt_classes = _curr_gt.gt_classes[:_k_gt]

        # plot ground truth boxes
        for _region, class_id in zip(gt_boxes, gt_classes):
            x1, y1, x2, y2 = _region

            x = x1; y = y1; w = x2 - x1; h = y2 - y1
            x *= _scale; y *= _scale; w *= _scale; h *= _scale

            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=3))
            ax.text(x, y, _class_labels[class_id], style='italic', bbox={'facecolor':'white', 'alpha':1.0, 'pad':1})

        # plot predicted boxes
        for _region in _boxes:
            x1, y1, x2, y2, class_id, score = _region
            x = x1; y = y1; w = x2 - x1; h = y2 - y1
            x *= _scale; y *= _scale; w *= _scale; h *= _scale

            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='yellow', linewidth=3, alpha=score))
            ax.text(x, y, _class_labels[class_id] + " %.2f"%(score), style='italic', bbox={'facecolor':'white', 'alpha':1.0, 'pad':1})

        fig = plt.gcf()
        plt.tick_params(labelbottom=False, labelleft=False)
        plt.show()
 
