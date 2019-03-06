#!/usr/bin/env python
# coding: utf-8

# ## Setting up

# In[1]:


# %pdb 1 
import torch
assert torch.__version__ == '0.4.0'
import pickle
from easydict import EasyDict as edict
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import random
import numpy as np

random.seed(2)
plt.rcParams['figure.figsize'] = [15, 7]

datasplit = 'vg_alldata_minival'
cache_path = os.path.join('../data/cache/%s_gt_roidb.pkl'%(datasplit))
feature_path = os.path.join('../data/features/%s/'%(datasplit))
print(feature_path)


# In[2]:


def normalize_image(_im_summary): 
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
    
def visualize_regions_with_gt(_im, _boxes, _class_labels, _curr_gt, _k_gt, _scale):
    plt.rcParams['figure.figsize'] = [15, 7]
    plt.imshow(_im)
    ax = plt.gca()
    
    _k_gt = min(_k_gt, len(_curr_gt.boxes))
    gt_boxes = _curr_gt.boxes[:_k_gt]
    gt_classes = _curr_gt.gt_classes[:_k_gt]

    for _region, class_id in zip(gt_boxes, gt_classes):
        x1, y1, x2, y2 = _region
        
        x = x1; y = y1; w = x2 - x1; h = y2 - y1
        x *= _scale; y *= _scale; w *= _scale; h *= _scale
        
        print("%s: (%1.f, %1.f, %1.f, %1.f)"%(_class_labels[class_id], x, y, w, h))
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=3))
        ax.text(x, y, _class_labels[class_id], style='italic', bbox={'facecolor':'white', 'alpha':1.0, 'pad':1})
    
    
    # plot prediction
    for _region in _boxes:
        x1, y1, x2, y2, class_id, score = _region
        x = x1; y = y1; w = x2 - x1; h = y2 - y1
        x *= _scale; y *= _scale; w *= _scale; h *= _scale
        
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor='yellow', linewidth=3, alpha=score))
        ax.text(x, y, _class_labels[class_id] + " %.2f"%(score), style='italic', bbox={'facecolor':'white', 'alpha':1.0, 'pad':1})
    
    fig = plt.gcf()
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.show()
    
def formalize_bbox(_im_summary): 
    """
    Extract bboxes from all classes and return a list of bbox. 
    Each element of the list is in the form: [x, y, w, h, class_id, score]. 
    The returned list is sorted descendingly according to score. 
    """
    boxes = [] # each element: x, y, w, h, class_id, score 
    for class_id, bboxes in enumerate(_im_summary.pred.bbox_nms):
        for bbox in bboxes:
            x, y, w, h, score = bbox
            boxes.append([x, y, w, h, class_id, score])

    boxes.sort(key=lambda x: -x[5])
    return boxes

def show_random_image(_feature_path, _images_index, _k, _gt, _k_gt): 
    idx = random.randint(0, len(_images_index)-1)

    # ground truth
    print("Image index is: %d" %_images_index[idx])
    curr_im_path = str(_images_index[idx]) + ".pkl"
    curr_gt = edict(_gt[idx])
    
    # image data
    im_summary = pickle.load(open(os.path.join(_feature_path, curr_im_path), 'rb'))
    im = normalize_image(im_summary) # image in RGB with [0, 1] range
    boxes = formalize_bbox(im_summary) # bboxes sorted by confidence
    _scale = im_summary.gt.im_info[0][2]
    
    _k = min(_k, len(boxes)) # num predictions to show
    _k_gt = min(_k_gt, len(curr_gt.boxes)) # num gt to show
    visualize_regions_with_gt(im, boxes[:_k], class_labels, curr_gt, _k_gt, _scale)


# ## Things you can use

# In[3]:


# curr_im_path = random.choice(images_dirs)


# In[4]:


# im_summary = pickle.load(open(os.path.join(feature_path, curr_im_path), 'rb'))
imdb_meta = pickle.load(open(os.path.join(feature_path, '%s.meta'%(datasplit)), 'rb'))
# print('Keys for im_summary: %s' %(str(im_summary.keys())))
print('Keys for imdb_meta: %s' %(str(imdb_meta.keys())))


# ### Ground Truth Info

# In[5]:


import gzip
gt =  pickle.load(gzip.open(os.path.join(cache_path), 'rb'),encoding='latin1')


# In[6]:


# import gzip
# gt =  pickle.load(gzip.open(os.path.join(cache_path), 'rb'),encoding='latin1')
# curr_gt = edict(gt[0])
# print(curr_gt.keys())
# print([class_labels[class_id] for class_id in curr_gt.gt_classes])


# In[7]:


# curr_gt.boxes


# ## Showing how to unpack meta info

# In[8]:


class_labels = imdb_meta.imdb_classes # idx_to_label
print(class_labels)


# In[9]:


image_index = imdb_meta.imdb_image_index  # image name of JPEG files
print(image_index[:10])


# ## Showing how to unpack image info

# In[10]:


# im = normalize_image(im_summary) # image in RGB with [0, 1] range
# print(im.shape) 
# boxes = formalize_bbox(im_summary) # bboxes sorted by confidence
# print(boxes[:2])
# print('num bboxes for this image is %d'%(len(boxes)))


# In[11]:


# x1, y1, x2, y2, class_id, score = boxes[0]
# print("x1, y1, x2, y2 = (%.1f, %.1f, %.1f, %.1f)" %(x1, y1, x2, y2))
# print(class_labels[class_id])
# print("score = %.2f" %(score))


# ## Features of image

# A feature map extracted from VGG. It has dimension: `depth x width x height (?)`. This feature map is shared by both Region Proposal Network (RPN) and the final stages of FasterRCNN. 

# In[12]:


# im_summary.pred.base_feat.shape 


# ### What is 100? 

# The top-k most confidence boxes: 

# In[13]:


# boxes[:2]


# Let's visualize the above boxes: 

# In[14]:


# # display_image(im)
# visualize_regions(im, boxes[:3], class_labels)


# Show a few more examples: 

# In[15]:


num_pred = 0
num_gt = 4
show_random_image(feature_path, image_index, num_pred, gt, num_gt)
show_random_image(feature_path, image_index, num_pred, gt, num_gt)
show_random_image(feature_path, image_index, num_pred, gt, num_gt)
show_random_image(feature_path, image_index, num_pred, gt, num_gt)
show_random_image(feature_path, image_index, num_pred, gt, num_gt)
show_random_image(feature_path, image_index, num_pred, gt, num_gt)
show_random_image(feature_path, image_index, num_pred, gt, num_gt)
show_random_image(feature_path, image_index, num_pred, gt, num_gt)
show_random_image(feature_path, image_index, num_pred, gt, num_gt)
show_random_image(feature_path, image_index, num_pred, gt, num_gt)


# ### What is 300?

# In[ ]:





# Pooled feature of each proposal region. It has dimension: `proposal x depth x width x height`. We are going to use this as features to learn Graph RCNN. 

# In[16]:


# im_summary.pred.pooled_feat.shape 


# In[17]:


# im_summary.pred.cls_prob.shape


# In[18]:


# im_summary.pred.scores_nms.shape


# In[19]:


# im_summary.pred.rois.shape


# In[20]:


# len(im_summary.pred.bbox_nms)


# In[ ]:





# # Test Zone

# In[21]:


get_ipython().system('jupyter nbconvert --to script feature_explore.ipynb')


# In[ ]:





# In[ ]:





# In[ ]:




