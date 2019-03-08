#!/usr/bin/env python
# coding: utf-8

# # Setting up

# In[1]:


# %load_ext autoreload
# %autoreaload 1  # 1 for the first call in %aimport
# %aimport visualizer

import torch
assert torch.__version__ == '0.4.0'
import sys
sys.path.append('/home/alex/faster-rcnn.pytorch/lib/visualization')
from visualizer import Visualizer
from easydict import EasyDict as edict
datasplit = 'vg_alldata_minival'


# In[2]:


import pickle, os


# In[3]:


vis = Visualizer(datasplit)
# curr_im_path = str(vis._images_index[0]) + ".pkl"
# im_summary = pickle.load(open(os.path.join(vis._feature_path, curr_im_path), 'rb'))
# boxes, probs, feats = vis.formalize_bbox(im_summary)


# In[4]:


vis.show_random_image(4, 3)


# In[5]:


im = vis.get_image_by_idx(0)


# In[9]:


meta = vis.get_meta()


# In[12]:


meta.imdb_classes[im.pred.bbox_nms[0][-2]]


# # Features of image

# Show a few more examples: 

# In[7]:


num_pred = 6
num_gt = 0
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
# vis.show_random_image(num_pred, num_gt)
# vis.show_random_image(num_pred, num_gt)
# vis.show_random_image(num_pred, num_gt)
# vis.show_random_image(num_pred, num_gt)
# vis.show_random_image(num_pred, num_gt)
# vis.show_random_image(num_pred, num_gt)
# vis.show_random_image(num_pred, num_gt)
# vis.show_random_image(num_pred, num_gt)
# vis.show_random_image(num_pred, num_gt)
# vis.show_random_image(num_pred, num_gt)
# vis.show_random_image(num_pred, num_gt)
# vis.show_random_image(num_pred, num_gt)


# ## What is 300?

# In[ ]:





# Pooled feature of each proposal region. It has dimension: `proposal x depth x width x height`. We are going to use this as features to learn Graph RCNN. 

# # Test Zone

# In[8]:


get_ipython().system('jupyter nbconvert --to script feature_explore.ipynb')


# In[ ]:




