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
datasplit = 'vg_alldata_minival'


# In[2]:


import pickle, os


# In[3]:


vis = Visualizer(datasplit)
curr_im_path = str(vis._images_index[0]) + ".pkl"
im_summary = pickle.load(open(os.path.join(vis._feature_path, curr_im_path), 'rb'))
boxes, probs, feats = vis.formalize_bbox(im_summary)


# In[15]:


im_summary = pickle.load(open(os.path.join(vis._feature_path, curr_im_path), 'rb'))
boxes, probs, feats = vis.formalize_bbox(im_summary)
im_summary.pred.pooled_feat = feats
im_summary.pred.bbox_nms = boxes
im_summary.pred.cls_prob = probs


# In[23]:


probs


# In[16]:


im_summary.pred.keys()


# In[5]:


boxes[:10]


# In[6]:


len(boxes[0])


# In[7]:


probs[0].shape


# In[8]:


feats[0].shape


# # Features of image

# Show a few more examples: 

# In[9]:


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

# In[10]:


get_ipython().system('jupyter nbconvert --to script feature_explore.ipynb')


# In[ ]:




