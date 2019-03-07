#!/usr/bin/env python
# coding: utf-8

# # Setting up

# In[1]:


# %pdb 1 
import torch
assert torch.__version__ == '0.4.0'
import sys
sys.path.append('/home/alex/faster-rcnn.pytorch/lib/visualization')
from visualizer import Visualizer
datasplit = 'vg_alldata_minival'


# In[2]:


vis = Visualizer(datasplit)


# # Features of image

# A feature map extracted from VGG. It has dimension: `depth x width x height (?)`. This feature map is shared by both Region Proposal Network (RPN) and the final stages of FasterRCNN. 

# ## What is 100? 

# Show a few more examples: 

# In[3]:


num_pred = 6
num_gt = 0
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)
vis.show_random_image(num_pred, num_gt)


# ## What is 300?

# In[ ]:





# Pooled feature of each proposal region. It has dimension: `proposal x depth x width x height`. We are going to use this as features to learn Graph RCNN. 

# In[4]:


# im_summary.pred.pooled_feat.shape 


# In[5]:


# im_summary.pred.cls_prob.shape


# In[6]:


# im_summary.pred.scores_nms.shape


# In[7]:


# im_summary.pred.rois.shape


# In[8]:


# len(im_summary.pred.bbox_nms)


# # Test Zone

# In[9]:


get_ipython().system('jupyter nbconvert --to script feature_explore.ipynb')


# In[ ]:




