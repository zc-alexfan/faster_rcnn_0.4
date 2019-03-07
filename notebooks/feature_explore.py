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


import pickle, os


# In[3]:


vis = Visualizer(datasplit)
curr_im_path = str(vis._images_index[0]) + ".pkl"
im_summary = pickle.load(open(os.path.join(vis._feature_path, curr_im_path), 'rb'))


# In[ ]:





# In[4]:


boxes, probs = vis.formalize_bbox(im_summary)


# In[ ]:


boxes[:10]


# In[ ]:


probs[:10]


# # Features of image

# A feature map extracted from VGG. It has dimension: `depth x width x height (?)`. This feature map is shared by both Region Proposal Network (RPN) and the final stages of FasterRCNN. 

# ## What is 100? 

# Show a few more examples: 

# In[ ]:


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

# In[ ]:


# im_summary.pred.pooled_feat.shape 


# In[ ]:


# im_summary.pred.cls_prob.shape


# In[ ]:


# im_summary.pred.scores_nms.shape


# In[ ]:


# im_summary.pred.rois.shape


# In[ ]:


# len(im_summary.pred.bbox_nms)


# # Test Zone

# In[ ]:


get_ipython().system('jupyter nbconvert --to script feature_explore.ipynb')


# In[ ]:




