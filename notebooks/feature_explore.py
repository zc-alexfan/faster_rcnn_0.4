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
sys.path.append('/home/alex/faster-rcnn.pytorch/lib/dataset_drivers')
from visual_genome_driver import visual_genome_driver
from easydict import EasyDict as edict


# In[2]:


datasplit = 'vg_alldata_minival'
driver = visual_genome_driver(datasplit)


# # Driver Functions

# In[3]:


driver.show_random_image(4, 3)


# In[4]:


num_gt = 5
num_pred = 2
driver.show_image(0, num_pred, num_gt)


# In[5]:


im_summary = driver.get_image_by_idx(0)
image_id = im_summary.info.image_idx
im_summary = driver.get_image_by_id(image_id)


# In[6]:


meta = driver.get_meta()


# # Meta object

# In[7]:


meta.keys()


# In[8]:


meta.imdb_name


# In[9]:


labels = meta.imdb_classes
print(labels[:10])


# In[10]:


image_index = meta.imdb_image_index
print(image_index[:10])


# # image object

# In[11]:


im_summary.keys()


# ## im_summary.info

# In[13]:


im_summary.info.keys()


# In[20]:


im_summary.info.data.shape # pixels


# In[30]:


print(im_summary.info.dim_scale)
print(im_summary.gt.height, im_summary.gt.width)
print(im_summary.info.dim_scale[0]/im_summary.gt.height)


# ## im_summary.gt

# In[14]:


im_summary.gt.keys()


# In[15]:


im_summary.gt.boxes[:3] # (x1, y1, x2, y2)


# In[16]:


im_summary.gt.gt_classes[:3] # (x1, y1, x2, y2)


# ## im_summary.pred

# In[31]:


im_summary.pred.keys()


# In[32]:


im_summary.pred.base_feat.shape


# In[35]:


len(im_summary.pred.pooled_feat)


# In[36]:


im_summary.pred.pooled_feat[0].shape


# In[37]:


len(im_summary.pred.cls_prob)


# In[39]:


im_summary.pred.cls_prob[0].shape


# In[40]:


len(im_summary.pred.boxes)


# In[41]:


im_summary.pred.boxes[0] # x1, y1, x2, y2, class_id, confidence


# In[43]:


meta.imdb_classes[im_summary.pred.boxes[0][-2]]


# # Test Zone

# In[ ]:


get_ipython().system('jupyter nbconvert --to script feature_explore.ipynb')


# In[ ]:




