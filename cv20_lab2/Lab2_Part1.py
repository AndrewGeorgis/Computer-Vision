#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sc
import LucasKanade as lk

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


head = [138, 88, 73, 123]
left_hand = [47,243,71,66]
right_hand = [162, 264, 83, 48]


# In[3]:


lk.DetectItem(head, rho=2.5, epsilon=0.07, threshold=5, reps=10, saveFolder='ResultsHead_Singlescale/') 


# In[4]:


lk.DetectItem(left_hand, rho=0.5, epsilon=0.01, threshold=30, reps=10, saveFolder='ResultsLeftHand_Singlescale/') 


# In[5]:


lk.DetectItem(right_hand, rho=2, epsilon=0.05, threshold=10, reps=20, saveFolder='ResultsRightHand_Singlescale/')  


# In[6]:


lk.DetectItem(head, rho=2.5, epsilon=0.08, threshold=1, reps=10, multiscale = True, N=3, saveFolder='ResultsHead_Multiscale/') 


# In[7]:


lk.DetectItem(left_hand, rho=0.7, epsilon=0.02, threshold=2, reps=10, multiscale = True, N=3, saveFolder='ResultsLeftHand_Multiscale/') 


# In[17]:


lk.DetectItem(right_hand, rho=2, epsilon=0.1, threshold=1, reps=10, multiscale = True, N=4, saveFolder='ResultsRightHand_Multiscale/')  


# In[ ]:




