# coding: utf-8

# In[10]:


import os
import glob
from collections import OrderedDict

path = os.path.join(os.path.abspath(os.path.curdir), 'flower_photos')

tree = os.walk(path)

relpath, dirs, files = next(tree)

extensions = ['JPG', 'JPEG', 'jpg', 'jpeg']
flower_collection = OrderedDict()

for dir in dirs:
    flower_collection[dir] = list()
    for extension in extensions:
        flower_path = os.path.join(relpath, dir, '*.' + extension)
        flower_collection[dir].extend(glob.glob(flower_path))
