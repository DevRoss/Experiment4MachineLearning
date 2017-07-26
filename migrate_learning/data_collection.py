# coding: utf-8

# In[10]:


import os
import glob
from collections import OrderedDict


def get_data_as_dict(folder)
    '''
    获取数据
    :param folder: flower 存放的文件夹
    :return: 一个花名和图片地址的键值对
    '''
    path = os.path.join(os.path.abspath(os.path.curdir), folder)

    tree = os.walk(path)

    relpath, dirs, files = next(tree)

    extensions = ['JPG', 'JPEG', 'jpg', 'jpeg']
    flower_collection = OrderedDict()

    for dir in dirs:
        flower_collection[dir] = list()
        for extension in extensions:
            flower_path = os.path.join(relpath, dir, '*.' + extension)
            flower_collection[dir].extend(glob.glob(flower_path))

    return flower_collection