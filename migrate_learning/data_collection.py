# coding: utf-8

# In[10]:


import os
import glob
import functools
import numpy as np


def percentage(testing_percentage, validation_percentage):
    def pre_decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            c = f(*args, **kwargs)
            result = dict()
            for flower_label, pics in c.items():
                training_images = list()
                testing_images = list()
                validation_images = list()
                for pic in pics:
                    chance = np.random.randint(100)
                    if chance < validation_percentage:
                        validation_images.append(pic)

                    elif chance < validation_percentage + testing_percentage:
                        testing_images.append(pic)

                    else:
                        training_images.append(pic)
                result[flower_label] = {
                    'training': training_images,
                    'testing': testing_images,
                    'validation': validation_images
                }
            return result

        return wrapper

    return pre_decorator


def get_flower(testing_percentage, validation_percentage):
    '''

    :param testing_percentage:
    :param validation_percentage:
    :return:
        result = {
                  flower_label:
                    {
                        'training': [],
                        'testing': [],
                        'validation': []
                    }
                }
    '''

    @percentage(testing_percentage, validation_percentage)
    def get_data_as_dict(folder):
        '''
        获取数据
        :param folder: flower 存放的文件夹
        :return: 一个花名和图片地址的键值对
        '''
        path = os.path.join(os.path.abspath(os.path.curdir), folder)

        tree = os.walk(path)

        relpath, dirs, files = next(tree)

        extensions = ['JPG', 'JPEG', 'jpg', 'jpeg']
        flower_collection = dict()

        for dir in dirs:
            flower_collection[dir] = list()
            for extension in extensions:
                flower_path = os.path.join(relpath, dir, '*.' + extension)
                flower_collection[dir].extend(glob.glob(flower_path))

        return flower_collection

    return get_data_as_dict


def get_image(result, label_name, index, category):
    '''

    :param result: 结果集合
    :param label_name: 标签名
    :param index: 想要图片的位置
    :param category: 分为训练集，测试集，验证集
    :return: 一个图片的绝对路径
    '''
    flowers = result[label_name][category]
    mod_index = index % len(flowers)
    return flowers[mod_index]

# a = get_flower(20, 30)('flower_photos')
# print(a['roses'])
