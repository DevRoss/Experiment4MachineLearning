# coding:utf-8
import jieba
import os
import re


def cut(path: str):
    ret = []
    with open(path, encoding='utf-8') as  fp:
        for line in fp:
            tmp = re.sub(r'[【】，。、·？！\[\]\(\)（）\\‘’\-/“”\da-zA-Z~…| +#\.《》]', '', line.strip())
            seg_list = jieba.cut(tmp, cut_all=False)
            ret.append(' '.join(seg_list))
    with open(path.replace('.txt', '_cut.txt'), 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(ret))


if __name__ == '__main__':
    cut('data/0.txt')
    cut('data/1.txt')
