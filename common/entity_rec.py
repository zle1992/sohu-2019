#!/usr/bin/python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2017 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""

通过多模式匹配命中标题中可以替换为词包的短语（使用python模块esmre）

Authors: zhangpanpan04
Date:    2017/09/30 16:05:19
"""

import sys
import esm
import os
#sys.path.append(os.path.abspath('.') + '/src/common')
#from singletone import Singleton
reload(sys)
sys.setdefaultencoding('utf8')

class Esmre():
    """
    多模式匹配类
    """
    _index = dict()
    def __init__(self):
        """__init__
        """
        if Esmre._index == dict():
            Esmre._index = esm.Index()
            self.__dict_file = "/sdb1/zhangle/2019/zhangle11/sohu-2019/data/nerDict.txt"
            for line in open(self.__dict_file):
                Esmre._index.enter(line.strip())
            # print
            Esmre._index.fix()

    @staticmethod
    def findall(title):
        """
        匹配title
        :param title:待匹配title
        :return: 命中列表 [((startpos，endpos),word)]
        """
        hitlist = list()
        for pos, hitWord in Esmre._index.query(title):
            #print hitWord
            hit_word = hitWord.decode("utf8")
            current_len = pos[1] - pos[0]
            hit_info = dict(pos=pos, hitword=hit_word)
            if len(hitlist) == 0:
                hitlist.append(hit_info)
                continue
            old_pos = hitlist[-1]['pos']
            old_len = old_pos[1] - old_pos[0]
            # 命中嵌套，前短后长，将前一个删除，如前一个为“巴塞”，后一个为“巴塞罗那”，
            # 则删除“巴塞”这个entity
            if old_len < current_len and old_pos[0] >= pos[0] and old_pos[1] <= pos[1]:
                hitlist.pop()
            # 命中嵌套，前长后短，当前结果丢弃，如前一个为“巴塞罗那”，后一个为“罗那”，
            # 则丢弃“罗那”这个entity
            if old_len >= current_len and old_pos[0] <= pos[0] and old_pos[1] >= pos[1]:
                continue
            # 命中交叉，只保留前者，如前一个为“巴塞”，后一个为“塞罗”，则丢弃“塞罗”这个entity
            if old_pos[0] < pos[0] < old_pos[1] < pos[1]:
                continue
            # 如果命中的前后两个实体之间有".", "", "·", "-"等符号，说明两个实体应该被识别为一个名字
            # if title[old_pos[1]: pos[0]] in [".", "", "·", "-"]:
            #     hitlist.pop()
            #     continue
            hitlist.append(hit_info)
            
        hitSet = [x['hitword'] for x in hitlist]
        #hitSet = set([x['hitword'] for x in hitlist])
        return hitSet



from read import read_json

esmre_obj = Esmre()
def esm_entity(title):
    return ' '.join(esmre_obj.findall(title))

def run(path,flog):
	df  = read_json(path,flag=flog)
	df['esm_title'] = df['title'].map(lambda x:esm_entity(x))
	df['esm_content'] = df['content'].map(lambda x:esm_entity(x))
	out_path = path+'.esm_entity.csv'
	df[[u'newsId',u'content', u'title','entity', 'emotion', 'esm_title', 'esm_content']].to_csv(out_path,sep='\t',index=False)
def main():
	run('/sdb1/zhangle/2019/zhangle11/sohu-2019/data/coreEntityEmotion_train.txt','train')
	run('/sdb1/zhangle/2019/zhangle11/sohu-2019/data/coreEntityEmotion_test_stage1.txt','test')

if __name__ == '__main__':
 	main()
__author__ = 'liukun12'
