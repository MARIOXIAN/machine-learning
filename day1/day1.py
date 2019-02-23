# -*- coding = utf-8 -*-

from datenorm import *
import numpy as np
import pandas as pd
import os

BASE_PATH = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_PATH,'data')

class Data(object):

    @staticmethod
    def _Getdata_(name):

        if name =='train':
            data_name = os.path.join(DATA_PATH,'jinnan_round1_train_20181227.csv')

        if name == 'test':
            data_name = os.path.join(DATA_PATH,'jinnan_round1_testA_20181227.csv')

        dataset = pd.read_csv(data_name,encoding = 'gbk')

        return dataset

    @staticmethod
    def _Pre_(train,test):

        tr_id = train[u'样本id']
        tr_result = train[u'收率']
        te_id = test[u'样本id']

        del train[u'样本id']
        del train[u'收率']
        del test[u'样本id']

        train['sample_id'] = tr_id
        train['result'] = tr_result
        test['sample_id'] = te_id

        return train,test

    def overall(self,name):

        if name =='train':
            dataset = self._Getdata_(name='train')

        if name == 'test':
            dataset = self._Getdata_(name='test')

        stats = []
        for col in dataset.columns:
            stats.append((col, dataset[col].nunique(), dataset[col].isnull().sum() * 100 / dataset.shape[0],
                          dataset[col].value_counts(normalize=True, dropna=False).values[0] * 100, dataset[col].dtype))

        stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values',
                                                'Percentage of values in the biggest category', 'type'])

        #对缺失值比例进行排序
        dataset_overall = stats_df.sort_values('Percentage of missing values', ascending=False)

        #取缺失值比例较大的10个特征
        top_missing  = dataset_overall[['Feature','Percentage of missing values']].head(10)

        # 取单个取值占比较大的10个特征
        top_percent = stats_df.sort_values('Percentage of values in the biggest category',
                                           ascending=False)[['Feature','Percentage of values in the biggest category']].head(20)

        return dataset_overall,top_missing,top_percent

    def drop_merge(self):

        train = self._Getdata_(name='train')
        test = self._Getdata_(name='test')

        train,test = self._Pre_(train,test)

        #删除某类别占比大于90的列,暂时保留A1,A3,A4？？

        for df in [train,test]:
            df.drop(['B3', 'B13', 'A13', 'A18', 'A23','A2','B2'], axis=1, inplace=True)

        train = train[train['result'] > 0.87] #删除收率小于0.87的异常值

        good_cols = list(train.columns)

        train = train[good_cols]
        good_cols.remove('result')
        test = test[good_cols]

        target = train['result']
        del train['result']
        data = pd.concat([train, test], axis=0, ignore_index=True)
        data = data.fillna(-1)

        return data

d = Data()

train_overall,top_missing,top_percent = d.overall(name='train')

final = d.drop_merge()

print('1')
print(final)
print('3')