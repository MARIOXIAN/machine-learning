# -*- coding = utf-8 -*-

from datenorm import *
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from scipy import sparse
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
import os
import time

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

    #删除，融合，时间戳转换
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

        # 1900/3/13 0:00:00是一个异常值，出现在A25(此列皆为整数），A26（此列为时间点）
        train.loc[train['A25'] == '1900/3/10 0:00', 'A25'] = train['A25'].value_counts().values[0]  # 以出现频率最高的数字代替
        train['A25'] = train['A25'].astype(int)

        data = pd.concat([train, test], axis=0, ignore_index=True)
        data = data.fillna(-1)

        for f in ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']:
            try:
                data[f] = data[f].apply(timeTranSecond)
            except:
                print(f, '应该在前面被删除了！')

        for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
            data[f] = data[f].apply(getDuration)

        return data,train,target

    #特征强化
    def data_boost(self):

        data,train,target  = self.drop_merge()

        data['sample_id'] = data['sample_id'].apply(lambda x: int(x.split('_')[1]))
        categorical_columns = [f for f in data.columns if f not in ['sample_id']]
        numerical_columns = [f for f in data.columns if f not in categorical_columns]

        # 强特
        data['b14/a1_a3_a4_a19_b1_b12'] = data['B14'] / (
                data['A1'] + data['A3'] + data['A4'] + data['A19'] + data['B1'] + data['B12'])

        feature_columns = ['b14/a1_a3_a4_a19_b1_b12']

        for item in feature_columns:
            numerical_columns.append(item)

        del data['A1']
        del data['A3']
        del data['A4']
        categorical_columns.remove('A1')
        categorical_columns.remove('A3')
        categorical_columns.remove('A4')

        # label encoder
        for f in categorical_columns:
            data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))
        train = data[:train.shape[0]]
        test = data[train.shape[0]:]
        print(train.shape)
        print(test.shape)

        ###################添加新特征，将收率进行分箱，然后构造每个特征中的类别对应不同收率的均值###########################

        train['target'] = target
        train['intTarget'] = pd.cut(train['target'], 5, labels=False)
        train = pd.get_dummies(train, columns=['intTarget'])
        li = ['intTarget_0.0', 'intTarget_1.0', 'intTarget_2.0', 'intTarget_3.0', 'intTarget_4.0']
        mean_columns = []
        for f1 in categorical_columns:
            cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
            if cate_rate < 0.90:
                for f2 in li:
                    col_name = 'B14_to_' + f1 + "_" + f2 + '_mean'
                    mean_columns.append(col_name)

                    # 前面每一个特征中，相同的取值下，分别对应后面的intTarget_1.0有一个值，求相同取值的intTarget_1.0的值的均值
                    order_label = train.groupby([f1])[f2].mean()
                    train[col_name] = train['B14'].map(order_label)

                    # 有缺失值就删除最新加入的一列特征列
                    miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
                    if miss_rate > 0:
                        # axis=1为删除列
                        train = train.drop([col_name], axis=1)
                        mean_columns.remove(col_name)
                    else:
                        test[col_name] = test['B14'].map(order_label)

        rate = []
        rate_columns = ['B14']
        for item in rate_columns:
            for f in li:
                count = train.groupby([item])[f].count().reset_index(name=item + f + '_all')
                count1 = train.groupby([item])[f].sum().reset_index(name=item + f + '_1')
                count[item + f + '_1'] = count1[item + f + '_1']
                count.fillna(value=0, inplace=True)
                count[item + f + '_rate'] = count[item + f + '_1'] / count[item + f + '_all']
                count[item + f + '_rate'] = count[item + f + '_rate'].astype(float)
                count.fillna(value=0, inplace=True)
                rate.append(item + f + '_rate')
                rate.append(item + f + '_1')
                rate.append(item + f + '_all')
                train = pd.merge(train, count, how='left', on=item)
                test = pd.merge(test, count, how='left', on=item)

        train.drop(li + ['target'], axis=1, inplace=True)

        # 删除卡方得分小于0.12的特征
        del_columns = ['B14_to_B14_intTarget_4.0_mean']
        for item in del_columns:
            del train[item]
            del test[item]
            mean_columns.remove(item)

        print(train.shape)
        print(test.shape)

        X_train = train[mean_columns + numerical_columns + rate].values
        X_test = test[mean_columns + numerical_columns + rate].values

        # one hot
        enc = OneHotEncoder()
        for f in categorical_columns:
            enc.fit(data[f].values.reshape(-1, 1))
            X_train = sparse.hstack((X_train, enc.transform(train[f].values.reshape(-1, 1))), 'csr')
            X_test = sparse.hstack((X_test, enc.transform(test[f].values.reshape(-1, 1))), 'csr')
        print(X_train.shape)
        print(X_test.shape)

        y_train = target.values

        return data,train,test,X_train,X_test,y_train,target

    #lgb预测
    def lgb_prediction(self):

        data, train, test,X_train,X_test,y_train,target = self.data_boost()
        # ----------------------------prediction---------------------------

        # ----------------------------lgb----------------------------------
        param = {'num_leaves': 120,
                 'min_data_in_leaf': 20,
                 'objective': 'regression',
                 'max_depth': -1,
                 'learning_rate': 0.01,
                 "min_child_samples": 20,
                 "boosting": "gbdt",
                 "feature_fraction": 0.9,
                 "bagging_freq": 1,
                 "bagging_fraction": 0.9,
                 "bagging_seed": 20,
                 "metric": 'mse',
                 "lambda_l1": 0.1,
                 "verbosity": -1}
        folds = KFold(n_splits=5, shuffle=True, random_state=2018)
        oof_lgb = np.zeros(len(train))
        predictions_lgb = np.zeros(len(test))

        for fold_, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
            print("fold n°{}".format(fold_ + 1))
            trn_data = lgb.Dataset(X_train[trn_idx], y_train[trn_idx])
            val_data = lgb.Dataset(X_train[val_idx], y_train[val_idx])

            num_round = 10000
            clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=200,
                            early_stopping_rounds=100)
            oof_lgb[val_idx] = clf.predict(X_train[val_idx], num_iteration=clf.best_iteration)

            predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / folds.n_splits

        print("CV score: {:<8.8f}".format(mean_squared_error(oof_lgb, target)))


d = Data()

#总体查看
train_overall,top_missing,top_percent = d.overall(name='train')


final = d.drop_merge()

d.lgb_prediction()