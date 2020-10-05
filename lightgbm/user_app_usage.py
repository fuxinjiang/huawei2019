# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
# 主要利用多线程进行加速处理数据
import pandas as pd
# 首先将原来的usage里面没有用的app过滤掉
import time
import multiprocessing
import os

import numpy as np
import json
from joblib import Parallel, delayed
import multiprocessing
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
# 训练集、测试集的路径
age_train_file = "../age_train.csv"
age_test_file = "../age_test.csv"
from scipy.sparse import csr_matrix
# 该函数主要用于过滤没有用的app的
def filter_app_usage(file_path, effctive_apps_files):
    time_start_ = time.time()
    user_app_usage = pd.read_csv(file_path, header=None, names=['id', 'app', 'duration', 'times', 'date'], usecols=['id', 'app', 'duration', 'times', 'date'])
    effective_apps = pd.read_csv(effctive_apps_files, header=None)
    effective_apps.columns = ['app']
    effective_apps_list = list(effective_apps['app'])
    user_app_usage_filter = user_app_usage.loc[user_app_usage.app.isin(effective_apps_list)]
    time_final_ = time.time()
    print("过滤有效的APP完毕!")
    print("花费时间为:")
    print(time_final_ - time_start_)
    return user_app_usage_filter
# 对过滤之后的APP进行计算
def duration_sum_times(data, effctive_apps_files):
    print("开始计算duration和times的总和计算!")
    time_start_duration_sum_times = time.time()
    data_train = pd.read_csv(age_train_file, header=None, names=['id', 'label'], usecols=['id'], dtype={'id': np.str}, index_col='id')
    data_test = pd.read_csv(age_test_file, header=None, names=['id'], dtype={'id': np.str}, index_col='id')
    effective_apps = pd.read_csv(effctive_apps_files, header=None)
    effective_apps.columns = ['app']
    effective_apps_list = list(effective_apps['app'])
    dictionary = dict(zip(effective_apps_list, list(range(len(effective_apps_list)))))
    data_train['trainrow'] = np.arange(data_train.shape[0])
    data_test['testrow'] = np.arange(data_test.shape[0])
    data_train.index = data_train.index.astype(np.str)
    data_test.index = data_test.index.astype(np.str)

    datalist = data['date'].unique().tolist()
    # 总共的APP数量
    app_numbers = 5000
    def temp(data, date):
        temp_1 = data.loc[(data['date'] == date)]
        temp_1.reset_index(drop=True, inplace=True)
        temp_1['appId_category'] = temp_1['app'].map(dictionary)
        temp_1 = temp_1.dropna().reset_index()
        temp_1 = temp_1.set_index('id')
        temp_1.index = temp_1.index.astype(np.str)
        temp_1_train = temp_1.merge(data_train, on='id')
        temp_1_test = temp_1.merge(data_test, on='id')
        temp_1_train_duration = csr_matrix((temp_1_train['duration'].values, (temp_1_train.trainrow, temp_1_train.appId_category)), shape=(data_train.shape[0], app_numbers))
        temp_1_test_duration = csr_matrix((temp_1_test['duration'].values, (temp_1_test.testrow, temp_1_test.appId_category)), shape=(data_test.shape[0], app_numbers))
        temp_1_train_times = csr_matrix((temp_1_train['times'].values, (temp_1_train.trainrow, temp_1_train.appId_category)), shape=(data_train.shape[0], app_numbers))
        temp_1_test_times = csr_matrix((temp_1_test['times'].values, (temp_1_test.testrow, temp_1_test.appId_category)), shape=(data_test.shape[0], app_numbers))
        user_app_usage_train_duration[date] = temp_1_train_duration
        user_app_usage_test_duration[date] = temp_1_test_duration
        user_app_usage_train_times[date] = temp_1_train_times
        user_app_usage_test_times[date] = temp_1_test_times

    with multiprocessing.Manager() as manager:
        user_app_usage_train_duration = manager.dict()
        user_app_usage_test_duration = manager.dict()
        user_app_usage_train_times = manager.dict()
        user_app_usage_test_times = manager.dict()
        multipro = []

        for i, date in enumerate(datalist):
            print(date)
            thread_name = "usage_thead_%d" % i
            multipro.append(multiprocessing.Process(target=temp, name=thread_name, args=(data, date,)))
        for process in multipro:
            process.start()
        for process in multipro:
            process.join()

        for i, date in enumerate(datalist):
            if i == 0:
                train_duration = user_app_usage_train_duration[date]
                test_duration = user_app_usage_test_duration[date]

                train_times = user_app_usage_train_times[date]
                test_times = user_app_usage_test_times[date]
            else:
                train_duration += user_app_usage_train_duration[date]
                test_duration += user_app_usage_test_duration[date]

                train_times += user_app_usage_train_times[date]
                test_times += user_app_usage_test_times[date]

    sparse.save_npz('train_sparse_matrix_duration_sum.npz', train_duration)  # 保存
    sparse.save_npz('test_sparse_matrix_duration_sum.npz', test_duration)  # 保存
    sparse.save_npz('train_sparse_matrix_times_sum.npz', train_times)  # 保存
    sparse.save_npz('test_sparse_matrix_times_sum.npz', test_times)  # 保存

    time_final_duration_sum_times = time.time()
    print("训练集的usage和测试集的usage已经划分好了！")
    print("花费时间为:", time_final_duration_sum_times - time_start_duration_sum_times)

if __name__ == "__main__":

    time_start = time.time()
    # 读取原始文件
    file_path2 = '../user_app_usage.csv'
    # 过滤有用的APP
    effective_apps_file = "../effective_apps_max_features_5000.csv"
    user_app_usage_filter = filter_app_usage(file_path2, effective_apps_file)
    print("读取usage文件完毕!")

    # 对APP的duration还有times进行加和处理
    duration_sum_times(user_app_usage_filter, effective_apps_file)
    del user_app_usage_filter

    time_final = time.time()
    print("总共用时为：", time_final - time_start)



