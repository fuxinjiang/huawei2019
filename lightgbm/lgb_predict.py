#首先利用xgboost进行预测
#主要对一万维的数据进行分析
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import json
import codecs
import os
import lightgbm as lgb
from sklearn.model_selection import  KFold
from scipy import sparse
from sklearn.metrics import f1_score

def lgb_model(config, lgb_train, num_round, valid_sets, early_stopping_rounds, verbose_eval):
	'''

	:param config:lightgbm的基本参数
	:param lgb_train: 训练数据集
	:param num_round: 总共训练的次数
	:param valid_sets: 验证集合
	:param early_stopping_rounds:提前停止
	:return:
	'''
	lgb_mdel = lgb.train(config, lgb_train, num_boost_round = num_round, valid_sets = valid_sets, early_stopping_rounds=early_stopping_rounds, verbose_eval = verbose_eval)
	return lgb_mdel

def save_lgbmodel(model, file_path):
	if not os.path.exists(file_path):
		os.makedirs(file_path)
	model.save_model(file_path + '/lgb_model.txt', num_iteration=lgb_model.best_iteration)


if __name__ == "__main__":

	# 首先读取所有的信息
	data_train = pd.read_csv("data_train.csv")
	data_test = pd.read_csv("data_test.csv")
	# 将里面的分类属性去除
	# names_categ = 'city prodName color ct carrier'.split()
	names_cont = list(set(data_train.columns.tolist())-set(['uid', 'label']))
	# 读取连续的属性
	train_x_cont = np.array(data_train[names_cont].values)
	test_x_cont = np.array(data_test[names_cont].values)
	# 读取离散的属性
	# train_x_categ = sparse.load_npz("x_train_categ.npz")
	# test_x_categ = sparse.load_npz("x_test_categ.npz")
	# actived里面出现频率前面5000个app
	X_act_cv = sparse.load_npz("train_actived_cv_max_features_5000.npz")
	X_act_cv_test = sparse.load_npz("test_actived_cv_max_features_5000.npz")
	#根据actived里面5000个app，提取每一个app在30天内的总使用时间和总使用次数
	train_sparse_matrix_duration_npz = sparse.load_npz("./duration_and_times/train_sparse_matrix_duration_sum.npz")
	test_sparse_matrix_duration_npz = sparse.load_npz("./duration_and_times/test_sparse_matrix_duration_sum.npz")
	train_sparse_matrix_times_npz = sparse.load_npz("./duration_and_times/train_sparse_matrix_times_sum.npz")
	test_sparse_matrix_times_npz = sparse.load_npz("./duration_and_times/test_sparse_matrix_times_sum.npz")
	#print(np.shape(train_x_categ))
	#print(np.shape(test_x_categ))
	print(np.shape(train_sparse_matrix_duration_npz))
	print(np.shape(test_sparse_matrix_duration_npz))
	print(np.shape(train_sparse_matrix_times_npz))
	print(np.shape(test_sparse_matrix_times_npz))
	train_x = sparse.hstack((train_x_cont, X_act_cv, train_sparse_matrix_duration_npz, train_sparse_matrix_times_npz))
	print(np.shape(train_x))
	age_label_index = np.loadtxt(open("age_train.csv", "rb"), delimiter=",")
	train_y = age_label_index[:, 1]-1
	x_train, x_dev, y_train, y_dev = train_test_split(train_x, train_y, test_size=0.2)
	uid_df = pd.read_csv("age_test.csv", sep=',', header=None)
	uid_df.columns = ['id']
	x_test = sparse.hstack((test_x_cont, X_act_cv_test, test_sparse_matrix_duration_npz, test_sparse_matrix_times_npz))
	print(np.shape(x_train))
	print(np.shape(x_dev))
	print(np.shape(x_test))
	#lgbtm模型
	config_lgb = json.load(codecs.open('lgb_config.json', 'r', 'utf-8'))
	print("模型设置参数为:")
	print(config_lgb)
	data_train_lgb = lgb.Dataset(x_train, label=y_train) # 组成训练集
	data_dev_lgb = lgb.Dataset(x_dev, label=y_dev)
	num_round = 5000
	early_stopping_rounds = 100
	verbose_eval = 10
	lgb_model = lgb_model(config_lgb, data_train_lgb, num_round, [data_train_lgb, data_dev_lgb], early_stopping_rounds, verbose_eval)
	save_lgbmodel(lgb_model, os.getcwd() + '/model')
	test_prediction = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration)
	train_prediction = lgb_model.predict(x_train, num_iteration=lgb_model.best_iteration)
	np.save('lgb_train_prediction_prob.npy', train_prediction)
	np.save('lgb_test_prediction_prob.npy', test_prediction)
	test_prediction_label = np.argmax(test_prediction, axis=1)
	test_df = pd.DataFrame(test_prediction_label+1)
	test_df.columns = ['label']
	data_final = pd.DataFrame()
	data_final['id'] = uid_df['id']
	data_final['label'] = test_df['label'].astype(int)
	data_final.to_csv('submission_lgb.csv', index=False)

