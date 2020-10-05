# coding=utf-8
# /usr/bin/env python3
'''
Author:Fuxin Jiang
Email:jiangfuxin17@mails.ucas.ac.cn
'''
'''
数据说明，比赛数据(脱敏后)抽取的时间范是某连续30天的数据。总体上看，训练分为训练集数据文件、测试集数据文件、用户基本特征数据集、用户行为类汇总特征
数据集、用户激活过的app列表、30天的APP使用日志、APP类别元数据
age_train.csv代表训练样本，各字段之间由逗号隔开 1代表小于18岁、2代表19-23周岁、3代表24-34岁、4代表35-44岁、5代表45-54岁、6代表大于等于55周岁
训练数据总共2010000，测试数据502500
'''
'''
用户基本特征数据集user_basic_info.csv每一行代表一个用户的基本信息，包含用户人口属性、设备基本属性、各字段之间由逗号分隔，格式为:
"uld, gender, city, prodName, ramCapacity, ramLeftRation, romCapacity, romLeftRation, color, fontSize, ct,carrier, os "
用户标识（uId） 匿名化处理后的用户唯一标识（ID取值从1000001开始，依次递增）
性别（gender） 男/女（取值空间0,1）
常住地（city） 如深圳市、南京市等（匿名化处理，实际取值c001，c002….）
手机型号（prodName） 如mate10、honor 10等（匿名化处理，实际取值p001、p002……）
手机ram容量（ramCapacity） 手机ram的大小，以G为单位
ram剩余容量占比（ramLeftRation） 手机剩余的容量占总容量的比例
rom容量（romCapacity） 手机rom的大小，以G为单位
rom剩余容量占比（romLeftRation） 手机剩余rom容量占总rom容量的比例
手机颜色（color） 手机机身的颜色
字体大小（fontSize） 手机设置的字体大小
上网类型（ct） 2G/3G/4G/WIFI
移动运营商（carrier） 移动/联通/电信/其他
手机系统版本（os）AndroId操作系统的版本号
总共2512500条
'''
'''
用户行为类汇总特征数据集user_behavior_info.csv每行代表一个用户的行为类信息,包含对设备的使用行为汇总数据。
用户标识（uId） 匿名化处理后的用户唯一标识（ID取值从1000001开始，依次递增）
开机次数（bootTimes） 一段时间内(30天)手机的总开机次数
手机A特性使用次数（AFuncTimes） 一段时间内(30天) 手机A特性使用次数
手机B特性使用次数（BFuncTimes） 一段时间内(30天) 手机B特性使用次数
手机C特性使用次数（CFuncTimes） 一段时间内(30天) 手机C特性使用次数
手机D特性使用次数（DFuncTimes） 一段时间内(30天) 手机D特性使用次数
手机E特性使用次数（EFuncTimes） 一段时间内(30天) 手机E特性使用次数
手机F特性使用次数（FFuncTimes） 一段时间内(30天) 手机F特性使用次数
手机G特性使用情况（FFuncSum）   一段时间内(30天)G特性使用情况（数值)
总共2512500条
'''
'''
用户的激活APP列表文件user_app_actived.csv 每一行代表一条用户激活app的记录(APP激活的含义为用户安装并使用该APP)。特征文件格式为:
"uld, appld# appld# appld# appld# appld......"uld为用户标识，appld为app应用的唯一标识，多个app以"#"分隔
用户标识（uId） 匿名化处理后的用户唯一标识（ID取值从1000001开始，依次递增）
应用标识（appId） 匿名化处理后的app唯一标识
总共2512500条
'''
'''
app使用行为日志文件user_app_usage.csv存放了30天内按天统计每个用户对具体某个app的累计打开次数和使用时长，
用户标识（uId） 匿名化处理后的用户唯一标识（ID取值从1000001开始，依次递增）
应用标识（appId） 匿名化处理后的app唯一标识
使用时长（duration） 1天内用户对某app的累计使用时长
打开次数（times） 1天内用户对某app的累计打开次数
使用日期（use_date） 用户对某app的使用日期
总共651007719条
'''

'''
app对应类别文件app_info.csv每一行代表一条app的信息，格式如下:
应用标识（appId） appId为app应用的唯一标识
应用类型（category） app所属的应用类型
总共188864条
'''
import pandas as pd
from collections import Counter
from scipy import sparse
from sklearn import preprocessing as Pre
import multiprocessing
import time
train_path = "age_train.csv"
test_path = "age_test.csv"
user_basic_info_path = "user_basic_info.csv"
user_behavior_info_path = "user_behavior_info.csv"
app_info_path = "app_info.csv"
user_app_actived_path = "user_app_actived.csv"
def data_pre():
    time_start = time.time()
    #提取训练集样本
    data_train = pd.read_csv(train_path, header=None)
    data_train.columns = ['uid', 'label']
    #提取测试集样本
    data_test = pd.read_csv(test_path, header=None)
    data_test.columns = ['uid']
    #读取用户基本信息
    user_basic_info = pd.read_csv(user_basic_info_path, header=None)
    user_basic_info.columns = ['uid', 'gender', 'city', 'prodName', 'ramCapacity', 'ramLeftRation', 'romCapacity',
                               'romLeftRation', 'color', 'fontSize', 'ct', 'carrier', 'os']
    # 看一下数据缺失情况
    print(user_basic_info.isnull().sum(axis=0))
    # 首先对离散属性缺失值进行众数补充(这些都是离散属性)
    names_categ = 'city prodName ramCapacity romCapacity color ct carrier os'.split()
    # 离散属性的值空间个数
    for cateq_name in names_categ:
        print(cateq_name, len(user_basic_info[cateq_name].unique()))
    # 首先计算出每一个离散属性的众数，然后再进行填充
    fillvals_cateq = user_basic_info[names_categ].mode()
    filldict_cateq = dict()
    for key in names_categ:
        filldict_cateq[key] = fillvals_cateq[key][0]
    user_basic_info.fillna(value=filldict_cateq, inplace=True)
    # 对连续属性进行均值补充
    names_cont = list(set(['uid', 'gender', 'city', 'prodName', 'ramCapacity', 'ramLeftRation', 'romCapacity', 'romLeftRation', 'color', 'fontSize', 'ct', 'carrier', 'os']) - set(names_categ))
    user_basic_info[names_cont] = user_basic_info[names_cont].fillna(user_basic_info[names_cont].mean())
    # 异常值处理
    user_basic_info.loc[user_basic_info['ramLeftRation'] >= 1, 'ramLeftRation'] =  user_basic_info['ramLeftRation'].mean()
    user_basic_info.loc[user_basic_info['romLeftRation'] >= 1, 'romLeftRation'] = user_basic_info['romLeftRation'].mean()
    # 对离散属性进行数字编码，为后面的one-hot进行准备，不过针对city和prodName，我们要对类别少的进行处理,类别少的单独作为一列

    prodName_mapping = {label: idx for idx, label in enumerate(user_basic_info['prodName'].unique().tolist())}
    user_basic_info['prodName'] = user_basic_info['prodName'].map(prodName_mapping)

    city_mapping = {label: idx for idx, label in enumerate(user_basic_info['city'].unique().tolist())}
    user_basic_info['city'] = user_basic_info['city'].map(city_mapping)
    #移动运营商进行one-hot编码
    carrier_mapping = {label: idx for idx, label in enumerate(user_basic_info['carrier'].unique().tolist())}
    user_basic_info['carrier'] = user_basic_info['carrier'].map(carrier_mapping)

    color_mapping = {label: idx for idx, label in enumerate(user_basic_info['color'].unique().tolist())}
    user_basic_info['color'] = user_basic_info['color'].map(color_mapping)
    #上网类型进行one-hot编码，然后进行后面进行存储
    ct_mapping = {label: idx for idx, label in enumerate(user_basic_info['ct'].unique().tolist())}
    user_basic_info['ct'] = user_basic_info['ct'].map(ct_mapping)

    names_categ = 'city prodName color ct carrier'.split()
    #首先提取出离散属性进行one-hot编码
    x_cateq = user_basic_info[names_categ].values
    enc = Pre.OneHotEncoder()
    enc.fit(x_cateq)
    #构造新的特征，例如总容量乘以剩余容量比例等于现在的有的容量
    user_basic_info.loc[:, 'ramLeft'] = user_basic_info['ramCapacity'] * user_basic_info['ramLeftRation']
    user_basic_info.loc[:, 'romLeft'] = user_basic_info['romCapacity'] * user_basic_info['romLeftRation']
    print(user_basic_info.head())
    print(user_basic_info.isnull().sum(axis=0))
    # 读取用户的行为信息
    user_behavior_info = pd.read_csv(user_behavior_info_path, header=None)
    user_behavior_info.columns = ['uid', 'bootTimes', 'AFuncTimes', 'BFuncTimes', 'CFuncTimes', 'DFuncTimes', 'EFuncTimes', 'FFuncTimes', 'GFuncTimes']
    # 看一下数据缺失情况
    print(user_behavior_info.isnull().sum(axis=0))
    # 进行异常值处理，这里采用均值补充
    user_behavior_info.loc[user_behavior_info['bootTimes'] < 0, 'bootTimes'] = user_behavior_info['bootTimes'].mean()
    user_behavior_info.loc[user_behavior_info['AFuncTimes'] < 0, 'AFuncTimes'] = user_behavior_info['AFuncTimes'].mean()
    user_behavior_info.loc[user_behavior_info['BFuncTimes'] < 0, 'BFuncTimes'] = user_behavior_info['BFuncTimes'].mean()
    user_behavior_info.loc[user_behavior_info['CFuncTimes'] < 0, 'CFuncTimes'] = user_behavior_info['CFuncTimes'].mean()
    user_behavior_info.loc[user_behavior_info['DFuncTimes'] < 0, 'DFuncTimes'] = user_behavior_info['DFuncTimes'].mean()
    user_behavior_info.loc[user_behavior_info['EFuncTimes'] < 0, 'EFuncTimes'] = user_behavior_info['EFuncTimes'].mean()
    user_behavior_info.loc[user_behavior_info['FFuncTimes'] < 0, 'FFuncTimes'] = user_behavior_info['FFuncTimes'].mean()
    user_behavior_info.loc[user_behavior_info['GFuncTimes'] < 0, 'GFuncTimes'] = user_behavior_info['GFuncTimes'].mean()
    app_info = pd.read_csv(app_info_path, header=None)
    app_info.columns = ['app_id', 'app_class']
    print(set(app_info['app_class']))
    app_class_to_id_dict = {}
    for class_name in set(app_info['app_class']):
        app_class_to_id_dict[class_name] = list(app_info.loc[app_info['app_class'] == class_name, 'app_id'])
    print("字典建立完毕！")
    # 计算每一个人激活app的总数
    user_app_actived = pd.read_csv(user_app_actived_path, header=None)
    user_app_actived.columns = ['uid', 'app_ids']

    app_total_list = []
    with open(user_app_actived_path, 'r') as fp:
        line = fp.readline()
        while line:
            appids = line.strip('').split(',')[1]
            app_total_list.append(len(set(appids.strip().split("#"))))
            line = fp.readline()
    user_app_actived['app_active_total'] = app_total_list
    print("激活APP总数计算完成!")
    # 计算每一位用户使用该类别的app个数
    def app_actived_num(app_class_name):
        print(app_class_name)
        app_total_list = []
        app_class_list = app_class_to_id_dict[app_class_name]
        with open(user_app_actived_path, 'r') as fp:
            line = fp.readline()
            while line:
                appids = line.strip().split(',')[1]
                app_total_list.append(len(set(appids.strip().split('#')) & set(app_class_list)))
                line = fp.readline()
        app_class_number[app_class_name] = app_total_list
    #只有manager里面的字典才能实现多线程里面的共享，这块要进行注意！！！
    with multiprocessing.Manager() as manager:
        app_class_number = manager.dict()
    #进行多进程对上面的进行计算解决
        multipro = []
        #选取有用的APP类别，去除没有用的APP类别，加快速度
        useful_attri = list(set(app_info['app_class'])-set(['合作壁纸*', '休闲娱乐', '模拟游戏', '角色游戏', '主题铃声', '策略游戏', '医疗健康', '体育射击', '电子书籍']))
        for i, class_name in enumerate(useful_attri):
            #定义进程的名字
            thread_name = "thead_%d" % i
            multipro.append(multiprocessing.Process(target=app_actived_num, name=thread_name, args=(class_name, )))
        for process in multipro:
            process.start()
        for process in multipro:
            process.join()
        print("多进程计算完毕!")
        for class_name in useful_attri:
            user_app_actived[class_name] = app_class_number[class_name]

    user_app_actived.drop(['app_ids'], axis=1, inplace=True)
    data_train = pd.merge(data_train, user_basic_info, how='left', on='uid')
    data_train = pd.merge(data_train, user_behavior_info, how='left', on='uid')
    data_train = pd.merge(data_train, user_app_actived, how='left', on='uid')
    data_test = pd.merge(data_test, user_basic_info, how='left', on='uid')
    data_test = pd.merge(data_test, user_behavior_info, how='left', on='uid')
    data_test = pd.merge(data_test, user_app_actived, how='left', on='uid')
    x_train_categ = enc.transform(data_train[names_categ].values)
    x_test_categ = enc.transform(data_test[names_categ].values)
    #存储离散特征
    sparse.save_npz("x_train_categ.npz", x_train_categ)
    sparse.save_npz("x_test_categ.npz", x_test_categ)
    #存储所有数据
    data_train.to_csv("data_train.csv", index=False, encoding="utf-8")
    data_test.to_csv("data_test.csv", index=False, encoding="utf-8")
    time_final = time.time()
    print(time_final-time_start)
    return data_train, data_test
if __name__ == "__main__":
    data_train, data_test = data_pre()

