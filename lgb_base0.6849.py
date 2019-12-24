#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 20:41:36 2019

@author: boushibun
"""

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import lightgbm as lgb
#import math
from sklearn.model_selection import KFold, StratifiedKFold
#from sklearn.utils import shuffle
import matplotlib.pyplot as plt
train1 = pd.read_csv("first_round_training_data.csv")
train2 = pd.read_csv("second_round_training_data.csv")
train = pd.concat([train1,train2],axis=0,ignore_index=True)
test = pd.read_csv("second_round_testing_data-2.csv")   
quality_map = {'Excellent': 0, 'Good': 1, 'Pass': 2, 'Fail': 3}
train['label'] = train['Quality_label'].map(quality_map)
# 将标签onehot编码，方便管理和统计。
# idea1：可以考虑做4个二分类，或者尝试使用mae 做loss 或者mse 做loss，并观察结果的分布以及线下得分。
train = pd.get_dummies(train, columns=['Quality_label'])
bin_label = ['Quality_label_Excellent', 'Quality_label_Good', 'Quality_label_Pass', 'Quality_label_Fail']

data = pd.concat([train, test], ignore_index=True)
data['id'] = data.index
# 因为1，属性特征目测是连续特征，2，数值大小分布差异过大，所以将属性做log变换之后做处理，会更合适一些。也可以考虑分桶处理
# 为什么做log变换更合理呢？试想一下，假如统计一个人某个数值属性，发现是如下的一个列表[1,2,1.1,1.5,20],
# 这种场景分布偏差较大的情况下，如果取均值作为特征，是否合适？
# 再比如，如果预测一组数值，一条极端数据对结果的影响超过了N多的数据，这样的模型是否是一个好的模型？

# 参数特征这里使用5-10
para_feat = ['Parameter{0}'.format(i) for i in [1,2,3,4,5,6,7,8,10]]
# 属性特征
attr_feat = ['Attribute{0}'.format(i) for i in range(1, 11)]+['Parameter9']

data[attr_feat] = np.log1p(data[attr_feat])
# or data[attr_feat] = np.log10(data[attr_feat] + 1)
# 此时绘图观察分布显得合理很多
for i in attr_feat:
    data[i].hist()
    plt.show()

# 使用预测属性的model去预测属性，
# 1、测试集没有属性，该怎么用？当然是预测它了。预测的方法，可以由模型获得
# 2、训练集的属性怎么用？既然测试集的属性是预测出来的，训练集也应该用同等性质的属性，也就是5折交叉预测出来的属性。
# 3、第一次想到该方法于2018年的光伏预测赛中，首见成效，之后教与ration，后在icme中，也用到类似方法。
# 4、该方法带来的提升目前不太稳定。6750 -- 6800。

def get_predict_w(model, data, label='label', feature=[], cate_feature=[], random_state=2018, n_splits=5,
                  model_type='lgb'):
    if 'sample_weight' not in data.keys():
        data['sample_weight'] = 1
    model.random_state = random_state
    predict_label = 'predict_' + label
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    data[predict_label] = 0
    test_index = (data[label].isnull()) | (data[label] == -1)
    train_data = data[~test_index].reset_index(drop=True)
    test_data = data[test_index]

    for train_idx, val_idx in kfold.split(train_data):
        model.random_state = model.random_state + 1

        train_x = train_data.loc[train_idx][feature]
        train_y = train_data.loc[train_idx][label]

        test_x = train_data.loc[val_idx][feature]
        test_y = train_data.loc[val_idx][label]
        if model_type == 'lgb':
            try:
                model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
                          eval_metric='mae',
                          categorical_feature=cate_feature,
                          sample_weight=train_data.loc[train_idx]['sample_weight'],
                          verbose=100
                          )
            except:
                model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
                          eval_metric='mae',
                          # categorical_feature=cate_feature,
                          sample_weight=train_data.loc[train_idx]['sample_weight'],
                          verbose=100)
        elif model_type == 'ctb':
            model.fit(train_x, train_y, eval_set=[(test_x, test_y)], early_stopping_rounds=100,
                      eval_metric='mae',
                      cat_features=cate_feature,
                      sample_weight=train_data.loc[train_idx]['sample_weight'],
                      verbose=100)
        train_data.loc[val_idx, predict_label] = model.predict(test_x)
        if len(test_data) != 0:
            test_data[predict_label] = test_data[predict_label] + model.predict(test_data[feature])
    test_data[predict_label] = test_data[predict_label] / n_splits
    return pd.concat([train_data, test_data], sort=True, ignore_index=True), predict_label


lgb_attr_model = lgb.LGBMRegressor(
    boosting_type="gbdt", num_leaves=31, reg_alpha=10, reg_lambda=5,
    max_depth=7, n_estimators=500,
    subsample=0.7, colsample_bytree=0.4, subsample_freq=2, min_child_samples=10,
    learning_rate=0.05, random_state=2019,
)

features = para_feat
for i in attr_feat:
    data, predict_label = get_predict_w(lgb_attr_model, data, label=i,
                                        feature=features, random_state=2019, n_splits=5)
    print(predict_label, 'done!!')

# 该方案共获得10个属性特征。
pred_attr_feat = ['predict_Attribute{0}'.format(i) for i in range(1, 11)]+['predict_Parameter9']
# int特征
for i in para_feat:
    data['int%s'%i]= data[i].apply(lambda x: round(x,2))
int_feat= ['intParameter{0}'.format(i) for i in [5,6,7,8,10]] 
#count特征
def counts_(columns):
    for col in columns:
        data['count%s'%col]=0
        reset_size=data.groupby([col]).size().reset_index()[col].values
        value_counts=data.groupby([col]).size().reset_index()[0].values
        for i,v in enumerate(reset_size):
            data['count%s'%col][data[col]==v]=value_counts[i]
counts_(['Parameter7','Parameter8']) 
count_feat= ['countParameter{0}'.format(i) for i in [7,8]]
#乘积特征
#pro特征
def pro_(columns):
    for col in columns:
        data['pro%s'%col]=0
        data['pro%s'%col]=data[col]*data[col]
pro_(['Parameter5','Parameter6','Parameter10']) 
pro_feat= ['proParameter{0}'.format(i) for i in [5,6,10]]
#交叉乘积特征
data['cross56']=data['Parameter5']*data['Parameter6']
data['cross510']=data['Parameter5']*data['Parameter10']
data['cross610']=data['Parameter6']*data['Parameter10']
cross_feat= ['cross{0}'.format(i) for i in [56,510,610]]
#nunique
def nunique_(columns):
    for col in columns:
        for j in ['Parameter1','Parameter2','Parameter3','Parameter4',]:
            data['nuni%s'%col+j[-1]]=data[col].map(data.groupby(col)[j].nunique())
nunique_(['Parameter5','Parameter6','Parameter7','Parameter8','Parameter10'])       
nunique_feat=['nuniParameter{0}'.format(i) for i in [51,52,53,54,61,62,63,64,71,72,
              73,74,81,82,83,84,101,102,103,104]]
##分箱特征
def qcut_(columns):
    for col in columns:
            data['qcut%s'%col]=pd.qcut(data[col].rank(method='first'),10,labels=False)
qcut_(['Parameter10'])            
#qcut_(['Parameter1','Parameter2','Parameter3','Parameter4',
       #'Parameter5','Parameter6','Parameter7','Parameter8','Parameter10'])
qcut_feat=['qcutParameter{0}'.format(i) for i in [10]]
#qcut_feat=['qcutParameter{0}'.format(i) for i in [5,6,7,8,10]]

###unique特征
def uni_(columns,i):
     data['uni%s'%i]=data[columns].apply(lambda x : 1 if x == data[columns].unique()[i] else 0)       
uni_('Parameter7',14)
uni_('Parameter7',8)
uni_('Parameter7',6)
uni_('Parameter7',4)
uni_feat= ['uni{0}'.format(i) for i in [14,8,6,4]]
#uniqs
a=data['Parameter8'].unique()
a.sort()
data['multiple1']=data['Parameter8'].apply(lambda x : 1 if x in [a[0],a[6],a[10],a[15],
     a[20],a[5],a[9],a[14],a[19],a[23],a[4]] else 0)
data['multiple2']=data['Parameter8'].apply(lambda x : 1 if x in [a[7],a[1],a[11],a[17],a[21]] else 0)    
data['multiple3']=data['Parameter8'].apply(lambda x : 1 if x in [a[2],a[8],a[3],a[12],
     a[18],a[13],a[22],a[24],a[26]] else 0) 
data['multiple4']=data['Parameter8'].apply(lambda x : 1 if x in [a[16]] else 0) 
mul_feat= ['multiple{0}'.format(i) for i in [1,2,3,4]]
####nul_feat
a=data['Parameter10'].unique()
a.sort()
data['nultiple1']=data['Parameter10'].apply(lambda x : 1 if x in [a[10],a[2],a[18],a[27],a[36]] else 0)
data['nultiple2']=data['Parameter10'].apply(lambda x : 1 if x in [a[11],a[3],a[20],a[29],a[38]] else 0)    
data['nultiple3']=data['Parameter10'].apply(lambda x : 1 if x in [a[13],a[4],a[22],a[31],a[40]] else 0) 
data['nultiple4']=data['Parameter10'].apply(lambda x : 1 if x in [a[14],a[5],a[23]] else 0) 
data['nultiple5']=data['Parameter10'].apply(lambda x : 1 if x in [a[15],a[7],a[24],a[33],a[34],
     a[16],a[8],a[42],a[43],a[25],a[0]] else 0) 
data['nultiple6']=data['Parameter10'].apply(lambda x : 1 if x in [a[17],a[9],a[26],a[35],a[44],a[1]] else 0)
data['nultiple7']=data['Parameter10'].apply(lambda x : 1 if x in [a[15],a[6]] else 0)
data['nultiple8']=data['Parameter10'].apply(lambda x : 1 if x in [a[21],a[12],a[30],a[39]] else 0)    
data['nultiple9']=data['Parameter10'].apply(lambda x : 1 if x in [a[19],a[28],a[37]] else 0) 
data['nultiple10']=data['Parameter10'].apply(lambda x : 1 if x in [a[41],a[32]] else 0) 
data['nultiple11']=data['Parameter10'].apply(lambda x : 1 if x in [a[45]] else 0) 
nul_feat= ['nultiple{0}'.format(i) for i in range(1,12)]
####lul_feat
a=data['Parameter7'].unique()
a.sort()
data['lultiple1']=data['Parameter7'].apply(lambda x : 1 if x in [a[0],a[3],a[6],a[9],a[13]] else 0)
data['lultiple2']=data['Parameter7'].apply(lambda x : 1 if x in [a[1],a[4],a[7],a[10]] else 0)    
data['lultiple3']=data['Parameter7'].apply(lambda x : 1 if x in [a[2],a[5],a[8],a[11],a[14],a[12]] else 0) 
lul_feat= ['lultiple{0}'.format(i) for i in range(1,4)]
#####qul_feat
a=data['Parameter6'].unique()
a.sort()
data['qultiple1']=data['Parameter6'].apply(lambda x : 1 if x in [a[4],a[16],a[14],a[13],a[15],a[27],a[38],a[5],
     a[17],a[28],a[3],a[12],a[26],a[39],a[37],a[49],a[50],a[51],a[52],a[62],a[63],a[64],a[65],a[75],a[76],a[77],a[78],
     a[89],a[90]] else 0)
data['qultiple2']=data['Parameter6'].apply(lambda x : 1 if x in [a[6],a[18],a[29],a[40],a[7],a[19],a[30],a[41],a[53],
     a[54],a[66],a[67],a[79],a[80]] else 0)    
data['qultiple3']=data['Parameter6'].apply(lambda x : 1 if x in [a[11],a[25],a[24],a[35],a[36],a[1],a[10],a[2],
     a[23],a[33],a[34],a[22],a[44],a[45],a[46],a[47],a[48],a[57],a[58],a[59],a[60],a[61],a[70],a[72],a[73],a[74],
      a[83],a[84],a[85],a[86],a[87],a[88]] else 0)
data['qultiple4']=data['Parameter6'].apply(lambda x : 1 if x in [a[8],a[20],a[31],a[42],a[9],a[21],a[32],a[43],a[55],a[56],
     a[68],a[69],a[81],a[82]] else 0)
qul_feat= ['qultiple{0}'.format(i) for i in range(1,4)]

lgb_mc_model = lgb.LGBMClassifier(
    boosting_type="gbdt", num_leaves=23, reg_alpha=10, reg_lambda=5,
    max_depth=5, n_estimators=1300, objective='multiclass',
    subsample=0.7, colsample_bytree=0.7, subsample_freq=1, min_child_samples=5,
    learning_rate=0.05, random_state=42,
)

features = pred_attr_feat + int_feat + count_feat+nunique_feat+pro_feat+mul_feat+nul_feat+lul_feat+qul_feat#+uni_feat+cross_feat#+qcut_feat#+pro_feat

X = data[~data.label.isnull()][features]
y = data[~data.label.isnull()]['label']

skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
y_test = 0
# best_score = []
best_iter = []
for index, (trn_idx, test_idx) in enumerate(skf.split(X, y)):
    print(index)
    train_x, test_x, train_y, test_y = X.loc[trn_idx], X.loc[test_idx], y.loc[trn_idx], y.loc[test_idx]
    eval_set = [(test_x, test_y)]
    lgb_mc_model.fit(train_x, train_y, eval_set=eval_set,
                     #early_stopping_rounds=100,
                     verbose=100
                     )

    y_pred = lgb_mc_model.predict_proba(X.loc[test_idx][features])
    for i in range(4):
        data.loc[test_idx, 'pred_{0}'.format(i)] = y_pred[:, i]

    y_test = lgb_mc_model.predict_proba(data[data.label.isnull()][features],lgb_mc_model.best_iteration_) / 5 + y_test
    # best_score.append(lgb_mc_model.best_score_['valid_0']['multi_logloss'])
    best_iter.append(lgb_mc_model.best_iteration_)


sub = data[data.label.isnull()][['Group']].astype(int)
for i in range(4):
    sub['pred_{0}'.format(i)] = y_test[:, i]
labels = ['Excellent ratio', 'Good ratio', 'Pass ratio', 'Fail ratio']
sub.columns = ['Group'] + labels
sub = sub.groupby('Group')[labels].mean().reset_index()
sub[['Group'] + labels].to_csv('sub.csv', index=False)

# [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
# 这种玩意看着真难受，谁有法子弄掉 ，盼复
