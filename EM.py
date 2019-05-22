# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# 数据探索
pd.set_option('display.max_columns', None)
data = pd.read_csv('./heros.csv', encoding='gb18030')
# print(data.head())
# print(data.info())
# print(data.describe())
# print(data.describe(include=['O']))
# print(data['最大攻速'].value_counts())
# print(data['攻击范围'].value_counts())
# print(data['主要定位'].value_counts())

# 数据变换
data_dw = data['主要定位']
data.drop(['最大攻速', '主要定位', '次要定位 '], axis=1, inplace=True)
data['攻击范围'].replace(['近战', '远程'], [0, 1], inplace=True)
data.set_index(['英雄'], inplace=True)
# print(data.head())

# 标准化
data = data.astype('float64')
ss = StandardScaler()
n_data = ss.fit_transform(data)
print(n_data.shape)

# PCA 降维
# pca = PCA(n_components='mle')
pca = PCA(n_components=10)
n_data = pca.fit_transform(n_data)
print(n_data.shape)
print(pca.explained_variance_ratio_)


# EM聚类 使用GMM 高斯混合模型
gmm = GaussianMixture(n_components=30, covariance_type='full')
res = gmm.fit_predict(n_data)

res_data = pd.DataFrame({'分类结果':res, '定位':data_dw.values}, index=data.index)
res_data.reset_index(inplace=True)
res_data.sort_values(['分类结果'], inplace=True)

# res_data.to_csv('./result/res_30.csv', index=False)