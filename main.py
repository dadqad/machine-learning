#  pip install pandas
#  pip install numpy
#  pip install sklearn
#  pip install matplotlib

import pandas as pd 
import numpy as np
import scipy                      #统计分析库
from scipy import stats          #scipy库的stats模块
import sklearn                    #机器学习库
from sklearn.model_selection import train_test_split, GridSearchCV  #  train_test_split 划分数据集包  GridSearchCV 网格搜索超参数包
from sklearn.linear_model import LinearRegression  # 线性回归导包 
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
os.system('cls')
# matplotlib其实是不支持显示中文的 显示中文需要一行代码设置字体
predict_index = '压缩回弹性率（%）'
print('预测列名',predict_index)
mpl.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）
all_data = pd.read_excel('./C题数据.xlsx',sheet_name='data3')
all_data
train_data = all_data[['接收距离(cm)','热风速度(r/min)']]
label = all_data[['厚度mm','孔隙率（%）','压缩回弹性率（%）']]
test_data = [[38,33,28,23,38,33,28,23],[850,950,1150,1250,1250,1150,950,850]]
test_data =np.array(test_data)
test_data = test_data.T
print('待预测数据:',test_data)
model = GridSearchCV(LinearRegression(),cv=3,param_grid={},scoring='neg_mean_squared_error',refit=True)   #线性回归模型 MSE评价方式 网格搜索法 ， 3折交叉验证
model.fit(train_data,label[predict_index],)
print("训练集MSE:",abs(model.score(train_data,label[predict_index])))    #打印预测的MSE,该值越接近于0越好
print('预测结果',model.predict(test_data))