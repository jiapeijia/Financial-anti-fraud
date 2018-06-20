#coding:utf-8
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model.logistic import LogisticRegression
from sklearn import metrics
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
import time
from sklearn.model_selection import GridSearchCV
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
 
 
 
#先用logistic进行训练和预测
df = pd.read_csv('data/feature005.csv')
df.drop(['recoveries', 'collection_recovery_fee', 'total_rec_prncp'], axis = 1, inplace =True)
# print(df.head(5))
Y = df.loan_status
X = df.drop('loan_status', 1, inplace = False)
# print(X.shape)
# print(X.columns)
# print(X)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)
#=========================================================================================
# print(x_train)
lr = LogisticRegression()
# lr.score(X, y, sample_weight)
start= time.time()
lr.fit(x_train, y_train)
train_predict = lr.predict(x_train)
train_f1 = metrics.f1_score(train_predict, y_train)
train_acc = metrics.accuracy_score(train_predict, y_train)
train_rec = metrics.recall_score(train_predict, y_train)
print("逻辑回归模型上的效果入下：")
print("在训练集上f1_mean的值为%.4f" % train_f1, end=' ')
print("在训练集上的精确率的值为%.4f" % train_acc, end=' ')
print("在训练集上的查全率的值为%.4f" % train_rec)
test_predict = lr.predict(x_test)
test_f1 = metrics.f1_score(test_predict, y_test)
test_acc = metrics.accuracy_score(test_predict, y_test)
test_rec = metrics.recall_score(test_predict, y_test)
print("在测试集上f1_mean的值为%.4f" % test_f1, end = ' ')
print("在训练集上的精确率的值为%.4f" % test_acc, end=' ')
print("在训练集上的查全率的值为%.4f" % test_rec)
end = time.time()
print(end-start)
#=================================================================
print("随机森林效果如下" + "=" * 30)
rf = RandomForestClassifier()
start = time.time()
rf.fit(x_train, y_train)
train_predict = rf.predict(x_train)
train_f1 = metrics.f1_score(train_predict, y_train)
train_acc = metrics.accuracy_score(train_predict, y_train)
train_rec = metrics.recall_score(train_predict, y_train)
print("在训练集上f1_mean的值为%.4f" % train_f1, end=' ')
print("在训练集上的精确率的值为%.4f" % train_acc, end=' ')
print("在训练集上的查全率的值为%.4f" % train_rec)
test_predict = rf.predict(x_test)
test_f1 = metrics.f1_score(test_predict, y_test)
test_acc = metrics.accuracy_score(test_predict, y_test)
test_rec = metrics.recall_score(test_predict, y_test)
print("在测试集上f1_mean的值为%.4f" % test_f1, end = ' ')
print("在训练集上的精确率的值为%.4f" % test_acc, end=' ')
print("在训练集上的查全率的值为%.4f" % test_rec)
end = time.time()
print(end - start)
#====================================================================== 
# print("GBDT上效果如下" + "=" * 30)
# gb = GradientBoostingClassifier()
# start = time.time()
# gb.fit(x_train, y_train)
# train_predict = gb.predict(x_train)
# train_f1 = metrics.f1_score(train_predict, y_train)
# train_acc = metrics.accuracy_score(train_predict, y_train)
# train_rec = metrics.recall_score(train_predict, y_train)
# print("在训练集上f1_mean的值为%.4f" % train_f1, end=' ')
# print("在训练集上的精确率的值为%.4f" % train_acc, end=' ')
# print("在训练集上的查全率的值为%.4f" % train_rec)
# test_predict = gb.predict(x_test)
# test_f1 = metrics.f1_score(test_predict, y_test)
# test_acc = metrics.accuracy_score(test_predict, y_test)
# test_rec = metrics.recall_score(test_predict, y_test)
# print("在测试集上f1_mean的值为%.4f" % test_f1, end = ' ')
# print("在训练集上的精确率的值为%.4f" % test_acc, end=' ')
# print("在训练集上的查全率的值为%.4f" % test_rec)
# end = time.time()
# print(end-start)
#======================================================================================================================
# print("支持向量机的效果如下" + "=" * 30)
# sv = svm.SVC(kernel = 'linear') #C = 1, probability = True, decision_function_shape = 'ovo', random_state = 0
# start = time.time()
# sv.fit(x_train, y_train)
# train_predict = sv.predict(x_train)
# train_f1 = metrics.f1_score(train_predict, y_train)
# train_acc = metrics.accuracy_score(train_predict, y_train)
# train_rec = metrics.recall_score(train_predict, y_train)
# print("在训练集上f1_mean的值为%.4f" % train_f1, end=' ')
# print("在训练集上的精确率的值为%.4f" % train_acc, end=' ')
# print("在训练集上的查全率的值为%.4f" % train_rec)
# test_predict = sv.predict(x_test)
# test_f1 = metrics.f1_score(test_predict, y_test)
# test_acc = metrics.accuracy_score(test_predict, y_test)
# test_rec = metrics.recall_score(test_predict, y_test)
# print("在测试集上f1_mean的值为%.4f" % test_f1, end = ' ')
# print("在训练集上的精确率的值为%.4f" % test_acc, end=' ')
# print("在训练集上的查全率的值为%.4f" % test_rec)
# end = time.time()
# print(end - start)
#================================================================================================================  
# start = time.time()
# parameters ={
#     'kernel':['linear','sigmoid','poly'],
#     'C':[0.01, 1],
#     'probability':[True, False]  
#     }
# clf = GridSearchCV(svm.SVC(random_state = 0), param_grid = parameters, cv = 5)
# clf.fit(x_train, y_train)
# print('最优参数是：',end=' ')
# print(clf.best_params_)
# print('最优模型准确率是：', end = ' ')
# print(clf.best_score_)
# end = time.time()
# print(end-start)
  
feature_importance = rf.feature_importances_#度量特征权重的接口
# print(feature_importance)
# print(feature_importance.max())
feature_importance = 100.0*(feature_importance/feature_importance.max())
index = np.argsort(feature_importance)[-13:]
plt.barh(np.arange(13), feature_importance[index], color = 'dodgerblue', alpha = 0.4)
print(np.array(X.columns)[index])
plt.yticks(np.arange(10+0.25), np.array(X.columns)[index])
plt.xlabel('Relative importance')
plt.title('Top 10 Importance Variable')
plt.show()   




