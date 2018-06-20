#coding:utf-8

import pandas as pd
import numpy as np
import sys

df = pd.read_csv('data/LoanStats3a.csv',skiprows=1, low_memory = True)
df.drop('id', axis = 1, inplace = True)
df.drop('member_id', 1, inplace = True)
df.term.replace(to_replace= '[^0-9]+', value='',inplace = True, regex = True)
df.int_rate.replace('%',value = '',inplace = True, regex = True)
df.drop('sub_grade', 1, inplace = True)
df.drop('emp_title', 1, inplace = True)
df.emp_length.replace('n/a', np.nan, inplace = True)
df.emp_length.replace(to_replace='[^0-9]+', value='',inplace = True, regex= True)
#删除值都为空的列
df.dropna(axis = 1, how = 'all', inplace= True)
#删除值都为空的行
df.dropna(axis = 0, how = 'all', inplace= True)
#进一步分析原来数据，确定要删除的特征
# debt_settlement_flag_date     98 non-null object
# settlement_status             155 non-null object
# settlement_date               155 non-null object
# settlement_amount             155 non-null float64
# settlement_percentage         155 non-null float64
# settlement_term               155 non-null float64
df.drop(['debt_settlement_flag_date','settlement_status',\
         'settlement_date','settlement_amount',\
         'settlement_percentage','settlement_term'],\
         1, inplace = True)
#删除float类型中重复值较多的特征
# for col in df.select_dtypes(include=['float']).columns:
#     print('col {} has {}'.format(col, len(df[col].unique())))
# col delinq_2yrs has 13
# col inq_last_6mths has 29
# col mths_since_last_delinq has 96
# col mths_since_last_record has 114
# col open_acc has 45
# col pub_rec has 7
# col total_acc has 84
# col out_prncp has 1
# col out_prncp_inv has 1
# col collections_12_mths_ex_med has 2
# col policy_code has 1
# col acc_now_delinq has 3
# col chargeoff_within_12_mths has 2
# col delinq_amnt has 4
# col pub_rec_bankruptcies has 4
# col tax_liens has 3
df.drop(['delinq_2yrs','inq_last_6mths',\
         'mths_since_last_delinq','mths_since_last_record',\
         'open_acc','pub_rec','total_acc',\
         'out_prncp','out_prncp_inv','collections_12_mths_ex_med',\
         'policy_code','acc_now_delinq','chargeoff_within_12_mths',\
         'delinq_amnt','pub_rec_bankruptcies','tax_liens'],\
         1, inplace = True)
#删除object类型中重复值较多的特征
# for col in df.select_dtypes(include=['object']).columns:
#     print('col {} has {}'.format(col, len(df[col].unique())))
# 
# col term has 2
# col grade has 7
# col emp_length has 11
# col home_ownership has 5
# col verification_status has 3
# col issue_d has 55
# col pymnt_plan has 1
# col purpose has 14
# col zip_code has 837
# col addr_state has 50
# col earliest_cr_line has 531
# col initial_list_status has 1
# col last_pymnt_d has 113
# col next_pymnt_d has 99
# col last_credit_pull_d has 125
# col application_type has 1
# col hardship_flag has 1
# col disbursement_method has 1
# col debt_settlement_flag has 2
df.drop(['term','grade','emp_length','home_ownership',\
         'verification_status','issue_d','pymnt_plan',\
         'purpose','zip_code','addr_state','earliest_cr_line',\
         'initial_list_status','last_pymnt_d','next_pymnt_d',\
         'last_credit_pull_d','application_type','hardship_flag',\
         'disbursement_method','debt_settlement_flag'],\
         1, inplace = True)
# desc                       29243 non-null object
# title                      42523 non-null object
df.drop(['desc','title'], 1, inplace = True)
# 对标签进行处理
df.loan_status.replace('Fully Paid',int(1), inplace=True)
df.loan_status.replace('Charged Off',int(0), inplace=True)
df.loan_status.replace('Does not meet the credit policy. Status:Fully Paid',\
                       np.nan, inplace=True)
df.loan_status.replace('Does not meet the credit policy. Status:Charged Off',\
                       np.nan, inplace=True)
#删除标签是nan的实例
df.dropna(subset = ['loan_status'], axis=0, how ='any', inplace = True)
#用0区填充所有的空值
df.fillna(0, inplace=True)
df.fillna(0.0, inplace=True)
#删除相关性较强的列(特征)
df.drop(['loan_amnt','funded_amnt','total_pymnt'], 1, inplace = True)
# cor = df.corr()
# cor.iloc[:,:] = np.tril(cor, k=-1)
# #相关系数矩阵重构
# cor = cor.stack()
# print(cor[(cor>0.55)|(cor<-0.55)])
# loan_amnt  
# funded_amnt
# total_pymnt 
##再次打印信息，查看是否有非float类型数据，将其做哑变量处理
# print(df.head(10))
df= pd.get_dummies(df)
df.to_csv('data/feature005.csv')
print(df.info())






























