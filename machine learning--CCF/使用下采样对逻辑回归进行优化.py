import os
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
from gensim import corpora,models,similarities
import gensim
from gensim.models import word2vec
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score,KFold
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.externals import joblib
from numpy import *
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import xgboost as xgb
import csv
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV
#竞赛的评价指标为Logloss
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



kf = KFold()

df_news = pd.read_table("d://train.txt",names=['label','comment'],encoding='utf-8')

df_news_test = pd.read_table("d://test_new.txt",names=['comment'],encoding='gb18030')

count_classes = pd.value_counts(df_news['label'],sort=True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud class histogram")
plt.xlabel('Label')
plt.ylabel("Frequenct")

# 对数据进行下采样处理
x = df_news.ix[:,df_news.columns == 'comment']
y = df_news.ix[:,df_news.columns == 'label']

number_records_fraud = len(df_news[df_news.label == 1])
fraud_indices = np.array(df_news[df_news.label == 1].index)

normal_indices = df_news[df_news.label == 0].index

random_normal_indices = np.random.choice(normal_indices,number_records_fraud,replace=False)
random_normal_indices = np.array(random_normal_indices)

under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

under_sample_data = df_news.iloc[under_sample_indices,:]

x_undersample = under_sample_data.ix[:,under_sample_data.columns == 'comment']
y_undersample = under_sample_data.ix[:,under_sample_data.columns == 'label']

print('Percentage of normal transactions:',len(under_sample_data[under_sample_data.label == 0])/len(under_sample_data))
print("Percentage of fraud transactions: ",len(under_sample_data[under_sample_data.label == 1])/len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))

# content = df_news.comment.values.tolist()
content = x_undersample.comment.values.tolist()
content_S = []
for line in content:
    words = jieba.lcut(line)
    if(len(words) > 1 and words != '\r\n'):
        content_S.append(words)
content_test = df_news_test.comment.values.tolist()
content_test_S = []
for line in content_test:
    words = jieba.lcut(line)
    if(len(words) > 1 and words != '\r\n'):
        content_test_S.append(words)

df_content = pd.DataFrame({'content_S':content_S})
df_test_content = pd.DataFrame({'content_test_S':content_test_S})

stopwords = pd.read_csv('d://stopwords.txt',index_col=False,sep='\t',quoting=3,names=['stopword'],encoding='utf-8')
def drop_stopwords(contents,stopwords):
    contents_clean = []
    for line in contents:
        line_clean = []
        for word in line:
            if word in stopwords:
                continue
            line_clean.append(word)
        contents_clean.append(line_clean)
    return contents_clean

#调用停用词函数
contents = df_content.content_S.values.tolist()
stopwords = stopwords.stopword.values.tolist()
contents_clean = drop_stopwords(contents,stopwords)

stopwords_test = pd.read_csv('d://stopwords.txt',index_col=False,sep='\t',quoting=3,names=['stopword'],encoding='utf-8')
def drop_stopwords(contents_test,stopwords_test):
    contents_test_clean = []
    for line in contents_test:
        line_clean = []
        for word in line:
            if word in stopwords_test:
                continue
            line_clean.append(word)
        contents_test_clean.append(line_clean)
    return contents_test_clean
#调用停用词函数
contents_test = df_test_content.content_test_S.values.tolist()
stopwords_test = stopwords_test.stopword.values.tolist()
contents_test_clean = drop_stopwords(contents_test,stopwords_test)

df_content = pd.DataFrame({'contents_clean':contents_clean})
df_content_test = pd.DataFrame({'contents_clean':contents_test_clean})

#选定特征值和目标值并把其转换为DataFrame格式
df_train = pd.DataFrame({'contents_clean':contents_clean,'label':y_undersample['label']})

df_test = pd.DataFrame({'contents_test_clean':contents_test_clean})


#设定特征值和目标值
x = df_train['contents_clean'].values
y = df_train['label'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

z = df_test['contents_test_clean'].values

#1.对文本进行特征值化处理
#1.1把数据处理成CountVectorizer()需要的输入格式：要求每句话之间用逗号隔开
z_words = []
for i in range(len(z)):
    try:
        z_words.append(' '.join(z[i]))
    except:
        print(i.word_index)

words = []
for i in range(len(x_train)):
    try:
        words.append(' '.join(x_train[i]))
    except:
        print(i.word_index)
print("============================================")
print(words[0])


#再把特征值的测试集x_test处理成CountVectorizer()需要的输入格式
test_words = []
for i in range(len(x_test)):
    try:
        test_words.append(' '.join(x_test[i]))
    except:
        print(i.word_index)
print("=================================================")

#1.2文本分类时要对文本进行特征值化
vec = CountVectorizer(analyzer='word',max_features=4000,lowercase=False)
vec.fit(words)
vec.fit(test_words)
vec.fit(z_words)

# #使用tf-idf进行关键特征值抽取
# vec = TfidfVectorizer(analyzer='word',max_features=4000,lowercase=False)
# vec.fit(words)
# vec.fit(test_words)
# vec.fit(z_words)

#调用逻辑回归算法
lr = LogisticRegression(penalty = 'l2')
param = {'C':[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,3.0,1.6,1.7,1.8,1.9,2.1,2.2,2.3,2.4,100]}
gc_lr = GridSearchCV(lr,param_grid=param,cv = 3)
gc_lr.fit(vec.transform(words),y_train)
# lr.fit(vec.transform(words),y_train)
y_predict_lr = gc_lr.predict(vec.transform(test_words))
datas_lr = gc_lr.predict(vec.transform(z_words))
print("逻辑回归：================================")
print("准确率为：",gc_lr.score(vec.transform(test_words),y_test))
print("召回率:",classification_report(y_test,y_predict_lr))
np.savetxt("CCF_out_Logistic_undersample.txt",datas_lr,fmt=['%.g'],newline='\n')
print("查看选择的参数模型：",gc_lr.best_params_)
print("逻辑回归保存成功！")


#调用逻辑回归算法
lr = LogisticRegression(penalty = 'l2',solver = 'lbfgs',max_iter=80,class_weight={0:0.5,1:0.5})
param = {'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,2.0,2.5,3.0,1.6,1.7,1.8,1.9,2.1,2.2,2.3,2.4]}
gc_lr = GridSearchCV(lr,param_grid=param,cv = 3)
gc_lr.fit(vec.transform(words),y_train)
# lr.fit(vec.transform(words),y_train)
y_predict_proba_lr = gc_lr.predict_proba(vec.transform(test_words))
y_test_predictions_high_recall = y_predict_proba_lr[:,1] > 0.5
# datas_lr = gc_lr.predict(vec.transform(z_words)
print("逻辑回归：================================")
print("准确率为：",gc_lr.score(vec.transform(test_words),y_test))
print("召回率:",classification_report(y_test,y_test_predictions_high_recall))
np.savetxt("CCF_out_Logistic_undersample_proba.txt",datas_lr,fmt=['%.g'],newline='\n')
print("查看选择的参数模型：",gc_lr.best_params_)
print("逻辑回归proba保存成功！")

