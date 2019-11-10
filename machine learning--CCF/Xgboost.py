from sklearn.metrics import mean_absolute_error,make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold,train_test_split
from scipy.sparse import csr_matrix,hstack
from xgboost import XGBRegressor,XGBClassifier
import os
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
from gensim import corpora,models,similarities
import gensim
from sklearn.model_selection import train_test_split,GridSearchCV
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
from xgboost import XGBClassifier
import csv


df_news = pd.read_table("d://train.txt",names=['label','comment'],encoding='utf-8')
df_news_test = pd.read_table("d://test_new.txt",names=['comment'],encoding='gb18030')

# jieba分词器要求我们输入的是List形式
content = df_news.comment.values.tolist()
content_S = []
for line in content:
    words = jieba.lcut(line)
    if(len(words) > 1 and words != '\r\n'):
        content_S.append(words)
# print(Content_S)

content_test = df_news_test.comment.values.tolist()
content_test_S = []
for line in content_test:
    words = jieba.lcut(line)
    if(len(words) > 1 and words != '\r\n'):
        content_test_S.append(words)



#将分好词的列表转换成DataFrame形式（因为list没有value这个属性）
df_content = pd.DataFrame({'content_S':content_S})
df_test_content = pd.DataFrame({'content_test_S':content_test_S})

#读入停用词表来删除停用词
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

# #可使用jiebe.analyse.extract_tags()提取关键字
# index = 2400
# print(df_news['comment'][index])
# content_S_str = "".join(content_S[index])
# print(" ".join(jieba.analyse.extract_tags(content_S_str,topK=5,withWeight=False)))
#
# #LDA：主题模型
# dictionary = corpora.Dictionary(contents_clean)
# corpus = [dictionary.doc2bow(sentence) for sentence in contents_clean]
#
# lda = gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=20)#类似Kmeans自己指定k值
#
# #一号分类结果
# print(lda.print_topic(1,topn=5))
#
# for topic in lda.print_topics(num_topics = 20,num_words = 5):
#     print(topic[1])

#选定特征值和目标值并把其转换为DataFrame格式
df_train = pd.DataFrame({'contents_clean':contents_clean,'label':df_news['label']})
print(df_train.tail())
df_test = pd.DataFrame({'contents_test_clean':contents_test_clean})
print(df_test.tail())


#设定特征值和目标值
x = df_train['contents_clean'].values
y = df_train['label'].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

z = df_test['contents_test_clean'].values

# pca.fit(z_words)
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
# vec = CountVectorizer(analyzer='word',max_features=4000,lowercase=False)
# vec.fit(words)
# vec.fit(test_words)
# vec.fit(z_words)


cv = CountVectorizer()
cv.fit(words)
cv.fit(test_words)
cv.fit(z_words)
tf = TfidfTransformer()
tfidf = tf.fit_transform(cv.transform(words))
weight = tfidf.toarray()
print(tfidf.shape)
test_tfidf = tf.transform(cv.transform(test_words))
test_weight = test_tfidf.toarray()
z_tfidf = tf.fit_transform(cv.transform(z_words))
z_weight = z_tfidf.toarray()
print(test_weight.shape)

train_opinion = np.array(y_train)
dtrain = xgb.DMatrix(weight,label = train_opinion)
dtest = xgb.DMatrix(test_weight)
dz = xgb.DMatrix(z_weight)
params = {'max_depth':6,'eta':0.5,'eval_metric':'merror','silent':1,'objective':'multi:softmax','num_class':11}
# params = {
#     'seed': 0,
#     'eta': 0.1,
#     'colsample_bytree': 0.5,
#     'silent': 1,
#     'subsample': 0.5,
#     'objective': 'reg:linear',
#     'max_depth': 5,
#     'min_child_weight': 3
# }
num_round = 100
bst = xgb.train(params, dtrain,num_round)
predict = bst.predict(dtest)
datas = bst.predict(dz)
print(predict)
print("Xgboost召回率:",classification_report(y_test,predict))
np.savetxt("CCF_out_Xgboost.txt",datas,fmt=['%.g'],newline='\n')
print("Xgboost结果保存成功！")
#使用tf-idf进行关键特征值抽取
#vec = TfidfVectorizer(analyzer = 'word',max_features=4000,lowercase=False)
# vec.fit(words)
# vec.fit(z_words)

# print(words)
# new_vector=vec.transform(words)
# print(type(new_vector))

# vec = TfidfTransformer()
# vec.fit(cv.transform(words))
# vec.fit(cv.transform(z_words))

# #调用朴素贝叶斯算法：
# mlt = MultinomialNB(alpha=1.0)
# mlt.fit(vec.transform(words),y_train)
# # y_predict = mlt.predict(vec.transform(test_words))
# y_predict = mlt.predict(vec.transform(test_words))
# # datas = mlt.predict(vec.transform(test_words))
# # datas = mlt.predict(vec.transform(z_words))
# datas = mlt.predict(vec.transform(z_words))
# # print(datas)
# # print(type(datas))
# print("朴素贝叶斯：================================")
# print("准确率为：",mlt.score(vec.transform(test_words),y_test))
# print("召回率:",classification_report(y_test,y_predict))
# np.savetxt("CCF_out.txt",datas,fmt=['%.g'],newline='\n')
# # datas.to_csv('d://file.csv')
# file = open('d://CCF_out_Bayes.txt','w')
# # file.writable(str(datas_variable))
# print("朴素贝叶斯保存成功！")

# # #调用SVM算法
# s = svm.SVC(C=2,kernel='rbf',gamma=10,decision_function_shape='ovr')
# s.fit(vec.transform(words),y_train)
# y1_predict = s.predict(vec.transform((test_words)))
# print("SVM：================================")
# print("准确率为：",s.score(vec.transform(test_words),y_test))
# print("召回率:",classification_report(y_test,y1_predict))

# # 调用随机森林算法
# rf = RandomForestClassifier()
# param = {"n_estimators": [30,60,90,120,150],"max_depth":[15,25,30,35,40]}
# gc = GridSearchCV(rf,param_grid = param,cv = 5)
# gc.fit(vec.transform(words),y_train)
# y2_predict = gc.predict(vec.transform(test_words))
# print("随机森林：================================")
# print("准确率为：",gc.score(vec.transform(test_words),y_test))
# print("召回率:",classification_report(y_test,y2_predict))
# print("查看选择的参数模型：",gc.best_params_)

# #调用决策树算法
# dt = DecisionTreeClassifier(criterion='gini')
# param_tree = {"max_depth":[25,30,35,40,45,50,55,60,65]}
# gc_tree = GridSearchCV(dt,param_grid = param_tree,cv=4)
# gc_tree.fit(vec.transform(words),y_train)
# y3_predict = gc_tree.predict(vec.transform(test_words))
# datas_tree = gc_tree.predict(vec.transform(z_words))
# print("决策树：================================")
# print("准确率为：",gc_tree.score(vec.transform(test_words),y_test))
# print("召回率:",classification_report(y_test,y3_predict))
# print("查看选择的参数模型：",gc_tree.best_params_)
# np.savetxt("CCF_out_tree.txt",datas_tree,fmt=['%.g'],newline='\n')
# # datas.to_csv('d://file.csv')
# file = open('d://CCF_out_tree.txt','w')
# # file.writable(str(datas_variable))
# print("决策树保存成功！")

#调用KNN算法
# knn = KNeighborsClassifier(algorithm='auto')
# param_knn = {"n_neighbors":[3,5,8,10]}
#
# gc_knn = GridSearchCV(knn,param_grid=param_knn,cv=3)
# gc_knn.fit(vec.transform(words),y_train)
# y4_predict = gc_knn.predict(vec.transform(test_words))
# datas_knn = mlt.predict(vec.transform(z_words))
# print("KNN：================================")
# print("准确率为：",gc_knn.score(vec.transform(test_words),y_test))
# print("召回率:",classification_report(y_test,y4_predict))
# print("查看选择的参数模型：",gc_knn.best_params_)
# np.savetxt("CCF_out_KNN.txt",datas_knn,fmt=['%.g'],newline='\n')
# # datas.to_csv('d://file.csv')
# file = open('d://CCF_out_KNN.txt','w')
# # file.writable(str(datas_variable))
# print("KNN保存成功！")



# np.savetxt("CCF_out.txt",datas,fmt=['%.g'],newline='\n')
# datas.to_csv('d://file.csv')
# file = open('d://CCF_out.txt','w')
# # file.writable(str(datas_variable))
# print("保存成功！")







