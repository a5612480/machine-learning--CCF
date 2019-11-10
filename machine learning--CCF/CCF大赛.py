import os
import numpy as np
import pandas as pd
import jieba
import jieba.analyse
from gensim import corpora,models,similarities
import gensim
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from numpy import *


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
stopwords = pd.read_csv('d://stop_words_zh.txt',index_col=False,sep='\t',quoting=3,names=['stopword'],encoding='utf-8')
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

stopwords_test = pd.read_csv('d://stop_words_zh.txt',index_col=False,sep='\t',quoting=3,names=['stopword'],encoding='utf-8')
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
print(words[0])
#1.2文本分类时要对文本进行特征值化
# cv = CountVectorizer(analyzer='word',max_features=4000,lowercase=False)
# cv.fit(words)

#再把特征值的测试集x_test处理成CountVectorizer()需要的输入格式
test_words = []
for i in range(len(x_test)):
    try:
        test_words.append(' '.join(x_test[i]))
    except:
        print(i.word_index)


#使用tf-idf进行关键特征值抽取
vec = TfidfVectorizer(analyzer = 'word',max_features=4000,lowercase=False)
vec.fit(words)

# print(words)
new_vector=vec.transform(words)
# print(type(new_vector))
#调用朴素贝叶斯算法：
mlt = MultinomialNB()
mlt.fit(vec.transform(words),y_train)
datas = mlt.predict(vec.transform(z_words))
print(datas)
print(type(datas))



# np.set_printoptions(suppress=True)
np.savetxt("CCF_out.txt",datas,fmt=['%.g'],newline='\n')
# datas.to_csv('d://file.csv')
# file = open('d://CCF_out.txt','w')
# file.writable(str(datas_variable))
print("保存成功！")
# print("准确率为：",mlt.score(vec.transform(test_words),y_test))

#模型保存
# os.chdir("d://")
# joblib.dump(datas,"out.txt")
# print("模型成功保存了")