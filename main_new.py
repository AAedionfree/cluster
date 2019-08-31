#!/usr/bin/env python
# coding: utf-8

# ### dependence ###

# In[1]:


from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import hierarchical
from sklearn.cluster import estimate_bandwidth
from sklearn.metrics import calinski_harabaz_score
from sklearn.metrics import silhouette_score
from jieba.analyse import *
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
# from Bio.Cluster import kcluster
# import synonyms
import pandas as pd
import jieba
jieba.add_word("报警人", freq=56, tag='nr')
jieba.add_word("系", freq=56, tag='r')
from math import isnan
import math
import numpy as np
import re
import csv
import jieba.posseg as pseg
import gensim


# ### main function ###

# In[2]:


n_clusters = 24

def deal_sklearn_model(model,
                       weight,
                       data,
                       df,
                       X,
                       print_details=0,
                       distance='euclidean'):
    kind = df[1].tolist()
    if print_details:
        print(model.labels_)
        print(len(model.labels_))
        details = {}
        for i in range(len(data)):
            if model.labels_[i] not in details:
                details[model.labels_[i]] = [[data[i]], [kind[i]]]
            else:
                details[model.labels_[i]][0].append(data[i])
                details[model.labels_[i]][1].append(kind[i])
        number = 0
        for key, list in details.items():
            this_kind = 0
            print("第%d类" % key)
            for i in range(len(list[0])):
                this_kind = this_kind + 1
                print("属于", list[1][i], ' ', list[0][i])
            print(this_kind)
            print()
            print()
            print()
            number += this_kind
        print("all_number == ", number)
    check_result(weight, model, distance)


    # K-Means 算法
def K_Means(weight, data, df, X, print_details=0):
    print("K_Means算法")
    K_number = 16
    iter_number = 700
    kmodel = KMeans(n_clusters=n_clusters,
                    n_init=10,
                    max_iter=iter_number,
                    init='k-means++',
                   n_jobs=-1).fit(weight)
    deal_sklearn_model(kmodel, weight, data, df, X, print_details)
    return kmodel.labels_


# DBSCAN 算法
def AAedion_DBSCAN(weight, data, df, X, print_details=0, distance='euclidean'):
    print("DBSCAN算法")
    DBmodel = DBSCAN(eps=0.5, min_samples=10, metric=distance,n_jobs=-1).fit(weight)
    deal_sklearn_model(DBmodel, weight, data, df, X, print_details, distance)
    print("DBSCAN算法完成")
    return DBmodel.labels_


def AAedion_Birch(weight, data, df, X, print_details=0, distance='euclidean'):
    print('Birch算法')
    Birch_model = Birch(n_clusters=n_clusters).fit(weight)
    deal_sklearn_model(
        Birch_model,
        weight,
        data,
        df,
        X,
        print_details,
    )
    return Birch_model.labels_


def AAedion_SpectralClustering(weight,
                               data,
                               df,
                               X,
                               print_details=0,
                               distance='euclidean',n_jobs=-1):
    print('SpectralClustering算法')
    Spe_model = SpectralClustering(n_clusters=n_clusters).fit(weight)
    deal_sklearn_model(
        Spe_model,
        weight,
        data,
        df,
        X,
        print_details,
    )
    return Spe_model.labels_


def AAedion_AffinityPropagation(weight,
                                data,
                                df,
                                X,
                                print_details=0,
                                distance='euclidean'):
    print('AffinityPropagatio算法')
    Aff_model = AffinityPropagation(damping=0.5,
                                    max_iter=200,
                                    convergence_iter=15,
                                    copy=True,
                                    preference=None,
                                    affinity='euclidean',
                                    verbose=False).fit(weight)
    deal_sklearn_model(
        Aff_model,
        weight,
        data,
        df,
        X,
        print_details,
    )
    return Aff_model.labels_


def AAedion_Meanshift(weight,
                      data,
                      df,
                      X,
                      print_details=0,
                      distance='euclidean'):
    print('MeanShift算法')
#     bandwidth = estimate_bandwidth(weight, quantile=0.3,n_jobs=-1)
#    print(bandwidth)
    MeanShift_model = MeanShift(bandwidth=1,
                                seeds=None,
                                bin_seeding=True,
                                min_bin_freq=1,
                                cluster_all=False,
                                n_jobs=-1).fit(weight)
    print('Meanshfit finish')
    deal_sklearn_model(
        MeanShift_model,
        weight,
        data,
        df,
        X,
        print_details,
    )
    return MeanShift_model.labels_


def check_result(weight, model, distance):
    #print("共%d类" %model.labels_)
    print("result of harabaz_score: ",
          calinski_harabaz_score(weight, model.labels_))
    print("result of silhouette_score: ",
          silhouette_score(weight, model.labels_, metric='euclidean'))


def print_TF_IDF_values():
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重
        print("-------这里输出第", i, u"类文本的词语tf-idf权重------")
        # for j in range(len(word)):
        print(list(zip(all_word, weight[i])))


# ### stop word ###

# In[3]:


# 人工停用词表
special_word = ['称', '，', '。', '在', '年', '月', '(', '（','民警','报警','观音桥','称有','称一','称车','称其','接警','人称']

# 加载停用词表
Stopword_FilePath = "../stopWord.txt"
stopWordList = pd.read_csv(Stopword_FilePath, sep="\r")
for word in special_word:
    stopWordList.loc[stopWordList.size] = word
print("停用词表的大小是", stopWordList.size)

# 加载数据
FilePath = "../data.csv"
df = pd.read_csv(FilePath,header=None)

# 停用词过滤
data = df[5]
kind = df[2]
X = []
delete = []
record = {}
person_number = 0
place_number = 0
pronoun_number = 0
beside_number = 0
all_number = 0
flag_record = {}
flag_form = {}
flag_result = {}
data_keywords = []
#     data_new = []
#     kind_new = []
#     k = 0
#     for i in range(len(data)):
#         try:
#             if isnan(float(data[i])):
#                 k += 1
#         except ValueError:    
#              data_new.append(data[i])
#              kind_new.append(kind[i])
#     data = data_new
#     kind = kind_new
for i in range(len(data)):
    cut_result = []
    person_name = []
    place_name = []
    pronoun = []
    beside = []
    # 按词性过滤
    #print(data[i])
    k = pseg.cut(data[i])
    for word, flag in k:
        flag_form[word] = flag
        if flag not in flag_record:
            flag_record[flag] = 1
        else:
            flag_record[flag] += 1
            
        if flag == 'nr' or flag == 'nr1' or flag == 'nr2':
            person_name.append(word)
            person_number += 1
            continue
        if flag == 'ns' or flag == 'nz':
            place_name.append(word)
            place_number += 1
            continue
        if flag == 'r' or flag == 'rr' or flag == 'rz':
            pronoun.append(word)
            pronoun_number += 1
            continue
        if flag == 'w' or flag == 'h' or flag == 'k' or flag == 'xx'                 or flag == 'o' or flag == 'u' or flag == 'm' or flag == 'd'                 or flag == 'f':
            beside.append(word)
            beside_number += 1
            continue
    # 提取关键词列表
    for keyword, weight in extract_tags(data[i], withWeight=True):
        if keyword in person_name or keyword in place_name or keyword in pronoun or keyword in beside:
            all_number += 1
            delete.append(keyword)
            continue
        # if flag_form[keyword] not in flag_result:
        #     flag_result[flag_form[keyword]] = [keyword]
        # else:
        #     flag_result[flag_form[keyword]].append(keyword)
        cut_result.append(keyword)
    # print(data[i],": 关键词",cut_result)

    # 用停用词库筛选关键词列表
    temp = ""
    for j in cut_result:
        if j in stopWordList.values or re.search("[0-9]+", j) is not None                 or re.search("[a-z]+", j) is not None or re.search("[A-Z]+", j) is not None:
            if j not in delete:
                delete.append(j)
        else:
            if temp == '':
                temp = j
            else:
                temp = temp + " " + j
            if j in record:
                record[j] = record[j] + 1
            else:
                record[j] = 1
    X.append(temp)
    data_keywords.append(temp)
flag_record = sorted(flag_record.items(), key=lambda item: item[1], reverse=True)
result_X = []
for string in X:
    temp = ''
    for j in string.split(' '):
        if j == '':
            continue
        if record[j] < 5:
            continue
        temp = temp + " " + j
        # print(j)
    result_X.append(temp)
X_str = np.array(result_X)
record = sorted(record.items(), key=lambda item: item[1], reverse=True)
print('finish')
#     print("被清除的词有", delete)
#     print("最终词典", record)
#     print("过滤结果", X_str)
#     print('person_number: ', person_number)
#     print('place_number: ', place_number)
#     print('pronoun_number: ', pronoun_number)
#     print('beside_number: ', beside_number)
#     print('all_number: ', all_number)
#     print('flag_record: ', flag_record)

# for w, f in flag_result.items():
#     print('%s类' % w)
#     for str in f:
#         print(str)
#     print()
#     print()
#     print()


# ### construction TF-IDF ###

# #### synonym word replace ####

# In[4]:


wx_from_text = gensim.models.KeyedVectors.load_word2vec_format('E:/chrome_download/2000000tencent.txt')
model = wx_from_text.wv


# In[5]:


# preprocessing
data = result_X
synonym_map = {}
no_synonym_words_count = 0
for i in range(len(data)):
    for string in data[i].split(' '):
        if string not in synonym_map:
            try:
                synonym_list = model.most_similar(string)
                for item in synonym_list:
                    synonym_word = item[0]
                    if synonym_word not in synonym_map:
                        synonym_map[synonym_word] = string
#                         print('add ',synonym_word,'to ',string)
            except Exception as e:
                    no_synonym_words_count += 1
#                 print('no word ',string)
# replace
result = []
for i in range(len(data)):
    temp = ""
    for string in data[i].split(' '):
        if string in  synonym_map:
            replace_string = synonym_map[string]
            print(string," change to ",replace_string)
        else:
            replace_string = string
        if temp != "":
            temp = temp + " " + replace_string
        else:
            temp = replace_string
    result.append(temp)
X_str = np.array(result)


# #### construction TF-IDF based on synonym words ####

# In[6]:


vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
X = vectorizer.fit_transform(X_str)  # 将文本转为词频矩阵
tfidf = transformer.fit_transform(X)  # 计算tf-idf，
all_word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重

# print_TF_IDF_values()
print("weight shape:", weight.shape)
print("word shape:", len(all_word))


# ### Cluster ###

# #### cluster ####

# In[7]:


def clustering(print_details,weight,DB):
    KMeans_labels = K_Means(weight, data, df, X, print_details)
#     if DB == 1:
#         DS_labels = AAedion_DBSCAN(weight, data, df, X, print_details)
#     else:
#         DS_labels = AAedion_SpectralClustering(
#         weight, data, df, X, print_details)
    Birch_labels = AAedion_Birch(weight, data, df, X, print_details)
    all_labels = [
        KMeans_labels, #DS_labels, 
        Birch_labels
    ]
    #print(all_labels[0][1])
    return all_labels


# KMeans_labels = K_Means(weight, data, df, X, print_details)
# DBSCAN_labels = AAedion_DBSCAN(weight, data, df, X, print_details)
# Brich_labels = AAedion_Birch(weight, data, df, X, print_details)
# SpectralClustering_labels = AAedion_SpectralClustering(weight, data, df, X, print_details)

### can't use 

# AAedion_AffinityPropagation(weight, data, df, X, 1)
# AAedion_Meanshift(weight, data, df, X, 1)


# #### change weight ####

# In[8]:


def deal(all_labels):
    cluster_result = []
    for i in range(0, len(all_labels[0])):
        cluster_result.append('')
    for labels in all_labels:
        for i in range(0, len(labels)):
            cluster_result[i] = cluster_result[i] + " " + str(labels[i])
    return cluster_result


def Change(all_labels, weight):
    cluster_result = deal(all_labels)
    cluster_map = {}
    for i in range(len(cluster_result)):
        string = cluster_result[i]
        if string in cluster_map:
            cluster_map[string][0].append(data[i])
            cluster_map[string][1].append(i)
            cluster_map[string][2] += 1
        else:
            cluster_map[string] = [[data[i]], [i], 1]
    
    cluster_map = sorted(cluster_map.items(),
                         key=lambda item: item[1][2],
                         reverse=True)
    print("聚类总情况数: ", len(cluster_map))
    # for item in cluster_map:
    #     print("所属:  ", item[0], "数量： ", item[1][2])
    goal_words = {}
    for i in range(0, len(cluster_map) // 2):
        #         if cluster_map[i][1][2] < len(data)/40:
        #             continue
        print(i, cluster_map[i][1][2])
        wordcount_record = {}
        for item in cluster_map[i][1][1]:
            #     print(data_keywords[item])
            for word in data_keywords[item].split(" "):
                if word == ' ' or word == '':
                    continue
                if word in wordcount_record:
                    wordcount_record[word] += 1
                else:
                    wordcount_record[word] = 1
        wordcount_record = sorted(wordcount_record.items(),
                                  key=lambda item: item[1],
                                  reverse=True)
        # increase the first and second word
        alpha = 1.5
        for item in wordcount_record[0:2]:
            increase_rate = 1 + alpha * (item[1] / cluster_map[i][1][2])
            print(item[0], " ", item[1]," increase rate: ",increase_rate)
            if item[0] not in goal_words:
                weight = changeWeight(weight, all_word, item[0],
                                      1 + alpha * (item[1] / cluster_map[i][1][2]),
                                      0)
                goal_words[item[0]] = 1
    return weight

def Change_Chi(all_labels, weight):
    types = deal(all_labels) #类list
    corpus = data #数据类
    words = all_word #特征词
    
    cluster_result = types
    cluster_map = {}
    for i in range(len(cluster_result)):
        string = cluster_result[i]
        if string in cluster_map:
            cluster_map[string][0].append(data[i])
            cluster_map[string][1].append(i)
            cluster_map[string][2] += 1
        else:
            cluster_map[string] = [[data[i]], [i], 1]
    
    cluster_map = sorted(cluster_map.items(),
                         key=lambda item: item[1][2],
                         reverse=True)
    print("聚类总情况数: ", len(cluster_map))
    print("最大类的样本数为：", cluster_map[0][1][2])
    
    types_words_ratio = {}
    types_words_N = len(corpus)
    types_words_A = {} #属于某类别ci也含有特征词的文本数目
    types_words_B = {} #不属于某类别ci也含有特征词的文本数目
    types_words_C = {} #属于某类别ci但不含有特征词的文本数目
    types_words_D = {} #不属于某类别ci也不含有特征词的文本数目
    
    for i in range(len(cluster_map)):
        t = cluster_map[i][0]
        for w in words:
            pair = t + w;
            for i in range(len(corpus)):
                if types[i] == t and w in corpus[i]:
                    types_words_A[pair] = types_words_A.get(pair, 0) + 1
                    continue
                if types[i] != t and w in corpus[i]:
                    types_words_B[pair] = types_words_B.get(pair, 0) + 1
                    continue
                if types[i] == t and w not in corpus[i]:
                    types_words_C[pair] = types_words_C.get(pair, 0) + 1
                    continue
                if types[i] != t and w not in corpus[i]:
                    types_words_D[pair] = types_words_D.get(pair, 0) + 1
            
    words_suit_types = {} #CHI值所对应的类
    words_suit_marks = {} #CHI值
    words_types_marks = {} #不同类的CHI排序
    for i in range(len(cluster_map)):
        words_types_marks[cluster_map[i][0]] = {}
        
    for word in words:
        for i in range(len(cluster_map)):
            t = cluster_map[i][0]
            pair = t + word
            CHI = types_words_N * math.pow((types_words_A.get(pair, 0) * types_words_D.get(pair, 0) - types_words_B.get(pair, 0) * types_words_C.get(pair, 0)), 2) 
            CHI = CHI / ((types_words_A.get(pair, 0) + types_words_C.get(pair, 0)) * (types_words_B.get(pair, 0) + types_words_D.get(pair, 0)))
            CHI = CHI / ((types_words_A.get(pair, 0) + types_words_B.get(pair, 0)) * (types_words_C.get(pair, 0) + types_words_D.get(pair, 0)))
            if CHI > words_suit_marks.get(word, 0):
                words_suit_marks[word] = CHI
                words_suit_types[word] = t
        types = words_suit_types[word]
        marks = words_suit_marks[word]
        words_types_marks[types][word] = marks
            
    for key in words_types_marks.keys():
        temp = words_types_marks[key]
        if len(temp) == 0:
            continue
        wordcount_record = sorted(temp.items(),
                                  key=lambda item: item[1],
                                  reverse=True)
        # increase the first and second word
        alpha = 0.005
        if len(wordcount_record) >= 2:
            for item in wordcount_record[0:2]:
                increase_rate = 1 + alpha * item[1] / cluster_map[i][1][2]
                print(item[0], " ", item[1]," increase rate: ",increase_rate)
                weight = changeWeight(weight, all_word, item[0], 1 + alpha * (item[1] / cluster_map[i][1][2]),0)
        else:
            item = wordcount_record[0]
            increase_rate = 1 + alpha * item[1] / cluster_map[i][1][2]
            print(item[0], " ", item[1]," increase rate: ",increase_rate)
            weight = changeWeight(weight, all_word, item[0], 1 + alpha * (item[1] / cluster_map[i][1][2]),0)
            
    return weight
      
    

def changeWeight(weight, all_word, word, change_rate, print_details=0):
    change_array = np.eye(weight.shape[1])
    for i in range(len(all_word)):
        if all_word[i] == word:
            break
    change_array[i][i] = change_rate
    weight = np.dot(weight, change_array)
    return weight


# #### training ####

# In[9]:


print_details = 0
DB = 0
for i in range(0, 7):
    print('第%d次' % (i + 1))
    all_labels = clustering(print_details, weight, DB)
    weight = Change_Chi(all_labels, weight)


# #### print result ####

# In[10]:


min_value = 0 
for i in range(30,40):
    kmodel = KMeans(n_clusters=i,
                    n_init=10,
                    max_iter=700,
                    init='k-means++',
                   n_jobs=-1).fit(weight)
    harabaz_value = calinski_harabaz_score(weight,kmodel.labels_)
    silhouette_value = silhouette_score(weight,kmodel.labels_)
    print('final value: ',harabaz_value," n:",i)
    if harabaz_value > min_value:
        min_value = harabaz_value
        labels_ = kmodel.labels_
    
details = {}
kind = df[2].tolist()
for i in range(len(data)):
    if labels_[i] not in details:
        details[labels_[i]] = [[data[i]], [kind[i]],[i]]
    else:
        details[labels_[i]][0].append(data[i])
        details[labels_[i]][1].append(kind[i])
        details[labels_[i]][2].append(i)

        
with open("输出.csv","w",encoding="utf-8",newline="") as datacsv:
#dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
    csvwriter = csv.writer(datacsv,dialect = ("excel"))
    #csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
    for key, list in details.items():
        for i in range(len(list[0])):
            csvwriter.writerow([key, list[1][i], list[0][i]])
        


# In[ ]:


print()
for key,value in details.items():
    if key == 1:
        break
word_count = {}
for item in value[2]:
    for string in data_keywords[item].split(" "):
        if string in word_count:
            word_count[string] += 1
        else:
            word_count[string] = 1
word_count_list = sorted(word_count.items(),
                         key=lambda item: item[1],
                         reverse=True)
for i in range(len(word_count_list)):
    print(word_count_list[i])

