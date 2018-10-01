
# coding: utf-8

# In[3]:


import os
import pandas as pd
import numpy as np
# -*- coding: utf-8 -*-


# # train_validation set 정리

# In[9]:

#training data 갖고오기
os.chdir(r"C:\Users\SAMSUNG\Downloads")
train_df = pd.read_csv("refinal_modeling_df.csv", encoding='utf-8', header=1 , engine='python')
train_df
train_df.head()


# In[10]:

col=train_df.columns
col


# In[11]:

# column name 변경
train_df.rename(columns={col[0]:"label", col[1]:"ratings", col[2]:"clicks",
                         col[19]:"MOVIE", col[20]:"DRAMA",col[21]:"BOOK" 
                         },inplace=True)

train_df.rename(columns={col[0]:"label"},inplace=True)
train_df.head()


# In[12]:

# clicks 정규화 by using min-max scaler
from sklearn.preprocessing import MinMaxScaler
minmax_scaler = MinMaxScaler().fit(train_df[['ratings']])
minmax_mat= minmax_scaler.transform(train_df[['ratings']])
minmax_mat

# clicks정규화 값 새로운 column 으로 추가
train_df['N_ratings'] = minmax_mat[:,0:1]
train_df.columns


# In[13]:

# pieces 역시 정규화
minmax_scaler2 = MinMaxScaler().fit(train_df[['pieces']])
minmax_mat2= minmax_scaler2.transform(train_df[['pieces']])
minmax_mat2

train_df['N_pieces'] = minmax_mat2[:,0:1]
train_df.columns


# In[14]:

# OSMU 여부를 종합적으로 판단하기 위해 Y변수(영화, 드라마) 하나로 합치기
df1= train_df["MOVIE"]
df2= train_df["DRAMA"]
#df3= train_df["BOOK"]
#df4= df1+df2+df3
# BOOK빼고 movie, drama로 y변수 재정비
df4= df1+df2

# Dataframe에 Column 추가
train_df["SUM"]= df4
train_df.head()

train_df["OSMU"]=np.where(train_df["SUM"]>0, 1,0)
train_df.columns


# In[15]:

#Column 순서 변경
train_df= train_df[['label','finish','pieces', 'ratings', 'clicks', 'N_pieces','N_ratings', 'N_clicks',
       'episode', 'omnibus', 'story', 'daily', 'comic', 'fantasy', 'action',
       'drama', 'pure', 'sensibility', 'thrill', 'historical', 'sports',
       'MOVIE', 'DRAMA', 'BOOK', 'SUM', 'OSMU']]


# In[16]:

# train_x 정리
all_x = train_df[['N_pieces','N_ratings', 'N_clicks',
       'episode', 'omnibus', 'story', 'daily', 'comic', 'fantasy', 'action',
       'drama', 'pure', 'sensibility', 'thrill', 'historical', 'sports','BOOK']]

# 나중에 변수 중요도 할 때 쓸 것
all_x2 = train_df[['N_pieces','N_ratings', 'N_clicks',
       'episode', 'omnibus', 'story', 'daily', 'comic', 'fantasy', 'action',
       'drama', 'pure', 'sensibility', 'thrill', 'historical', 'sports','BOOK']]

all_x = np.array(all_x.iloc[0:439,:])
all_x

# train_y정리
all_y = np.array(train_df['OSMU'][0:439])


# In[17]:

all_x


# In[18]:

all_y


# In[19]:

# Data split
# train_test_split()함수를 이용하여 training data를 training set과 test set으로 분할
# 완결웹툰을 Training data로
from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(all_x,
                                                    all_y, 
                                                  test_size=0.3, random_state=0)

print("X_train 크기: ", X_train.shape)
print("y_train 크기: ", y_train.shape)
print("X_test 크기: ", X_test.shape)
print("y_test 크기: ", y_test.shape)


# # model 비교

# In[20]:

#Model1. Decision Tree

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_train, y_train)

print("훈련 데이터 결과: ", tree.score(X_train, y_train))
print("검증 데이터 결과: ", tree.score(X_test, y_test))
# Tree Pruning
tree2 = DecisionTreeClassifier(max_depth = 4, random_state=0)
tree2.fit(X_train, y_train)

print("훈련 데이터 결과(d=4): ", tree2.score(X_train, y_train))
print("검증 데이터 결과(d=4): ", tree2.score(X_test, y_test))


# In[14]:

# 1-1. DT n_neighbors for문 돌리기
############### n=4 가 최적 ################
accuracy = []

max_depth_settings = range(1,10)
for i in max_depth_settings:
    tree_i = DecisionTreeClassifier(max_depth = i, random_state=0)
    tree_i.fit(X_train, y_train)
    accuracy.append([i, tree_i.score(X_train, y_train), tree_i.score(X_test, y_test)])

  
results = pd.DataFrame(accuracy, columns=['n','training accuracy','test accuracy'])
results.head(10)


# In[21]:

# 1-2. 트리를 만드는 결정에 각 특성이 얼마나 중요한지를 평가하는 특성 중요도

print("특성 중요도:\n{}".format(tree.feature_importances_))


# In[22]:

# Model2. Random Forest
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators=2,random_state=0)
forest.fit(X_train, y_train)

print("훈련 데이터 결과(e=2): ", forest.score(X_train, y_train))
print("검증 데이터 결과(e=2): ", forest.score(X_test, y_test))

forest2 = RandomForestClassifier(n_estimators=5, random_state=0)
forest2.fit(X_train, y_train)

print("훈련 데이터 결과(e=5): ", forest2.score(X_train, y_train))
print("검증 데이터 결과(e=5): ", forest2.score(X_test, y_test))


# In[23]:

# 2-1. RF n_neighbors for문 돌리기
############### n=2 가 최적 ################
accuracy1 = []

n_estimators_settings = range(1,10)
for i in n_estimators_settings:
    forest_i = RandomForestClassifier(n_estimators=i,random_state=0)
    forest_i.fit(X_train, y_train)
    accuracy1.append([i, forest_i.score(X_train, y_train), forest_i.score(X_test, y_test)])

  
results = pd.DataFrame(accuracy1, columns=['n','training accuracy','test accuracy'])
results.head(10)


# In[24]:

# 2-2. Feature Selection with sklearn
# 모델 기반 선택/ 변수의 중요도 계산
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

select = SelectFromModel(RandomForestRegressor(n_estimators=2, random_state=0),
                         threshold="median")
select.fit(X_train,y_train)

select. get_support()

selected_idx=np.where(select.get_support()==True)
selected_idx

all_x2.columns[selected_idx]
# 변수의 중요도(영향력)가 높은 것으로 선택된 변수들은 다음과 같다.


# In[19]:

######### RF 변수 중요도 ##########
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

f_names = all_x2.columns[:]
f_importances = forest.feature_importances_

for name, importance in zip(f_names, f_importances):
    print(name, importance)

s_importances = pd.Series(f_importances, index=f_names)
s_importances.sort_values(ascending=False)

f_df = pd.DataFrame(s_importances)
f_df.to_csv(r'C:\DataScience\webtoon_data\test\f_importance.txt')


# In[25]:

# 2-3. 모델 기반 선택/ 정확도 비교
rf=RandomForestRegressor(n_estimators=2, random_state=0)
rf.fit(X_train, y_train)
print("전체 변수 사용:",rf.score(X_test,y_test))

X_train_selected=select.transform(X_train)
X_test_selected= select.transform(X_test)

rf.fit(X_train_selected,y_train)
print("선택 변수 사용:",rf.score(X_test_selected,y_test))


# In[26]:

# Modle3. knn
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 10, metric='manhattan')
knn.fit(X_train, y_train)

print("훈련 데이터 결과: ", knn.score(X_train, y_train))
print("검증 데이터 결과: ", knn.score(X_test, y_test))


# In[22]:

# 3-1. knn n_neighbors for문 돌리기
############### n=4 가 최적 ################
accuracy2 = []

neighbors_settings = [1,3,10,30,50]
for i in neighbors_settings:
    knn_i = KNeighborsClassifier(n_neighbors = i, metric='manhattan')
    knn_i.fit(X_train, y_train)
    accuracy2.append([i, knn_i.score(X_train, y_train), knn_i.score(X_test, y_test)])

  
results = pd.DataFrame(accuracy, columns=['n','training accuracy','test accuracy'])
results


# In[27]:

# Cross validation(K-fold) 교차검증 진행
# K-fold CV with Python for classification

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
k_fold=KFold(n_splits=20,shuffle=True, random_state=0)


# In[30]:

# 1. K-fold(교차검증) & Decision Tree
dt = DecisionTreeClassifier(max_depth =4, random_state=0)
scoring='accuracy'
score = cross_val_score(dt,all_x, all_y, 
                                cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[31]:

# 교차검증 결과 DT의 평균 정확도는 88.14 (max_depth=4)
np.mean(score)
round(np.mean(score)*100,2)


# In[32]:

# 2. K-fold(교차검증) & Random Forest
rf= RandomForestClassifier(n_estimators=2) #2개의 decision tree 사용
rf


# In[33]:

scoring='accuracy'
score= cross_val_score(rf,all_x,all_y, 
                       cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[34]:

# 교차검증 결과 RF의 평균 정확도는 89.96 (n_estimators=2)
round(np.mean(score)*100,2)


# In[35]:

# 3.K-fold(교차검증) & KNN

knn=KNeighborsClassifier(n_neighbors=2)
knn


# In[36]:

scoring='accuracy'
score= cross_val_score(knn,all_x,all_y, 
                       cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# In[37]:

# 교차검증 결과 KNN의 평균 정확도는 90.88 (n=10)
round(np.mean(score)*100,2)


# In[66]:

################## test set #################
# x test data set
all_x_test = np.array(all_x2.iloc[439:,:])


# y label set
all_y_test = np.array(train_df['OSMU'][439:])


# In[67]:

################## predict ##################

y_pred = forest.predict(all_x_test)

from sklearn.metrics import accuracy_score

print("테스트 세트에 대한 예측값  \n {}".format(y_pred))
print("테스트 세트의 정확도 : {:.2f}".format(forest.score(all_x_test, all_y_test)))

import collections
print(collections.Counter(y_pred))
# 0: 미완결 ; 1: 완결



# In[68]:

y_index = np.where(y_pred == 1)
print(y_index)

print(train_df['label'][459])
print(train_df['OSMU'][459])

print(train_df['label'][493])
print(train_df['OSMU'][493])

print(train_df['label'][511])
print(train_df['OSMU'][511])

print(train_df['label'][528])
print(train_df['OSMU'][528])


# In[69]:

all_y_test = np.where(all_y_test ==1)
print(all_y_test)


# In[70]:



print(train_df['label'][466])
print(train_df['OSMU'][466])

print(train_df['label'][476])
print(train_df['OSMU'][476])

print(train_df['label'][513])
print(train_df['OSMU'][513])

print(train_df['label'][516])
print(train_df['OSMU'][516])

