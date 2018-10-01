# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 03:22:13 2018

@author: HOSHIN CHO
"""

from bs4 import BeautifulSoup
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen

###############################################################################
#### 웹툰의 종류(에피소드,옴니버스,스토리) 별로 이름, 고유번호, 평점 df에 저장 ####
############################################################################### 

types=['episode','omnibus','story']

episode_df=pd.DataFrame(columns=("value","name","artist","ratings","episode","omnibus","story"))
omnibus_df=pd.DataFrame(columns=("value","name","artist","ratings","episode","omnibus","story"))
story_df=pd.DataFrame(columns=("value","name","artist","ratings","episode","omnibus","story"))

#3개의 종류('episode','omnibus','story')를 차례로 링크에 넣어서 각 페이지에서 고유번호, 작품명, 작가, 별점 크롤링
for t in types:
    genre_url='http://comic.naver.com/webtoon/genre.nhn?view=list&order=ViewCount&genre=%s'%t
    genre_html=urlopen(genre_url)
    genre_soup=BeautifulSoup(genre_html,'html.parser')
    
    get_name=genre_soup.select('td.subject > a > strong')
    get_artist=genre_soup.select('td > a') #작품명, 작가 짝으로 불러옴 (리스트의 홀수 인덱스 = 작가이름)
    get_ratings=genre_soup.select('td > div > strong')
    
    # 파싱 : 불러온 제목,별점, 별점참여수에서 태그 없애기
    name=[];artist=[];ratings=[];value=[]
    for i in range(len(get_name)):
        name.append(get_name[i].text)
        ratings.append(get_ratings[i].text)
    
    #작가이름 파싱
    for j in range(len(get_artist)):
        if j%2==1:
            value.append(get_artist[j].get('onclick').split('\',\'')[1])
            artist.append(get_artist[j].text)
    
    #각 데이터를 해당하는 type_df에 저장
    for i in range(len(get_name)):
        if t=='episode':
            episode_df.loc[i]=[value[i],name[i],artist[i],ratings[i],1,0,0]
        elif t=='omnibus':
            omnibus_df.loc[i]=[value[i],name[i],artist[i],ratings[i],0,1,0]
        elif t=='story':
            story_df.loc[i]=[value[i],name[i],artist[i],ratings[i],0,0,1]
         
#3개 타입의 df 합치기            
type_df=pd.DataFrame(columns=("value","name","artist","ratings","episode","omnibus","story"))
frames=[episode_df,omnibus_df,story_df]
type_df=pd.concat(frames, ignore_index=True)
type_df=type_df.sort_values(by=['value'])
type_df.to_csv(r"C:\Users\HOSHIN CHO\Desktop\result_df\type_df.csv")

#df.drop_duplicates() 중복된행제거

# 중복된 행 처리
#zeros=pd.Series(zero,index=genres)
zero=[0,0,0]
for a in type_df[:]['value']:
    print('-------------------------------')
    dup=type_df[type_df['value']==a].index.values       # dup = a 라는 고유번호를 가진 모든 index
    for b in dup:                                       # 같은 고유번호 가진 row의 genre boolean값 합치기 (ex. 1,0,1)
        print('b=',b)    
        t=type_df.iloc[b][4:].tolist()                  #해당 고유번호의 genre변수값 리스트로 저장
        zero=[x+y for x,y in zip(zero,t)]               #리스트 원소끼리의 합
        if b==dup[-1]:
            for b in dup:
                type_df.loc[b:b,'episode':'story']=zero #더해진 합을 genre_df에 입력 및 나머지 중복된 row의 장르변수값 0으로 초기화
                zero=[0,0,0]                            #리스트 초기화
                

to_drop=[]                                              
for i in range(725):                                    #중복된 열 중 genre 변수값이 모두 0인 index 저장
   if type_df.iloc[i][4:].tolist()==zero:              
       to_drop.append(i)

type_df=type_df.drop(type_df.index[to_drop])         #중복된 열 중 genre 변수값이 모두 0인 index 삭제

#중복된 행 없음

#export
type_df=type_df.sort_values(by=['name'])
type_df.to_csv(r"C:\Users\HOSHIN CHO\Desktop\result_df\type_df.csv")


###############################################################################
#####웹툰의 장르(에피소드,옴니버스,스토리)별로 이름, 고유번호, 평점을 df에 저장####
###############################################################################      

genres=['daily','comic','fantasy','action','drama','pure','sensibility','thrill','historical','sports']

daily_df=pd.DataFrame(columns=('value','name','artist','ratings','daily','comic','fantasy','action','drama','pure','sensibility','thrill','historical','sports'))
comic_df=pd.DataFrame(columns=('value','name','artist','ratings','daily','comic','fantasy','action','drama','pure','sensibility','thrill','historical','sports'))
fantasy_df=pd.DataFrame(columns=('value','name','artist','ratings','daily','comic','fantasy','action','drama','pure','sensibility','thrill','historical','sports'))
action_df=pd.DataFrame(columns=('value','name','artist','ratings','daily','comic','fantasy','action','drama','pure','sensibility','thrill','historical','sports'))
drama_df=pd.DataFrame(columns=('value','name','artist','ratings','daily','comic','fantasy','action','drama','pure','sensibility','thrill','historical','sports'))
pure_df=pd.DataFrame(columns=('value','name','artist','ratings','daily','comic','fantasy','action','drama','pure','sensibility','thrill','historical','sports'))
sensibility_df=pd.DataFrame(columns=('value','name','artist','ratings','daily','comic','fantasy','action','drama','pure','sensibility','thrill','historical','sports'))
thrill_df=pd.DataFrame(columns=('value','name','artist','ratings','daily','comic','fantasy','action','drama','pure','sensibility','thrill','historical','sports'))
historical_df=pd.DataFrame(columns=('value','name','artist','ratings','daily','comic','fantasy','action','drama','pure','sensibility','thrill','historical','sports'))  
sports_df=pd.DataFrame(columns=('value','name','artist','ratings','daily','comic','fantasy','action','drama','pure','sensibility','thrill','historical','sports'))
            
#10개의 장르를 차례로 링크에 넣어서 각 페이지에서 고유번호, 작품명, 작가, 별점 크롤링
for g in genres:
    genre_url='http://comic.naver.com/webtoon/genre.nhn?view=list&order=ViewCount&genre=%s'%g
    genre_html=urlopen(genre_url)
    genre_soup=BeautifulSoup(genre_html,'html.parser')
    
    #CSS selector를 사용하여 가져오기 (단, tag와 값이 같이 반환됨 ex)<h3>1141. 코디<h3>)
    get_name=genre_soup.select('td.subject > a > strong')
    get_artist=genre_soup.select('td > a') #작품명, 작가 짝으로 불러옴 (리스트의 홀수 인덱스 = 작가이름)
    get_ratings=genre_soup.select('td > div > strong')
    
    # 파싱 : 불러온 제목,별점, 별점참여수에서 태그 없애기
    name=[];artist=[];ratings=[];value=[]
    for i in range(len(get_name)):
        name.append(get_name[i].text)
        ratings.append(get_ratings[i].text)
    
    #작가이름 파싱
    for j in range(len(get_artist)):
        if j%2==1:
            value.append(get_artist[j].get('onclick').split('\',\'')[1])
            artist.append(get_artist[j].text)
    
    #각 데이터를 해당하는 장르_df에 저장
    for i in range(len(get_name)):
        if g==genres[0]:
            daily_df.loc[i]=[value[i],name[i],artist[i],ratings[i],1,0,0,0,0,0,0,0,0,0]
        elif g==genres[1]:
            comic_df.loc[i]=[value[i],name[i],artist[i],ratings[i],0,1,0,0,0,0,0,0,0,0]
        elif g==genres[2]:
            fantasy_df.loc[i]=[value[i],name[i],artist[i],ratings[i],0,0,1,0,0,0,0,0,0,0]
        elif g==genres[3]:
            action_df.loc[i]=[value[i],name[i],artist[i],ratings[i],0,0,0,1,0,0,0,0,0,0]
        elif g==genres[4]:
            drama_df.loc[i]=[value[i],name[i],artist[i],ratings[i],0,0,0,0,1,0,0,0,0,0]
        elif g==genres[5]:
            pure_df.loc[i]=[value[i],name[i],artist[i],ratings[i],0,0,0,0,0,1,0,0,0,0]
        elif g==genres[6]:
            sensibility_df.loc[i]=[value[i],name[i],artist[i],ratings[i],0,0,0,0,0,0,1,0,0,0]
        elif g==genres[7]:
            thrill_df.loc[i]=[value[i],name[i],artist[i],ratings[i],0,0,0,0,0,0,0,1,0,0]
        elif g==genres[8]:
            historical_df.loc[i]=[value[i],name[i],artist[i],ratings[i],0,0,0,0,0,0,0,0,1,0]
        elif g==genres[9]:
            sports_df.loc[i]=[value[i],name[i],artist[i],ratings[i],0,0,0,0,0,0,0,0,0,1]

#9개 장르의 df 합치기            
genre_df=pd.DataFrame(columns=('value','name','artist','ratings','daily','comic','fantasy','action','drama','pure','sensibility','thrill','historical','sports'))
frames=[daily_df,comic_df,fantasy_df,action_df,drama_df,pure_df,sensibility_df,thrill_df,historical_df,sports_df]
genre_df=pd.concat(frames, ignore_index=True)
genre_df.head




#df.drop_duplicates() 중복된행제거

# 중복된 행 처리
#zeros=pd.Series(zero,index=genres)
zero=[0,0,0,0,0,0,0,0,0,0]
for a in genre_df[:]['value']:
    print('-------------------------------')
    dup=genre_df[genre_df['value']==a].index.values     # dup = a 라는 고유번호를 가진 모든 index
    for b in dup:                                       # 같은 고유번호 가진 row의 genre boolean값 합치기 (ex. 1,0,0,0,1,0,0,0,0,1)
        print('b=',b)    
        t=genre_df.iloc[b][4:].tolist()                 #해당 고유번호의 genre변수값 리스트로 저장
        zero=[x+y for x,y in zip(zero,t)]               #리스트 원소끼리의 합
        if b==dup[-1]:
            for b in dup: 
                genre_df.loc[b:b,'daily':'sports']=zero #더해진 합을 genre_df에 입력 및 나머지 중복된 row의 장르변수값 0으로 초기화
                zero=[0,0,0,0,0,0,0,0,0,0]              #리스트 초기화
                

to_drop=[]                                              
for i in range(1156):                                   #중복된 열 중 genre 변수값이 모두 0인 index 저장
   if genre_df.iloc[i][4:].tolist()==zero:              
       to_drop.append(i)
         
genre_df=genre_df.drop(genre_df.index[to_drop])         #중복된 열 중 genre 변수값이 모두 0인 index 삭제

#export
genre_df=genre_df.sort_values(by=['name'])
genre_df.to_csv(r"C:\Users\HOSHIN CHO\Desktop\result_df\genre_df.csv")

##############################################################################
########################## type_df, genre_df합치기 ##########################
##############################################################################

var=['value','name','artist','ratings',
                      'episode','omnibus','story',
                      'daily','comic','fantasy',
                      'action','drama','pure','sensibility',
                      'thrill','historical','sports']
final_df=pd.DataFrame(columns=(var))
final_df=pd.concat([type_df,genre_df],axis=0,ignore_index=True,keys=['t','g'])


final_df=final_df[var]
final_df=final_df.fillna(0)

zero=[0,0,0,0,0,0,0,0,0,0,0,0,0]
for a in final_df[:]['value']:
    print('-------------------------------')
    dup=final_df[final_df['value']==a].index.values     # dup = a 라는 고유번호를 가진 모든 index
    for b in dup:                                       # 같은 고유번호 가진 row의 genre boolean값 합치기 (ex. 1,0,0,0,1,0,0,0,0,1)
        print('b=',b)    
        t=final_df.iloc[b][4:].tolist()                 #해당 고유번호의 genre변수값 리스트로 저장
        zero=[x+y for x,y in zip(zero,t)]               #리스트 원소끼리의 합
        if b==dup[-1]:
            for b in dup: 
                final_df.loc[b:b,'episode':'sports']=zero #더해진 합을 genre_df에 입력 및 나머지 중복된 row의 장르변수값 0으로 초기화
                zero=[0,0,0,0,0,0,0,0,0,0,0,0,0]   

to_drop=[]                                              
for i in range(len(final_df)):                          #중복된 열 중 genre 변수값이 모두 0인 index 저장
   if final_df.iloc[i][4:].tolist()==zero:              
       to_drop.append(i)

final_df=final_df.drop(final_df.index[to_drop])         #중복된 열 중 genre 변수값이 모두 0인 index 삭제

#export                       
final_df=final_df.sort_values(by=['name'])
final_df.to_csv(r"C:\Users\HOSHIN CHO\Desktop\result_df\final_df.csv")






















