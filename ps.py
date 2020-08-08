import pandas as pd
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import joblib

##########데이터 로드

df = pd.read_csv('https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt', delimiter='\t', keep_default_na=False)

labels = ['negative', 'positive']

##########데이터 분석

##########데이터 전처리

x_data = df['document'].to_numpy()
y_data = df['label'].to_numpy()

for i, document in enumerate(x_data):
    document = BeautifulSoup(document, "html.parser").text #HTML 태그 제거
    okt = Okt()
    clean_words = okt.nouns(document) #어간 추출
    #print(clean_words) #['봄', '신제품', '소식']
    document = ' '.join(clean_words)
    x_data[i] = document
#print(x_data[:2]) #['봄 신제품 소식', '쿠폰 선물 무료 배송', '데 백화점 일', '파격 일 오늘 할인', '인기 제품 기간 한정 일', '오늘 일정 확인', '프로젝트 진행 상황 보고', '계약', '회의 일정 등록', '오늘 일정'] 

vectorizer = TfidfVectorizer()
vectorizer.fit(x_data)
x_data = vectorizer.transform(x_data)
print(x_data[:2]) #

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=777, stratify=y_data)

joblib.dump(vectorizer, 'model/naver_movie_review_positive_classification_model_vectorizer.pkl')

##########모델 생성

model = MultinomialNB(alpha=1.0)

##########모델 학습

model.fit(x_train, y_train)

##########모델 검증

print(model.score(x_train, y_train)) #1.0

print(model.score(x_test, y_test)) #1.0

joblib.dump(model, 'model/naver_movie_review_positive_classification_model.pkl')

##########모델 예측

document = '재미있다'
document = BeautifulSoup(document, "html.parser").text #HTML 태그 제거
okt = Okt()
clean_words = okt.nouns(document) #어간 추출
#print(clean_words) #['봄', '신제품', '소식']
document = ' '.join(clean_words)
print(document) #spam ?

x_test = [document]
x_test = vectorizer.transform(x_test)
print(x_test) #

y_predict = model.predict(x_test)
print(labels[y_predict[0]]) #ham