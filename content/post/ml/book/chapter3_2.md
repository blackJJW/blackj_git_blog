---
title: "[ML][Book][혼공 머신러닝+딥러닝] ch3-2 선형 회귀"
description: ""
date: "2022-04-22T16:27:45+09:00"
thumbnail: ""
categories:
  - "ML"
tags:
  - "Python"
  - "ML"

---
<!--more-->

## k-최근접 이웃의 한계


```python
import numpy as np

perch_length = np.array(
    [8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 
     21.0, 21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 
     22.5, 22.7, 23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 
     27.3, 27.5, 27.5, 27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 
     36.5, 36.0, 37.0, 37.0, 39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 
     40.0, 42.0, 43.0, 43.0, 43.5, 44.0]
     )
perch_weight = np.array(
    [5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0]
     )
```


```python
from sklearn.model_selection import train_test_split

# 훈련 세트와 테스트 세트로 나눔
train_input, test_input, train_target, test_target = train_test_split(perch_length, perch_weight, random_state = 42)

# 훈련 세트와 테스트 세트를 2차원 배열로 바꿈
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
```


```python
from sklearn.neighbors import KNeighborsRegressor

knr = KNeighborsRegressor(n_neighbors = 3)

knr.fit(train_input, train_target)
```




    KNeighborsRegressor(n_neighbors=3)




```python
# 이 모델을 사용해 길이가 50cm인 농어의 무게를 예측
print(knr.predict([[50]]))
```

    [1033.33333333]
    

- 결과가 실제와 다름


```python
import matplotlib.pyplot as plt

# 50cm 농어의 이웃을 구함
distances, indexes = knr.kneighbors([[50]])

# 훈련 세읕의 산점도
plt.scatter(train_input, train_target)

# 훈련 세트 중에서 이웃 샘플만 다시
plt.scatter(train_input[indexes], train_target[indexes], marker="D")

# 50cm 농어 데이터
plt.scatter(50, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter3_2/output_8_0.png)



```python
print(np.mean(train_target[indexes]))
```

    1033.3333333333333
    


```python
print(knr.predict([[100]]))
```

    [1033.33333333]
    


```python
# 100cm 농어의 이웃을 구함
distances, indexes = knr.kneighbors([[100]])

# 훈련 세읕의 산점도
plt.scatter(train_input, train_target)

# 훈련 세트 중에서 이웃 샘플만 다시
plt.scatter(train_input[indexes], train_target[indexes], marker="D")

# 100cm 농어 데이터
plt.scatter(100, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter3_2/output_11_0.png)


- 농어가 아무리 커도 무게가 더 늘어나지 않는다. 

## 선형 회귀(linear regression)
- sklearn.linear_model 패키지 아래에 LinearRegression 클래스


```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

# 선형 회귀 모델 훈련
lr.fit(train_input, train_target)

# 50cm 농어 예측
print(lr.predict([[50]]))
```

    [1241.83860323]
    

- $y = ax + b$
  - LinearRegression 클래스가 찾은 a, b : lr객체의 coef_와 intercept_ 속성에 저장


```python
print(lr.coef_, lr.intercept_)
```

    [39.01714496] -709.0186449535477
    

> coef_ 와 intercept_
  - 머신러닝 알고리즘이 찾은 값이라는 의미로 모델 파라미터(model parameter)라고 부름
  - 머신러닝 알고리즘의 훈련 과정 -> 최적의 모델 파라미터를 찾는 것과 같음 -> 모델 기반 학습
  - k-최근접 이웃에는 모델 파라미터가 없음 -> 훈련 세트를 저장하는 것이 훈련의 전부 -> 사례 기반 학습



```python
# 농어의 길이 15~50 까지의 직선

# 훈련 세트의 산점도
plt.scatter(train_input, train_target)

# 15~50까지 1차 방정식
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])

# 50cm 농어 데이터
plt.scatter(50, 1241.8, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter3_2/output_18_0.png)



```python
# 훈련 세트와 테스트 세트에 대한 R^2 점수

print(lr.score(train_input, train_target)) # 훈련 세트
print(lr.score(test_input, test_target)) # 테스트 세트
```

    0.939846333997604
    0.8247503123313558
    

- 과소 적합됨

## 다항 회귀
- 2차 방정식을 그리려면 길이를 제곱한 향이 훈련 세트에 포함되어야 함


```python
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
```


```python
print(train_poly.shape, test_poly.shape)
```

    (42, 2) (14, 2)
    


```python
lr = LinearRegression()
lr.fit(train_poly, train_target)

print(lr.predict([[50 ** 2, 50]]))
```

    [1573.98423528]
    


```python
print(lr.coef_, lr.intercept_)
```

    [  1.01433211 -21.55792498] 116.0502107827827
    

- 이 모델을 다음과 같은 그래프를 학습
  - $무게 = 1.01 \times 길이^2 - 21.6 \times 길이 + 116.05$
  - 이런 방정식을 다항식(polynomial)이라 부르며 다항식을 사용한 선형 회귀를 다항 회귀(polynomial regression)


```python
# 구간별 직선을 그리기 위해 15~49 까지의 정수 배열 생성
point = np.arange(15, 50)

# 훈련 세트 산점도
plt.scatter(train_input, train_target)

# 15에서 49까지의 2차 방정식 그래프
plt.plot(point, 1.01 * point ** 2 - 21.6 * point + 116.05)

# 50cm 농어 데이터
plt.scatter(50, 1574, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter3_2/output_27_0.png)



```python
# 훈련 세트와 테스트 세트에 대한 R^2 점수

print(lr.score(train_poly, train_target)) # 훈련 세트
print(lr.score(test_poly, test_target)) # 테스트 세트
```

    0.9706807451768623
    0.9775935108325122
    

## 마무리

### 키워드로 끝내는 핵심 포인트
- 선형 회귀 : 특성과 타깃 사이의 관계를 가장 잘 나타내는 선형 방정식을 찾음. 특성이 하나면 직선 방정식
  - 선형 회귀가 찾은 특성과 타깃 사이의 관계믐 선형 방정식의 계수 또는 가중치에 저장
  - 머신러닝에서 종종 가중치는 방정식의 기울기와 절편을 모두 의미하는 경우가 많음

- 모델 파라미터 : 선형 회귀가 찾은 가중치처럼 머신러닝 모델이 특성에서 학습한 파라미터
- 다항 회귀 : 다항식을 사용하여 특성과 타깃 사이의 관계를 나타냄.
  - 이 함수는 비선형일 수 있지만 선형 회귀로 표현할 수 있음

### 핵심 패키지와 함수
> scikit-learn
  - LinearRegression : 사이킷런의 선형 회귀 클래스
  - fit_intercept 매개변수를 False로 지정하면 절편을 학습하지 않음. 기본값 = True
  - 학습된 모델의 coef_ 속성은 특성에 대한 계수를 포함한 배열. 즉 이 배열의 크기는 특성의 개수와 같음
  - intercept_ 속성에는 절편이 저장

# Book : '혼자 공부하는 머신러닝 + 딥러닝', 박해선 지음, 한빛미디어