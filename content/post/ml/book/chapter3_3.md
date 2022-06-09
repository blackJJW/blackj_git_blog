---
title: "[ML][Book][혼공 머신러닝+딥러닝] ch3-3 특성 공학과 규제"
description: ""
date: "2022-04-22T16:28:45+09:00"
thumbnail: ""
categories:
  - "ML"
tags:
  - "Python"
  - "ML"

---
<!--more-->

## 다중 회귀
- 여러 개의 특성을 사용한 선형 회귀를 다중 회귀(multiple regression)
- 1개의 특성을 사용할 경우 선형 회귀 모델이 학습하는 것은 직선
- 2개의 특성을 사용할 경우에는 평면을 학습
- 3개일 경우 : 3차원 공간 이상을 그리거나 상상불가
  - 특성이 많은 고차원에서는 선형 회귀가 매우 복잡한 모델을 표현
- 특성 공학(feature engineering) : 기존의 특성을 사용해 새로운 특성을 뽑아내는 작업

## 데이터 준비
- 판다스(pandas) : 데이터 분석 라이브러리
- 데이터프레임(dataframe) : 판다스의 핵심 데이터 구조

1. read_csv() 함수로 데이터프레임을 생성
2. to_numpy() 메서드를 사용해 넘파이 배열로 바꿈


```python
import pandas as pd

df = pd.read_csv('https://bit.ly/perch_csv_data')
perch_full = df.to_numpy()
print(perch_full)
```

    [[ 8.4   2.11  1.41]
     [13.7   3.53  2.  ]
     [15.    3.82  2.43]
     [16.2   4.59  2.63]
     [17.4   4.59  2.94]
     [18.    5.22  3.32]
     [18.7   5.2   3.12]
     [19.    5.64  3.05]
     [19.6   5.14  3.04]
     [20.    5.08  2.77]
     [21.    5.69  3.56]
     [21.    5.92  3.31]
     [21.    5.69  3.67]
     [21.3   6.38  3.53]
     [22.    6.11  3.41]
     [22.    5.64  3.52]
     [22.    6.11  3.52]
     [22.    5.88  3.52]
     [22.    5.52  4.  ]
     [22.5   5.86  3.62]
     [22.5   6.79  3.62]
     [22.7   5.95  3.63]
     [23.    5.22  3.63]
     [23.5   6.28  3.72]
     [24.    7.29  3.72]
     [24.    6.38  3.82]
     [24.6   6.73  4.17]
     [25.    6.44  3.68]
     [25.6   6.56  4.24]
     [26.5   7.17  4.14]
     [27.3   8.32  5.14]
     [27.5   7.17  4.34]
     [27.5   7.05  4.34]
     [27.5   7.28  4.57]
     [28.    7.82  4.2 ]
     [28.7   7.59  4.64]
     [30.    7.62  4.77]
     [32.8  10.03  6.02]
     [34.5  10.26  6.39]
     [35.   11.49  7.8 ]
     [36.5  10.88  6.86]
     [36.   10.61  6.74]
     [37.   10.84  6.26]
     [37.   10.57  6.37]
     [39.   11.14  7.49]
     [39.   11.14  6.  ]
     [39.   12.43  7.35]
     [40.   11.93  7.11]
     [40.   11.73  7.22]
     [40.   12.38  7.46]
     [40.   11.14  6.63]
     [42.   12.8   6.87]
     [43.   11.93  7.28]
     [43.   12.51  7.42]
     [43.5  12.6   8.14]
     [44.   12.49  7.6 ]]
    


```python
import numpy as np

perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])
```


```python
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(perch_full, perch_weight, random_state=42)
```

## 사이킷런의 변환기
- 사이킷런은 특성을 만들거나 전처리하기 위한 다양한 클래스를 제공
- 이런 클래스를 변환기(transformer)
> LinearRegression과 같은 모델 클래스는 추정기(estimator)라고 부름
- 모델 클래스 : fit(), score(), predict()
- 변환기 클래스 : fit(), transform()

- 사용할 변환기 : polynomialFeature 클래스
  - sklearn.preprocessing 패키지에 포함
  


```python
from sklearn.preprocessing import PolynomialFeatures
```


```python
poly = PolynomialFeatures()
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
```

    [[1. 2. 3. 4. 6. 9.]]
    

- 훈련(fit)을 해야 변환(transform)이 가능
- 사이킷런의 일관된 api 때문에 두 단게로 나뉘어져 있음
  - 두 메서드를 하나로 붙인 fit_transform 메서드가 있음

- fit() 메서드는 새롭게 만들 특성 조합을 찾고 transform() 메서드는 실제로 데이터를 변환
- 변환기는 입력 데이터를 변환하는 데 타깃 데이터가 필요하지 않음
  - fit() 메서드에 입력 데이터만 전달
  - ex) 2개의 특성(원소)을 사진 샘플 [2, 3]이 6개의 특성을 가진 [1. 2. 3. 4. 6. 9.]로 바뀜

- PolynomialFeatures 클래스는 기본적으로 각 특성을 제곱한 항을 추가하고 특성끼리 서로 곱한 항을 추가
  - $무게 = a \times 길이 + b \times 높이 + c \times 두께 + d \times 1$
  - 선형 방정식의 절편을 항상 값이 1인 특성과 곱해지는 계수
  - (길이, 높이, 두께, 1)
- include_bias = False로 지정하여 다시 특성을 변환


```python
poly = PolynomialFeatures(include_bias=False)
poly.fit([[2, 3]])
print(poly.transform([[2, 3]]))
```

    [[2. 3. 4. 6. 9.]]
    

- include_bias=False로 지정하지 않아도 사이킷런 모델은 자동으로 특성에 추가된 절편 항을 무시

- train_input에 적용


```python
poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
print(train_poly.shape)
```

    (42, 9)
    

- get_feature_names_out() 메서드를 호출하면 9개의 특성이 각각 어떤 입력의 조합으로 만들어졌는지 알려줌


```python
poly.get_feature_names_out()
```




    array(['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2',
           'x2^2'], dtype=object)



- 'x0' : 첫 번째 특성
- 'x0^2 : 첫 번째 특성의 제곱
- 'x0 x1'은 첫 번째 특성과 두 번째 특성의 곱


```python
test_poly = poly.transform(test_input) # 테스트 세트 변환
```

## 다중 회귀 모델 훈련


```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
```

    0.9903183436982124
    

- 특성이 늘어나면 선형 회귀늬 능력은 매우 강함


```python
print(lr.score(test_poly, test_target))
```

    0.9714559911594134
    

- PolynomialFeatures 클래스의 degree 매개변수를 사용하며 필요한 고차항의 최대 차수를 지정할 수 있음.ㅡ


```python
poly = PolynomialFeatures(degree =5, include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
print(train_poly.shape)
```

    (42, 55)
    


```python
lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))
```

    0.9999999999991097
    


```python
print(lr.score(test_poly, test_target))
```

    -144.40579242684848
    

- 특성의 개수를 크게 늘리면 선형 모델은 아주 강력해짐
- 하지만 이런 모델은 훈련 세트에 너무 과대적합되므로 테스트 세트에서는 형편없는 점수를 만듬

## 규제 
- 규제(regularization) : 머신러닝 모델이 훈련 세트를 너무 과도하게 학습하지 못하도록 훼방하는 것
- 모델이 훈련 세트에 과대적합되지 않도록 만드는 것
- 선형 모델의 경우 특성에 곱해지는 계수(기울기)의 크기를 작게 만드는 일

- 규제를 적용하기 전 정규화
- StandardScaler 클래스 사용 -> 변환기의 일종


```python
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()
ss.fit(train_poly)
train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)
```

- 훈련 세트에서 학습한 평균과 표준편차는 StandardScaler 클래스 객체의 mean_, scale_ 속성에 저장

- 선형 회귀 모델에 규제를 추가한 모델을 릿지(ridge)와 라쏘(lasso)
- 릿지 : 계수를 제곱한 값을 기준으로 규제를 적용
- 라쏘 : 계수의 절댓값을 기준으로 규제를 적용

## 릿지 회귀


```python
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
```

    0.9896101671037343
    


```python
print(ridge.score(test_scaled, test_target))
```

    0.9790693977615397
    

- 훈련 테스트에 과대적합이 되지 않아 테스트 세트에서도 좋은 성능을 내고 있음
- 릿지와 라쏘 모델을 사용할 때 규제의 양을 임의로 조절 가능
  - alpha 매개변수 
    - 값이 크면 규제 강도가 세지므로 계수 값을 줄이고, 조금 더 과소적합되도록 유도
    - 값아 작으면 계수를 줄이는 역할이 줄어들고 선형 회귀 모델과 유사, 과대적합될 가능성 커짐

> 사람이 직접 지정해야 하는 매개변수
  - 머신러닝 모델이 학습할 수 없고 사람이 알려줘야 하는 파라미터를 하이퍼파라미터(hyperparameter)
  - 사이킷런과 같은 머신러닝 라이브러리에서 하이퍼파라미터는 클래스와 메서드의 매개변수로 표현

- 적절한 alpha 값을 찾는 방법 -> alpha 값에 대한 $R^2$값의 그래프 작성
  - 훈련 세트와 테스트 세트의 점수가 가장 가까운 지점이 최적의 alpha


```python
import matplotlib.pyplot as plt
train_score = []
test_score = []
```


```python
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
  # 릿지 모델 생성
  ridge = Ridge(alpha = alpha)
  # 릿지 모델 훈련
  ridge.fit(train_scaled, train_target)
  # 훈련 점수와 테스트 점수 저장
  train_score.append(ridge.score(train_scaled, train_target))
  test_score.append(ridge.score(test_scaled, test_target))
```

- alpha 값을 0.001부터 10배씩 늘렸기 때문에 이대로 그래프를 그리면 그래프 왼쪽이 너무 촘촘해짐
- 로그함수로 바꾸어 지수로 표현
  - 0.001 -> -3, 0.01 -> 22   
  
> 넘파이 로그 함수
  - np.log() : 자연상수 e를 밑으로 하는 자연 로그
  - np.log10() : 10을 밑으로 하는 상용로그


```python
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter3_3/output_46_0.png)


- blue : 훈련 세트 그래프
- orange : 테스트 세트 그래프
- 왼쪽 : 훈련 세트에는 잘 맞고 테스트 세트에는 형편없는 과대적합의 전형적 모습
- 오른쪽 : 훈련세트와 테스트 세트의 점수가 모두 낮아지는 과소적합
- 적절한 alpha 값은 두 그래프가 가장 가깝고 테스트 세트의 점수가 가장 높은 -1, 즉 $10^{-1}=0.1$


```python
ridge = Ridge(alpha = 0.1)
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))
```

    0.9903815817570366
    0.9827976465386926
    

## 라쏘 회귀


```python
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
```

    0.989789897208096
    


```python
print(lasso.score(test_scaled, test_target))
```

    0.9800593698421883
    


```python
# alpha 적용
train_score = []
test_score = []


alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
  # 라쏘 모델 생성
  lasso = Lasso(alpha = alpha, max_iter=10000)
  # 라쏘 모델 훈련
  lasso.fit(train_scaled, train_target)
  # 훈련 점수와 테스트 점수 저장
  train_score.append(lasso.score(train_scaled, train_target))
  test_score.append(lasso.score(test_scaled, test_target))
```

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:648: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.878e+04, tolerance: 5.183e+02
      coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random, positive
    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_coordinate_descent.py:648: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations, check the scale of the features or consider increasing regularisation. Duality gap: 1.297e+04, tolerance: 5.183e+02
      coef_, l1_reg, l2_reg, X, y, max_iter, tol, rng, random, positive
    


```python
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter3_3/output_53_0.png)


- blue : 훈련 세트 그래프
- orange : 테스트 세트 그래프
- 왼쪽 : 훈련 세트에는 잘 맞고 테스트 세트에는 형편없는 과대적합의 전형적 모습
- 오른쪽 : 훈련세트와 테스트 세트 점수가 좁혀지고 있음
- 적절한 alpha 값은 1, 즉 $10^1=10$


```python
lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))
print(lasso.score(test_scaled, test_target))
```

    0.9888067471131867
    0.9824470598706695
    


```python
# 라쏘 모델의 계수가 0일 경우
print(np.sum(lasso.coef_ == 0))
```

    40
    

- 55개의 특성을 주입, 라쏘 모델을 사용한 특성 15개
- 이런 특징 때무네 라쏘 모델을 유용한 특성을 골라내는 용도로 사용할 수 있다.

## 마무리

### 키워드로 끝내는 핵심 포인트
- 다중 회귀 : 여러 개의 특성을 사용하는 회귀 모델. 특성이 많으면 선형 모델은 강력한 성능을 발휘
- 특성 공학 : 주어진 특성을 조합, 새로운 특성을 만드는 작업 과정
- 릿지 : 규제가 있는 선형 회귀 모델 중 하나.
  - 선형 모델의 계수를 작게 만들어 과대 적합을 완화
  - 비교적 효과가 좋아 널리 사용
- 라쏘 : 릿지와 달리 계수 값을 아예 0으로 만들 수 있음
- 하이퍼파라미터 : 머신러닝 알고리즘이 학습하지 않는 파라미터
  - 이런 파라미터는 사람이 사전에 지정
  - 대표적으로 릿지와 라쏘의 규제 강도 alpha 파라미터

### 핵심 패키지와 함수

> pandas
  - read_csv() : csv파일을 읽어 판다스 데이터 프레임으로 변환
    - sep : 구분자 지정. 기본값 = 콤마(,)
    - header : 데이터프레임의 열 이름으로 사용할 csv파일의 행 번호 지정. 기본적으로 첫 번째 행을 열 이름으로 사용
    - skiprows : 파일에서 읽기 전에 건너 뛸 행의 개수를 지정
    - nrows : 파일에서 읽을 행의 개수 지정

> scikit-learn
  - PolynomialFeatures : 주어진 특성을 조합하여 새로운 특성을 생성
    - degree : 최고차수 지정. 기본값 = 2
    - interaction_only : True이면 거듭제곱 항은 제외되고 특성 간의 곱셈 항만 추가. 기본값 = False
    - include_bias : False이면 절편을 위한 특성을 추가하지 않음. 기본값 = True
  - Ridge : 규제가 있는 회귀 알고리즘이 릿지 회귀 모델을 훈련
    - alpha : 규제 강도 조절. 클수로 강도 세짐. 기본값 = 1
    - solver : 최적의 모델을 찾기 위한 방법 지정. 
      - 기본값 = 'auto'. 데이터에 따라 자동으로 선택
      - 'sag' : 확률적 평균 경사 하강법 알고리즘, 특성과 샘플 수가 많을 때에 성능이 빠르고 좋음
      - 'saga' : 'sag'의 개선 버전
    - random_state : solver가 'sag'나 'saga'일 때 넘파이 나눗 시드값을 지정할 수 있음
  - Lasso : 최적의 모델을 찾기 위해 좌표축을 따라 최적화를 수행하는 좌표 하강법(coordinate descent)을 사용
    - alpha, random_state : Ridge와 동일
    - max_iter : 알고리즘의 수행 반복 횟수 지정. 기본값 = 1000

# Book : '혼자 공부하는 머신러닝 + 딥러닝', 박해선 지음, 한빛미디어