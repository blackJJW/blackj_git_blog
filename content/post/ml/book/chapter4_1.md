---
title: "[ML][Book][혼공 머신러닝+딥러닝] ch4-1 로지스틱 회귀"
description: ""
date: "2022-04-22T16:29:45+09:00"
thumbnail: ""
categories:
  - "ML"
tags:
  - "Python"
  - "ML"

---
<!--more-->

## 럭키백의 확률
- 럭키백에 들어갈 수 있는 생선은 7개
- 럭키백에 들어간 생선의 크기, 무게 등이 주어졌을 때 7개 생선에 대한 확률을 출력
- 길이, 높이, 두께, 대각성 길이, 무게 사용

### 데이터 준비



```python
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
fish.head()
```






|   | Species | Weight | Length | Diagonal | Height  | Width  |
|---|---------|--------|--------|----------|---------|--------|
| 0 | Bream   | 242.0  | 25.4   | 30.0     | 11.5200 | 4.0200 |
| 1 | Bream   | 290.0  | 26.3   | 31.2     | 12.4800 | 4.3056 |
| 2 | Bream   | 340.0  | 26.5   | 31.1     | 12.3778 | 4.6961 |
| 3 | Bream   | 363.0  | 29.0   | 33.5     | 12.7300 | 4.4555 |
| 4 | Bream   | 430.0  | 29.0   | 34.0     | 12.4440 | 5.1340 |





> 데이터프레임(dataframe)
- 판다스에서 제공하는 2차원 표 형식의 주요 데이터 구조
- 넘파이 배열과 비슷하게 열과 행으로 이루어짐
- 통계와 그래프를 위한 메서드을 풍부하게 제공
- 넘파이로 상호 변환이 쉽고 사이킷런과도 잘 호환

- Species 열에서 고유한 값을 추출
- unique() 함수 사용


```python
print(pd.unique(fish['Species']))
```

    ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']
    

- Species -> 타깃


```python
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
```

- 데이터프레임에서 여러 열을 선택하면 새로운 데이터프레임이 반환
- to_numpy() 메서드로 넘파이 배열로 바꾸어 fish_input에 저장


```python
print(fish_input[:5])
```

    [[242.      25.4     30.      11.52     4.02  ]
     [290.      26.3     31.2     12.48     4.3056]
     [340.      26.5     31.1     12.3778   4.6961]
     [363.      29.      33.5     12.73     4.4555]
     [430.      29.      34.      12.444    5.134 ]]
    

> Species 열을 선택할 때 fish[['Species']]와 같이 두 개의 괄호를 사용하지 않도록 한다. 이렇게 하면 fish_target이 2차원이 된다.


```python
fish_target = fish['Species'].to_numpy()
```

- 훈련 세트와 테스트 세트로 나눔


```python
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(fish_input, fish_target, random_state = 42)
```

- 사이킷런의 StandardScaler 클래스를 사용해 훈련 세트와 테스트 세트를 표준화 전처리
- 훈련 세트와 통계 값으로 테스트 세트를 변환해야 함


```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

- k-최근접 이웃 분류기로 테스트 세트에 들어 있는 확률을 예측

### k-최근접 이웃 분류기의 확률
- 사이킷런의 KNeighborsClassifier 클래스 객체를 만들고 훈련 세트로 모델을 훈련한 다음 훈련 세트와 테스트 세트의 점수를 확인
- 최근접 이웃 개수인 k를 3으로 지정하여 사용


```python
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target))
print(kn.score(test_scaled, test_target))
```

    0.8907563025210085
    0.85
    

> 다중 분류(multi-class classification)
- 타깃 데이터에 2개 이상의 클래스가 포함된 문제

- 타깃값을 그대로 사이킷런 모델에 전달하면 순서가 자동으로 알파벳 순으로 매겨짐.
- KNeighborsClassifier에서 정렬된 타깃값은 classes_ 속성에 저장


```python
print(kn.classes_)
```

    ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
    

- 'Bream'이 첫 번째 클래스, 'Pakki'가 두 번째 클래스가 되는 식
- predict() 메서드으는 타깃값으로 예측을 출력
- 테스트 세트에 있는 처음 5개 샘플의 타깃값을 예측


```python
print(kn.predict(test_scaled[:5]))
```

    ['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']
    

- 사이킷런의 분류 모델은 predict_proba() 메서드로 확률값을 반환
- 테스트 세트에 있는 처음 5개의 샘플에 대한 확률을 출력
- 넘파이 round() 함수는 기본으로 소수점 첫째 자리에서 반올림
- decimals 매개변수로 유지할 소수점 아래 자릿수를 지정할 수 있음.


```python
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4)) # <- 소수점 네 번째 자리까지 표기, 다섯 번째 자리에서 반올림
```

    [[0.     0.     1.     0.     0.     0.     0.    ]
     [0.     0.     0.     0.     0.     1.     0.    ]
     [0.     0.     0.     1.     0.     0.     0.    ]
     [0.     0.     0.6667 0.     0.3333 0.     0.    ]
     [0.     0.     0.6667 0.     0.3333 0.     0.    ]]
    

- 첫 번째 열이 'Bream'에 대한 확률
- 두 번째 열이 'Parkki'에 대한 확률

> kneighbors() 메서드의 입력은 2차원 배열이어야 함.
- 이를 위해 넘파이 배열의 슬라이싱 연산자 사용
- 슬라이싱 연산자는 하나의 샘플만 선택해도 항상 2차원 배열이 만들어짐.


```python
distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])
```

    [['Roach' 'Perch' 'Perch']]
    

- 이 샘플의 이웃은 다서 번째 클래스인 'Roach'가 1개, 세 번째 클래스인 'Pearch'가 2개
- 따라서 다섯 번째 클래스에 대한 확률 1/3 = 0.3333
- 세 번째 클래스에 대한 확률 2/3 = 0.6667

## 로지스틱 회귀
- 로지스틱 회귀(logistic regression) : 회귀이지만 분류 모델
- 선형 회귀와 동일하게 선형 방정식을 학습
- ex) $z = a\times (Weight) + b\times (Length)+c\times (Diagonal)+ d\times (Height)+e\times (Width)+f$
- a, b, c, d, e -> 가중치 혹은 게수
- 특성은 늘었지만 다중 회귀를 위한 선형 방정식과 동일
- z는 어떤 값도 가능, 하지만 확류링 되려면 0 ~ 1(또는 0 ~ 100%) 사이 값이 되어야 함
- z가 아주 큰 양수 일 때 1이 되도록 바꾸는 방법은?
  - 시그모이드 함수(sigmoid function 또는 로지스틱 함수logistic function)를 사용하면 가능
  - 시그모이드 함수 : $\phi=\frac{1}{1+e^{-z}}$
  - 선형 방정식의 출력 z의 음수를 사용해 자연 상수 e를 거듭제곱하고 1을 더한 값의 역수를 취합
  - z가 무한하게 큰 음수일 경우 이 함수는 0에 가까워지고, z가 무한하게 큰 양수가 될 때는 1에 가까워 짐.
  - z = 0 일 때는 0.5
  - z가 어떤 값을 갖더라도 $\phi$는 절대로 0 ~ 1 사이의 범위를 벚어날 수 없음.
  - 0 ~ 1 사이의 값을 0 ~ 100%까지의 확률로 해석 가능


```python
import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))
plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter4_1/output_33_0.png)


- 로지스틱 회귀 모델 LogisticRegression 클래스
- 이진분류를 시행하면, 시그모이드 함수의 출력이 0.5보다 크면 양성 클래스, 0.5보다 작으면 음성 클래스로 판단    

> 딱 0.5일 경우
  - 라이브러리마다 다를 수 있음.
  - 사이킷런에서는 0.5일 때 음성 클래스로 판단

### 로지스틱 회귀로 이진 분류 수행
- 넘파이 배열은 True, False 값을 전달하여 행을 선택 가능
- 불리언 인덱싱(boolean indexing)
- ex) 'A'에서 'E'까지 5개의 원소로 이루어진 배열
  - 여기에서 'A'와 'C'만 골라내려면 첫 번째 원소롸 세 번째 원소만 True이고 나머지는 False인 배열을 전달



```python
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])
```

    ['A' 'C']
    

- 훈련 세트에서 도미(Bream)와 빙어(Smelt)의 행만 고름.


```python
bream_semlt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_semlt_indexes]
target_bream_smelt = train_target[bream_semlt_indexes]
```

- bream_smelt_indexes 뱅열은 도미와 빙어일 경우 True, 그 외에는 False
- 따라서 이 배열을 사용해 train_scaled와 train_target 뱅열에 불리언 인덱싱을 적용하면 손쉽게 도미와 빙어 데이터만 골라낼 수 있음
- LogisticRegression 클래스는 선형 모델이므로 sklearn.linear_model 아래에 있음


```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)
```




    LogisticRegression()




```python
print(lr.predict(train_bream_smelt[:5]))
```

    ['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']
    


```python
print(lr.predict_proba(train_bream_smelt[:5]))
```

    [[0.99759855 0.00240145]
     [0.02735183 0.97264817]
     [0.99486072 0.00513928]
     [0.98584202 0.01415798]
     [0.99767269 0.00232731]]
    


```python
print(lr.classes_)
```

    ['Bream' 'Smelt']
    

- 빙어(Smelt) -> 양성 클래스
- predict_proba() 메서드가 반환한 배열 값을 보면 두 번째 샘플만 양성 클래스인 빙어의 확률이 높음.
- 나머지는 모두 도미(Bream)로 예측

> 도미(Bream)를 양성 클래스로 사용하려면 어떻게?
- Bream의 타깃값을 1로 만들고 나머지를 0으로 만들어 사용


```python
print(lr.coef_, lr.intercept_)
```

    [[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]
    

- 이 로지스틱 회귀 모델이 학습한 방정식
  - $z = -0.404\times (Weight) - 0.576\times (Length)-0.663\times (Diagonal)-1.013+\times (Height)-0.732\times (Width)-2.161$
- decision_function() 메서드로 z값 출력 가능


```python
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)
```

    [-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ]
    

- 이 z값을 시그모이드 함수에 통과시키면 확률을 얻을 수 있음.
- 사이파이(scipy)라이브러리에도 시그모이드 함수 존재
  - expit()


```python
from scipy.special import expit
print(expit(decisions))
```

    [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731]
    

- predict_proba() 메서드 출력의 두 번째 열의 값과 동일
- decision_function() 메서드는 양성 클래스에 대한 z값을 반환

### 로지스틱 회귀로 다중 분류 수행
- LogisticRegression 클래스는 기본적으로 반복적인 알고리즘을 사용.
  - max_iter : 반복 횟수를 지정, 기본값 = 100
- 기본적으로 릿지 회귀와 같이 계수의 제곱을 규제.
  - L2 규제
  - 릿지 회귀에서는 alpha 매개변수로 규제의 양을 조절
    - alpha가 커지면 규제도 커짐
  - LinearRegression에서 규제를 제어하는 매개변수 = C
    - C는 작을수록 규제가 커짐, 기본값 = 1



```python
lr = LogisticRegression(C = 20, max_iter = 1000)
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

    0.9327731092436975
    0.925
    


```python
print(lr.predict(test_scaled[:5]))
```

    ['Perch' 'Smelt' 'Pike' 'Roach' 'Perch']
    


```python
proba = lr.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=3))
```

    [[0.    0.014 0.841 0.    0.136 0.007 0.003]
     [0.    0.003 0.044 0.    0.007 0.946 0.   ]
     [0.    0.    0.034 0.935 0.015 0.016 0.   ]
     [0.011 0.034 0.306 0.007 0.567 0.    0.076]
     [0.    0.    0.904 0.002 0.089 0.002 0.001]]
    


```python
print(lr.classes_)
```

    ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']
    

- 이진 분류는 샘플마다 2개의 확률을 출력하고 다중 분류는 샘플마다 클래스 개수 만큼 출력
- 이중 가장 높은 확률이 예측 클래스


```python
print(lr.coef_.shape, lr.intercept_.shape)
```

    (7, 5) (7,)
    

- 이 데이터는 5개의 특성을 사용하므로 coef_ 배열의 열은 5개
- 하지만 위를 보면 행은 7
  - 이진 분류에서 보았던 z를 7개나 계산
  - 다중 분류는 클래스마다 z값을 하나씩 계산
  - 가장 높은 z값을 출력하는 클래스가 예측 클래스
- 다중분류는 소프트맥스(softmax)함수를 사용하여 7개의 z값을 확률로 변환

> 소프트맥스 함수
- 시그모이드 함수는 하나의 선형 방정식의 출력값을 0 ~ 1 사이로 압축
- 소프트맥스 함수는 여러 개의 선형 방정식의 출력값을 0 ~ 1 사이로 압축하고 전체 합이 1이 되도록 한다. 
- 이를 위해 지수 함수를 사용하기 때문에 *정규화된 지수 함수*라고도 부름
- 먼저 7개의 z값의 이름을 z1에서 z7
- z1 ~ z7까지의 값을 사용해 지수 함수 $e^{z1}\sim e^{z7}$을 계산해 모두 더함
  - $e\_sum = e^{z1}+e^{z2}+e^{z3}+e^{z4}+e^{z5}+e^{z6}+e^{z7}$
  - $e^{z1}\sim e^{z7}$을 각각 $e\_sum$으로 나누어 주면 된다.
    - $s1 = \frac{e^{z1}}{e\\_sum}, s2 = \frac{e^{z2}}{e\\_sum}, s3 = \frac{e^{z3}}{e\\_sum} \cdots, s7 = \frac{e^{z7}}{e\\_sum}$
    - s1에서 s7까지 모두 더하면 분자와 분모가 같아지므로 1이 된다.

- 이진 분류에서처럼 decision_fuction() 메서드로 z1 ~ z7까지의 값을 구한 다음 소프트맥스 함수를 사용해 확률로 바꾼다.


```python
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
```

    [[ -6.5    1.03   5.16  -2.73   3.34   0.33  -0.63]
     [-10.86   1.93   4.77  -2.4    2.98   7.84  -4.26]
     [ -4.34  -6.23   3.17   6.49   2.36   2.42  -3.87]
     [ -0.68   0.45   2.65  -1.19   3.26  -5.75   1.26]
     [ -6.4   -1.99   5.82  -0.11   3.5   -0.11  -0.71]]
    

- 사이파이는 소프트맥스 함수도 제공
- scipy.special 아래 softmax() 함수를 임포트


```python
from scipy.special import softmax
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))
```

    [[0.    0.014 0.841 0.    0.136 0.007 0.003]
     [0.    0.003 0.044 0.    0.007 0.946 0.   ]
     [0.    0.    0.034 0.935 0.015 0.016 0.   ]
     [0.011 0.034 0.306 0.007 0.567 0.    0.076]
     [0.    0.    0.904 0.002 0.089 0.002 0.001]]
    

- 앞서 구한 decision 배열을 softmax() 함수에 전달
- softmax()의 axis 매개변수는 소프트맥스를 계산할 축을 지정
  - axis =1로 지정하여 각 행, 즉 각 샘플에 대해 소프트맥스를 계산
  - axis를 지정하지 않으면 배열 전체에 대해 소프트맥스를 계산

## 마무리

### 키워드로 끝내는 핵심 포인트
- 로지스틱 회귀 : 선형 방정식을 사용한 분류 알고리즘.
  - 선형 회귀와 달리 시그모이드 함수나 소프트맥스 함수를 사요하여 클래스 확률을 출력할 수 있다.
- 다중 분류 : 타깃 클래스가 2개 이상이 분류 문제.
  - 로지스틱 회귀는 다중 분류를 위해 소프트맥스 함수를 상요하여 클래스를 예측
- 시그모이드 함수 : 선형 방정식의 출력을 0과 1사이의 값으로 압축하며 이진 분류를 위해 사용
- 소프트맥수 함수 : 다중 분류에서 여러 선형 방정식의 출력 결과를 정규화하여 합이 1이 되도록 만든다. 

### 핵심 패키지와 함수
> scikit-learn
- LigisticRegression : 선형 분류 알고리즘인 로지스틱 회귀를 위한 클래스
  - solver : 사용할 알고리즘을 선택, 기본값 = 'lbfgs'
    - 'sag' : 확률적 평균 경사 하강법 알고리즘
      - 특성과 샘플 수가 많을 때 성능은 빠르고 좋음.
    - 'saga' : 'sag'의 개선 버전
  - penalty : L2 규제(릿지 방식), L1 규제(라쏘 방식)를 선택, 기본값 = 'l2'
  - C : 규제의 강도를 제어, 기본값 = 1.0
    - 값이 작을수록 규제가 강해짐
- predict_proba() : 예측확률을 반환
  - 이진 분류의 경우에는 샘플마다 음성 클래스와 양성 클래스에 대한 확률을 반환
  - 다중 분류의 경우에는 샘플마다 모든 클래스에 대한 확률을 반환
- decision_function() : 모델이 학습한 선형 방정식의 출력을 반환
  - 이 값이 0보다 크면 양성 클래스, 작거나 같으면 음성 클래스로 예측
  - 다중 분류의 경우 각 클래스마다 선형 방정식을 계산
    - 가장 큰 값이 예측 클래스

# Book : '혼자 공부하는 머신러닝 + 딥러닝', 박해선 지음, 한빛미디어