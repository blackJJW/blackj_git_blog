---
title: "[ML][Book][혼공 머신러닝+딥러닝] ch5-2 교차 검증과 그리드 서치"
description: ""
date: "2022-04-23T16:31:45+09:00"
thumbnail: ""
categories:
  - "ML"
tags:
  - "Python"
  - "ML"

---
<!--more-->

## 검증 세트
- 테스트 세트를 사용하지 안흐면 모델이 과대적합인지 과소적합인지 판단하기 어려움.
- 테스트 세트를 사용하지 않고 이를 측정하는 간단한 방법 -> 훈련 세트를 또 나누는 것.
- 이를 데이터를 검증 세트(validation set)

- 훈련 세트에서 모델을 평가하고 검증 세트로 모델을 평가.
- 테스트하고 싶은 매개변수를 바꿔가며 가장 좋은 모델을 선택.
- 이 매개변수를 사용해 훈련 세트와 검증 세트를 합쳐 전체 훈련 데이터에서 모델을 다시 훈련.
- 마지막에 테스트 세트에서 최종 점수를 평가.


```python
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
```


```python
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
```


```python
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size = 0.2, random_state=42)
```


```python
sub_input, val_input, sub_target, val_target = train_test_split(train_input, train_target, test_size = 0.2, random_state=42)
```

- train_test_split 함수를 2번 적용하여, 훈련 세트와 검증 세트로 나눠 준 것


```python
print(sub_input.shape, val_input.shape)
```

    (4157, 3) (1040, 3)
    


```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)
print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))
```

    0.9971133028626413
    0.864423076923077
    

- 이 모델은 훈련 세트에 과대적합

## 교차 검증(cross validation)
- 안정적인 검증 점수를 얻고 훈련에 더 많은 데이터를 사용 가능
- 검증 세트를 떼어 내어 평가하는 과정을 여러 번 반복.
- 그 다음 이 점수를 평균하여 최종 검증 점수를 얻음.

> 3-폴드 교차 검증
- 훈련 세트를 세 부분으로 나눠서 교차 검증을 수행하는 것을 3-폴드 교차 검증.
- 통칭 k-폴드 교차 검증(k-fold cross validation)
- 훈련 세트를 몇 부분으로 나누냐에 따라 k-겹 교차 검증이라고 부름.
- 보통 5-폴드, 10-폴드를 많이 사용.
  - 데이터의 80 ~ 90%까지 훈련에 사용 가능
  - 검증 세트가 줄어들지만 각 폴드에서 계산한 검증 점수를 평균하기 때문에 안정된 점수로 생각 가능.
- cross_validate() : 교차 검증 함수


```python
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)
```

    {'fit_time': array([0.01163411, 0.01068687, 0.01692414, 0.01103592, 0.00983691]), 'score_time': array([0.00145602, 0.00140238, 0.00137901, 0.00128579, 0.00525594]), 'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}
    

- fit_time, score_time, test_score 키를 가진 딕셔너리를 반환.
- 처음 2개의 키는 각각 모델을 훈련하는 시간과 검증하는 시간을 의미
- 각 키마다 5개의 숫자가 담겨져 있음
  - cross_validate() 함수는 기본적으로 5-폴드 교차 검증을 수행
    - cv : 폴드 수 지정.
- 교차 검증의 최종 점수는 test_score 키에 담긴 5개의 점수를 평균하여 얻음.
  - 이름은 test_score지만 검증 폴드의 점수


```python
import numpy as np
print(np.mean(scores['test_score']))
```

    0.855300214703487
    

- 교차 검증을 수행하면 입력한 모델에서 얻을 수 있는 최상의 검증 점수를 가늠 가능.
- cross_validate()는 훈련 세트를 섞어 폴드를 나누지 않음.
  - 앞서 train_test_split() 함수로 전체 데이터를 섞은 후 훈련 세트를 준비했기 때문에 섞을 필요 없음.
- 만약 훈련 세트를 섞으려면 분할기(splitter)를 지정

- 분할기는 교차 검증에서 폴드를 어떻게 나눌지 결정.
- cross_validate() 함수는 기본적으로 회귀 모델일 경우 KFold 분할기를 사용, 분류 모델일 경우 타깃 클래스를 골고루 나누기 위해 StratifiedKFold를 사용.


```python
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())
print(np.mean(scores['test_score']))
```

    0.855300214703487
    

- 만약, 훈련 세트를 섞은 후 10-폴드 교차 검증을 수행하려면 다음과 같이 한다.


```python
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean(scores['test_score']))
```

    0.8574181117533719
    

- n_splits : 몇(k)폴드 교차 검증을 할 지 지정
- KFold 클래스도 동일한 방식 사용 가능

## 하이퍼파라미터 튜닝

- 모델 파라미터 : 머신러닝 모델이 학습하는 파라미터
- 하이퍼파라미터 : 모델이 학습할 수 없어서 사용자가 지정해야만 하는 파라미터

- 하이퍼파라미터 튜닝 작업
  1. 라이브러리가 제공하는 기본값을 그대로 사용해 모델을 훈련.
  2. 검증 세트의 점수나 교차 검증을 통해서 매개변수를 조금씩 바꿔 본다.
  3. 모델마다 1 ~ 2개에서, 많게는 5 ~ 6개의 매개변수를 제공
    - 매개변수를 바꿔가면서 모델을 훈련, 교차 검증을 수행   

  > 사람의 개입 없이 하이퍼파라미터 튜닝을 자동으로 수행하는 기술을 'AutoML'   

  - 여러 조합을 사용할 때 그리드 서치(Grid Search)를 사용.
    - 사이킷런의 GridSearchCV 클래스는 하이퍼파라미터 탐색과 교차 검증을 한 번에 수행.
    - 별도로 cross_validate() 호출 필요 없음. 

- ex) 기본 매개 변수를 사용한 결정 트리 모델에서 min_inpurity_decrease 매개변수의 최적값을 찾기
  - 먼저 GridSearchCV 클래스를 임포트, 탐색할 매개변수와 탐색할 값의 리스트를 딕셔너리로 생성


```python
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}
```

- GridSearchCV 클래스에 탐색 대상 모델과 params 변수를 전달하여 그리드 서치 객체를 생성


```python
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
```

- gs 객체에 fit()을 호출
- cv 매개변수 기본값 = 5 -> min_impurity_decrease 값마다 5-폴드 -> 총 25번 훈련
- n_jobs : 병렬 실행할 CPU 코어 수 
  - -1 : 모든 코어 사용



```python
gs.fit(train_input, train_target)
```




    GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), n_jobs=-1,
                 param_grid={'min_impurity_decrease': [0.0001, 0.0002, 0.0003,
                                                       0.0004, 0.0005]})



- 사이킷런의 그리드 서치는 훈련이 끝나면 25개의 모델 중에서 검증 점수가 가장 높은 모델의 매개변수 조합으로 전체 훈려 세읕에서 자동으로 다시 모델을 훈련
- 이 모델은 gs객체의 best_estimator\_속성에 저장
- 일반 결정 트리에서 똑같이 사용가능


```python
dt = gs.best_estimator_
print(dt.score(train_input, train_target))
```

    0.9615162593804117
    

- 그리드 서치로 찾은 최적의 매개변수는 best_params_ 속성에 저장


```python
print(gs.best_params_)
```

    {'min_impurity_decrease': 0.0001}
    

- 각 매개변수에서 수행한 교차 검증의 평균점수는 cv_results_속성의 mean_test_score 키에 저장


```python
print(gs.cv_results_['mean_test_score'])
```

    [0.86819297 0.86453617 0.86492226 0.86780891 0.86761605]
    

- 넘파이 argmax() 사용하면 가장 큰 값의 인덱스를 추출
- 이 인덱스를 통해 params 키에 저장된 매개변수 출력 가능
- 이 값이 최상의 검증 점수를 만든 매개변수 조합


```python
best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results_['params'][best_index]) 
```

    {'min_impurity_decrease': 0.0001}
    

- 이 과정을 정리
  1. 먼저 탐색할 매개변수 지정.
  2. 그다음 훈련 세트에서 그리드 서치를 수행하여 최상의 평균 검증 점수가 나오는 매개변수 조합을 찾음. 이 조합은 그리드 서치 객체에 저장
  3. 그리드 서치는 최상의 매개변수에서 (교차 검증에 사용한 훈련 세트가 아니라) 전체 훈련 세트를 사용해 최종 모델을 훈련. 이 모델도 그리드 서치 객체에 저장


```python
params = {'min_impurity_decrease' : np.arange(0.0001, 0.001, 0.0001), 
          'max_depth' : range(5, 20, 1), # 정수만 사용 
          'min_samples_split' : range(2, 100, 10)} # 정수만 사용
```


```python
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)
```




    GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), n_jobs=-1,
                 param_grid={'max_depth': range(5, 20),
                             'min_impurity_decrease': array([0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008,
           0.0009]),
                             'min_samples_split': range(2, 100, 10)})




```python
print(gs.best_params_)
```

    {'max_depth': 14, 'min_impurity_decrease': 0.0004, 'min_samples_split': 12}
    


```python
print(np.max(gs.cv_results_['mean_test_score']))
```

    0.8683865773302731
    

### 랜덤 서치
- 매개변수의 값이 수치일 때 값의 범위나 간격을 미리 정하기 어려울 수 있음.
- 너무 많은 매개 변수 조건이 있어 그리드 서치 수행 시간이 오래 걸릴 수 있음.


- 랜덤 서치(random search)이용
- 매개변수 값의 목록을 전달하는 것이 아니라 매개변수를 샘플링할 수 있는 확률 분포 객체를 전달

> 싸이파이(scipy)
- 파이썬의 핵심 과학 라이브러리
- 적분, 보간, 선형 대수, 확률 등을 포함한 수치 계산 전용 라이브러리


```python
from scipy.stats import uniform, randint
```

- 싸이파이의 stats 서비 패키지에 있는 uniform과 randint 클래스는 모두 주어진 범위에서 고르게 값을 뽑음.
  - '균등 분포에서 샘플링'
  - randint : 정숫값을 뽑음
  - uniform : 실숫값을 뽑음


```python
rgen = randint(0, 10)
rgen.rvs(10)
```




    array([0, 3, 8, 2, 3, 8, 3, 8, 7, 1])




```python
np.unique(rgen.rvs(1000), return_counts=True)
```




    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
     array([111,  88, 104,  88, 117,  97,  91,  92,  90, 122]))




```python
ugen = uniform(0, 1)
ugen.rvs(10)
```




    array([0.03797408, 0.58763462, 0.98272677, 0.17041929, 0.05654931,
           0.09258853, 0.87524285, 0.76708824, 0.61140304, 0.36494506])



- min_samples_leaf : 리프 노드가 되기 위한 최소 샘플의 개수
- 어떤 노드가 분할하여 만들어질 자식 노드의 샘플 수가 이 값보다 작을 경우 분할하지 않음.


```python
params = {'min_impurity_decrease' : uniform(0.0001, 0.001), 
          'max_depth' : randint(20, 50),  
          'min_samples_split' : randint(2, 25),
          'min_samples_leaf' : randint(1, 25)
          } 
```

- 샘플링 횟수는 사이킷런의 랜점 서치 클래스인 RandomizedSearchCV의 n_iter 매개변수에 지정


```python
from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, 
                        n_iter=100, n_jobs=-1, random_state=42)
gs.fit(train_input, train_target)
```




    RandomizedSearchCV(estimator=DecisionTreeClassifier(random_state=42),
                       n_iter=100, n_jobs=-1,
                       param_distributions={'max_depth': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7feb8c126650>,
                                            'min_impurity_decrease': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7feb8bfb5ad0>,
                                            'min_samples_leaf': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7feb8c28bf50>,
                                            'min_samples_split': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7feb8bfb54d0>},
                       random_state=42)




```python
print(gs.best_params_)
```

    {'max_depth': 39, 'min_impurity_decrease': 0.00034102546602601173, 'min_samples_leaf': 7, 'min_samples_split': 13}
    


```python
print(np.max(gs.cv_results_['mean_test_score']))
```

    0.8695428296438884
    

- 최적의 모델은 이미 전체 훈련 세트(train_input, train_target)로 훈련, best_estimator_ 속성에 저장


```python
dt = gs.best_estimator_
print(dt.score(test_input, test_target))
```

    0.86
    

## 마무리

### 키워드로 끝내는 핵심 포인트
- 검증 세트 : 하이퍼파라미터 튜닝을 위해 모델을 평가할 때, 테스트 세트를 사용하지 않기 위해 훈련 세트에서 다시 떼어 낸 데이터 세트
- 교차 검증 : 훈련 세트를 여러 폴드로 나눈 다음 한 폴드가 검증 세트의 역할을 하고 나머지 폴드에서는 모델을 훈련.
  - 교차 검증은 이런 식으로 모든 폴드에 대해 검증 점수를 얻어 평균하는 방법
- 그리드 서치 : 하이퍼파라미터 탐색을 자동화해 주는 도구.
  - 탐색할 매개변수를 나열하면 교차 검증을 수행하여 가장 좋은 검증 점수의 매개변수 조합을 선택.
  - 마지막으로 이 매개변수 조합으로 최종 모델을 훈련
- 랜덤 서치 : 연속된 매개변수 값을 탐색할 때 유용.
  - 탐색할 값을 직접 나열하는 것이 아니고 탐색 값을 샘플링할 수 있는 확률 분포 객체를 전달
  - 지정된 횟수만큼 샘플링하여 교차 검증을 수행하기 때문에 시스템 자원이 허락하는 만큼 탐색량을 조절 가능.

### 핵심 패키지와 함수
> scikit-learn
- cross_validate() : 교차 검증을 수행하는 함수
  - 첫 번째 매개변수에 교차 검증을 수행할 모델 객체를 전다. 두 번째와 세 번째 매개변수에 특성과 타깃 데이터를 전달
  - scoring : 검증에 사용할 평가 지표를 지정 가능
    - 기본적으로 분류 모델은 정확도를 'accuracy'
    - 회귀 모델은 결정계수를 의미하는 'r2'
  - cv : 교차 검증 폴드 수나 스플리터 객체를 지정.
    - 기본값 = 5
    - 회귀일 때 : KFold
    - 분류일 때 : StratifiedKFold
  - n_jobs : 교차 검증을 수행할 때 사용할 CPU 코어 수 지정
    - 기본값 = 1
    - -1 : 모든 코어 사용
  - return_train_score 
    - True : 훈련 세트의 점수도 반환
    - 기본값 = False
- GridSearchCV : 교차 검증으로 하이퍼파라미터 탐색을 수행
  - 최상의 모델을 찾은 후 훈련 세트 전체를 사용해 최종 모델을 훈련
  - 첫 번째 매개변수로 그리드 서치를 수행할 모델 객체를 전달
  - 두 번째 매개변수로 탐색할 모델의 매개변수와 값을 전달
  - scoring, cv, n_jobs, return_train_score : cross_validate() 함수와 동일
- RandomizedSearchCV ; 교차 검증으로 랜덤한 하이퍼파라미터 탐색을 수행
  - 최상의 모델을 찾은 후 훈련 세트 전체를 사용해 최종 모델을 훈련
  - 첫 번째 매개변수로 그리드 서치를 수행할 모델 객체를 전달
  - 두 번째 매개변수에는 탐색할 모델의 매개변수와 확률 분포 객체를 전달
  - scoring, cv, n_jobs, return_train_score 매개변수는 cross_validate() 함수와 동일


# Book : '혼자 공부하는 머신러닝 + 딥러닝', 박해선 지음, 한빛미디어