---
title: "[ML][Book][혼공 머신러닝+딥러닝] ch5-3 트리의 앙상블"
description: ""
date: "2022-04-23T16:32:45+09:00"
thumbnail: ""
categories:
  - "ML"
tags:
  - "Python"
  - "ML"

---
<!--more-->

## 정형 데이터와 비정형 데이터
- 정형 데이터(structured data) : csv와 같이 어떤 구조로 되어있는 데이터
  - CSV, 데이터베이스, 엑셀에 저장하기 쉬움
- 비정형 데이터(unstructured data) : 데이터베이스나 엑셀로 표현하기 어려운 데이터
  - 텍스트 에디터, 사진, 음악

> 텍스트나 사진을 데이터베이스에 저장 가능한가?
- 저장 가능.
- 데이터베이스 중에는 구조적이지 않은 데이터를 저장하는 데 편리하도록 발전한 것이 많음.
- NoSQL : 엑셀이나 CSV에 담기 어려운 텍스트나 JSON 데이터를 저장하는 데 용이.

- 정형 데이터를 다루는 데 가장 뛰어난 성과를 내는 알고리즘 : 앙상블 학습(ensemble learning)
  - 대부분 결정 트리 기반

## 랜덤 포레스트

- 랜덤 포레스트(Random Forest) : 앙상블 학습의 대표 주자 중 하나로 안정적인 성능 덕분에 널리 사용.
  - 결정 트리를 랜덤하게 만들어 결정 트리의 숲을 생성
  - 각 결졍 트리의 예측을 사용해 최종 예측 생성

- ex) 1000개 가방에서 100개씩 샘플을 뽑는다면 먼저 1개를 뽀복, 뽑았던 1개를 다시 가방에 넣음.
  - 이런 식으로 계속해서 100개를 가방에서 뽑으면 중복된 샘플을 뽑을 수 있음.
  - 이럴게 만들어진 샘플을 부트스트랩 샘플(bootstrap sample)
- 기본적으로 부트스트랩 샘플은 훈련 세트의 크기와 같게 만듬.
  - 1000개 가방에서 중복하여 1000개의 샘플을 뽑기 때문데 부트스트랩 샘플은 훈련 세트와 크기가 같음


> 부트스트랩
- 데이터 세트에서 중복을 허용하여 데이터를 샘플링하는 방식을 의미.
- 부트스트랩 샘플이란 부트스트랩 방식으로 샘플링하여 분류한 데이터라는 의미.

- 각 노드를 분할할 때 전체 특성 중에서 일부 특성을 무작위로 고른 다음 이 중에서 최선의 분할을 찾음.
- 분류 모델인 RandomForestClassifier는 기본적으로 전체 특성 개수의 제곱근만큼의 특성을 선택.
  - 4개의 특성이 있다면 노드마다 2개를 랜덤하게 선택하여 사용
  - 다만, 회귀 모델인 RandomForestRegressor는 전체 특성을 사용

- 랜덤 포레스트는 랜덤하게 선택한 샘플과 특성을 사용하기 때문에 훈련 세트에 과대적합되는 것을 막아주고 검증 세트와 테스트 세트에서 안정적인 성능을 얻을 수 있음.


```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
wine = pd.read_csv('https://bit.ly/wine_csv_data')
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
```

- cross_validate()를 사용해 교차 검증 수행
- RandomForestClassifier : 기본적으로 100개의 결정 트리를 사용하므로 n_jobs=-1로 지정하여 모든 CPU 코어를 사용하는 것이 좋음.
- return_train_score를 True로 지정하면 검증 점수 뿐만 아니라 훈련 세트에 대한 점수도 같이 반환.
  - 기본값 = False
  - 훈련 세트와 검증 세트를 비교하면 과대적합을 파악하는 데 용이


```python
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(rf, train_input, train_target, return_train_score = True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.9973541965122431 0.8905151032797809
    

- 출력된 결과를 보면 훈련 세트에 다소 과적합

- 랜덤 포레스트는 결정 트리의 앙상블이기 때문에 DecisionTreeClassifier가 제공하는 중요한 매개변수를 모두 제공.
- 결정 트리의 큰 장점 중 하나인 특성 중요도를 계산.
  - 랜덤 포레스트의 특성 중요도는 각 결정 트리의 특성 중요도를 취합.


```python
rf.fit(train_input, train_target)
print(rf.feature_importances_)
```

    [0.23167441 0.50039841 0.26792718]
    

- 랜덤 포레스트가 특성의 일부를 랜덤하게 선택하여 결정 트리를 훈련.
- 그 결과 하나의 특성에 과도하게 집중하지 않고 좀 더 많은 특성이 훈련에 기여할 기회를 얻음.
- 이는 과대적합을 줄이고 일반화 성능을 높이는 데 도움.

- RandomForestClassifier
  - 자체적으로 모델을 평가하는 점수를 얻을 수 있다.
  - 훈련 세트에서 중복을 허용하여 부트스트랩 샘플을 만들어 결정 트리를 훈련.
    - 이 때 부트스트랩 샘플에 포합되지 않고 남는 샘플이 존재.
      - 이런 샘플을 OOB(out of bag) 샘플이고 함.
      - 이 샘플을 이용해여 부트스트랩 샘플로 훈련한 결정 트리를 평가 가능.
  - 이 점수를 얻으려면 RandomForestClassifier 클래스의 oob_score 매개변수를 True로 지정해야 함.
    - 이렇게 하면 랜덤 포레스트는 각 결정 트리의 OOB 점수를 평균하여 출력.


```python
rf = RandomForestClassifier(oob_score =True, n_jobs=-1, random_state=42)
rf.fit(train_input, train_target)
print(rf.oob_score_)
```

    0.8934000384837406
    

- OOB 점수를 사용하면 교차 검증을 대신할 수 있어서 결과적으로 훈련 세트에 더 많은 샘플을 사용 가능.

## 엑스트라 트리
- 엑스트라 트리(extra tree)는 랜덤 포레스트와 비슷하게 동작.
- 전체 특성 중에 일부 특성을 램덤하게 선택하여 노드를 분할하는 데 사용

- 랜덤 포레스트와 엑스트라 트리의 차이점은 부트스트랩 샘플을 사용하지 않는다는 점.
  - 결정 트리를 만들 때 전체 훈련 세트를 사용
  - 대신 노드를 분할할 대 가장 좋은 분할을 찾는것이 아니라 무작위로 분할.


- 엑스트라 트리가 사용하는 결정 트리가 splitter='random'인 결정 트리
- 하나이 결정 트리에서 특성을 무작위로 분할한다면 성능이 낮아지겠지만 많은 트리를 앙상블하기 때문에 과대적합을 막고 검증 세트의 점수를 높이는 효과.


```python
from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(n_jobs=-1, random_state=42)
scores = cross_validate(et, train_input, train_target, 
                        return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.9974503966084433 0.8887848893166506
    

- 보통 엑스트라 트리가 무작위성이 좀 더 크기 때문에 랜점 포레스트보다 더 많은 결정 트리를 훈련해야 함.
- 하지만 랜덤하게 노드를 분할하기 때문에 빠른 계산 속도가 엑스트라 트리의 장점

> 결정 트리는 최적의 분할을 찾는 데 시간을 많이 소모.
- 특히 고려 사항이 많을 때
- 만약 무작위로 나눈다면 훨씬 빨리 트리를 구성하는 것이 가능.

- 엑스트라 트리도 랜덤 포레스트와 같이 특성 중요도 제공


```python
et.fit(train_input, train_target)
print(et.feature_importances_)
```

    [0.20183568 0.52242907 0.27573525]
    

- 엑스트라 트리의 회귀 버전은 ExtraTreeRegressor 클래스

## 그레디언트 부스팅
 

- 그레디언트 부스팅(gradient boosting) : 깊이가 얕은 결정 트리를 사용하여 이전 트리의 오차를 보완하는 방식으로 앙상블을 하는 방법.
- 사이킷런의 GradientBoostingClassifier는 기본적으로 깊이가 3인 트리를 100개 사용.
- 깊이가 얕은 결정 트리를 사용하기 때문데 과대적합에 강하고 일반적으로 높은 일반화 성능을 기대 가능.

- 경사 하강법을 사용하여 트리를 앙상블에 추가
- 분류에서는 로지스틱 손실 함수를 사용하고 회귀에서는 평군 제곱 오차 함수를 사용

- 경사 하강법은 손실 함수를 산으로 정의하고 가장 낮은 곳을 찾아 내려노는 과정
  - 가장 낮은 곳을 찾아 내려오는 방법은 모델의 가중치와 절편을 조금씩 바꾸는 것.
- 그레디언트 부스팅은 결정트리를 계속 추가하면서 가장 낮은 곳을 찾아 이동
  - 깊이가 낮은 트리를 이용
  - 학습률 매개변수로 속도를 조절


```python
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(random_state=42)
scores = cross_validate(gb, train_input, train_target, 
                        return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.8881086892152563 0.8720430147331015
    

- 과대적합이 되지 않음.
- 그레디언트 부스팅은 결정 트리의 개수를 늘려도 과대적합에 매우 강함.
- 학습률을 증가시키고 트리의 개수를 느리면 성능이 조금 더 향상될 수 있음.



```python
gb = GradientBoostingClassifier(n_estimators = 500, learning_rate = 0.2, random_state=42)
scores = cross_validate(gb, train_input, train_target, return_train_score=True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.9464595437171814 0.8780082549788999
    


```python
gb.fit(train_input, train_target)
print(gb.feature_importances_)
```

    [0.15872278 0.68010884 0.16116839]
    

- 매개변수 subsample
  - 기본값 = 1.0 -> 전체 훈련 세트 사용
  - < 1 -> 훈련 세트의 일부 사용

- 일반적으로 그레디언트 부스팅이 랜덤 포레스트보다 조금 더 좋은 성능을 얻을 수 있음.
- 하지만 순서대로 트리를 추가하기에 속도가 느림.
  - GradientBoosingClassifier에는 n_jobs 가 존재하지 않음.
- GradientBoosting의 회귀 버전은 GradientBoostingRegressor

- 그레디언트 부스팅의 속도와 성능을 더욱 개선한 것이 히스토그램 기반 그레디언트 부스팅 

## 히스토그램 기반 그레디언트 부스팅


- 히스토그램 기반 그레디언트 부스팅(histogram-based gradient boosting) : 정형 데이터를 다루는 머신러닝 알고리즘 중에 가장 인기가 높은 알고리즘
- 입력 특성을 256개의 구간으로 나눔.
  - 따라서 노드를 분할할 때 최적의 분할을 매우 빠르게 찾을 수 있음.
  - 256개의 구간 중 하나를 떼어 놓고 누락된 값을 위해서 사용.
  - 따라서 입력에 누락된 특성이 있더라도 이를 따로 전처리할 필요는 없음.

- HistGradientBoostingClassifier : 기본 매개변수에서 안정적인 성능을 얻을 수 있음.
  - max_iter : 반복횟수 지정


```python
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(random_state=42)
scores = cross_validate(hgb, train_input, train_target,
                        return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    /usr/local/lib/python3.7/dist-packages/sklearn/experimental/enable_hist_gradient_boosting.py:17: UserWarning: Since version 1.0, it is not needed to import enable_hist_gradient_boosting anymore. HistGradientBoostingClassifier and HistGradientBoostingRegressor are now stable and can be normally imported from sklearn.ensemble.
      "Since version 1.0, "
    

    0.9321723946453317 0.8801241948619236
    

- 과대적합을 잘 억제하면서 그레이디언트 부스팅보다 조금 더 높은 성능을 제고.
- 특성 중요도 계산 : permutation_importance()
  - 특성을 하나씩 랜덤하게 섞어서 모델의 성능이 변화하는지를 관찰하여 어떤 특성이 중요한지를 계산.
  - n_repeats : 랜덤하게 섞을 횟수 지정
  


```python
from sklearn.inspection import permutation_importance

hgb.fit(train_input, train_target)
result = permutation_importance(hgb, train_input, train_target,
                                n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)
```

    [0.08876275 0.23438522 0.08027708]
    

- permutation_importance()
  - (특성 중요도(importances), 평균(importances_mean), 표준편차(importances_std))


```python
result = permutation_importance(hgb, test_input, test_target,
                                n_repeats=10, random_state=42, n_jobs=-1)
print(result.importances_mean)
```

    [0.05969231 0.20238462 0.049     ]
    


```python
hgb.score(test_input, test_target)
```




    0.8723076923076923



- 회귀 버전 : HistGradientBoostingRegressor

### XGBoost

- 다양한 부스팅 알고리즘 지원
- tree_method = 'hist'로 지정하면 히스토그램 기반 그레이디언트 부스팅 사용가능.


```python
from xgboost import XGBClassifier
xgb = XGBClassifier(tres_method='hist', random_state=42)
scores = cross_validate(xgb, train_input, train_target, return_train_score=True)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.8827690284750664 0.8708899089361072
    

### LightGBM


```python
from lightgbm import LGBMClassifier
lgb = LGBMClassifier(random_state=42)
scores = cross_validate(lgb, train_input, train_target, 
                        return_train_score = True, n_jobs=-1)
print(np.mean(scores['train_score']), np.mean(scores['test_score']))
```

    0.9338079582727165 0.8789710890649293
    

## 마무리

### 키워드로 끝내는 핵심 포인트
- 앙상블 학습 : 더 좋은 예측 결과를 만들기 위해 여러 개의 모델을 훈련하는 머신러닝 알고리즘
- 랜덤 포레스트 : 대표적인 결정 트리 기반의 앙상블 학습 방법
  - 부트스트랩 샘플을 사용하고 랜덤하게 일부 특성을 선택하여 트리를 만드는 것이 특징
- 엑스트라 트리 : 랜덤 포레스트와 비슷하게 결정 트리를 사용하여 앙상블 모델을 만들지만 부트슽랩 샘플을 사용하지 않음.
  - 랜덤하게 노드를 분할해 과대적합을 감소
- 그레이디언트 부스팅 : 랜덤 포레스트나 엑스트라 트리와 달리 결정 트리를 연속저그로 추가하여 손실 함수를 최소화하는 앙상블 방법
  - 훈련 속도가 조금 느리지만 더 좋은 성능 기대
  - 그레이디언트의 속도를 향상 시킨 것이 히스토그램 기반 그레이디언트 부스팅
    - 안정적인 결과와 높은 성능

### 핵심 패키지와 함수


#### scikit-learn


> RandomForestClassifier : 랜덤 포레스트 분류 클래스
  - n_estimators : 앙상블을 구성할 트리의 개수를 지정
    - 기본값 = 100
  - criterion : 불순도 지정
    - 기본값 = 'gini' : 지니 불순도
    - 'entropy' : 엔트로피 불순도
  - max_depth : 트리가 성장할 최대 깊이를 지정
    - 기본값 = None : 리프노드가 순수하거나 min_samples_split보다 샘플 개수가 적을 때까지 성장
  - min_samples_split : 노드를 나누기 위한 최소 샘플 개수
    - 기본값 = 2
  - max_features : 최적의 분할을 위해 탐색할 특성의 개수를 지정
    - 기본값 = 'auto' : 특성 개수의 제곱근
  - bootstrap : 부트스트랩 샘플을 사용할지 지정
    - 기본값 = True
  - oob_score ; OOB 샘플을 사용하여 훈련한 모델을 평가할지 지정
    - 기본값 = False
  - n_jobs : 병렬 실행에 사용할 CPU 코어 수 지정
    - 기본값 = 1: 하나의 코어
    - -1 : 모든 코어

> ExtraTreesClassifier : 엑스트라 트리 분류 클래스
- n_estimator, criterion, max_depth, min_samples_split, max_features -> 랜덤 포레스트와 동일
- boostrap : 부트스트랩 샘플을 사용할지 지정.
  - 기본값 = False
- oob_score : OOB 샘플을 사용하여 훈련한 모델을 평가할지 지정
  - 기본값 = False
- n_jobs : 병렬 실행에 사용할 CPU 코어 수 지정
  - 기본값 = 1: 하나의 코어
  - -1 : 모든 코어

> GradientBoostingClassifier : 그레이디언트 부스팅 분류 클래스
- loss : 손실 함수 지정
  - 기본값 = 'deviance'(로지스틱 손실 함수)
- learning_rate : 트리가 앙상블에 기여하는 정도를 조절
  - 기본값 = 0.1
- n_estimators : 부스팅 단계를 수행하는 트리의 개수
  - 기본값 = 100
- subsample : 사용할 훈련 세트의 샘플 비율을 지정
  - 기본값 = 1.0
- max_depth : 개별 회귀 트리의 최대 깊이
  - 기본값 = 3

> HistGradientBoostingClassifier : 히스토그램 기반 그레이디언트 부스팅 분류 클래스
- learning_rate : 학습률, 감쇠율
  - 기본값 = 0.1
  - 1.0이면 감쇠가 전혀 없음
- max_iter : 부스팅 단계를 수행하는 트리의 개수
  - 기본값 = 100
- amx_bins : 입력 데이터를 나눌 구간의 개수
  - 기본값 = 255
    - 이보다 크게 지정 불가
  - 여기에 1개의 구간이 누락된 값을 위해 추가

# Book : '혼자 공부하는 머신러닝 + 딥러닝', 박해선 지음, 한빛미디어