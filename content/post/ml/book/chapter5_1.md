---
title: "[ML][Book][혼공 머신러닝+딥러닝] ch5-1 결정 트리"
description: ""
date: "2022-04-22T16:31:45+09:00"
thumbnail: ""
categories:
  - "ML"
tags:
  - "Python"
  - "ML"

---
<!--more-->

## 로지스틱 회귀로 와인 분류


```python
import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')
```


```python
wine.head()
```





|   | alcohol | sugar | pH   | class |
|---|---------|-------|------|-------|
| 0 | 9.4     | 1.9   | 3.51 | 0.0   |
| 1 | 9.8     | 2.6   | 3.20 | 0.0   |
| 2 | 9.8     | 2.3   | 3.26 | 0.0   |
| 3 | 9.8     | 1.9   | 3.16 | 0.0   |
| 4 | 9.4     | 1.9   | 3.51 | 0.0   |




- class
  - 0 : 레드 와인
  - 1: 화이트 와인(양성 클래스)
  - 전체 문제에서 화이트와인을 골라내는 문제


```python
wine.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6497 entries, 0 to 6496
    Data columns (total 4 columns):
     #   Column   Non-Null Count  Dtype  
    ---  ------   --------------  -----  
     0   alcohol  6497 non-null   float64
     1   sugar    6497 non-null   float64
     2   pH       6497 non-null   float64
     3   class    6497 non-null   float64
    dtypes: float64(4)
    memory usage: 203.2 KB
    

> 누락된 값이 있는 경우
- 그 데이터를 버리거나 평균값으로 채운다.
- 항상 훈련 세트의 통계값으로 테스트 세트를 변환한다. 
  - 훈련 세트의 평균값으로 테스트 세트의 누락된 값을 채워야 함


```python
wine.describe()
```





|       | alcohol     | sugar       | pH          | class       |
|-------|-------------|-------------|-------------|-------------|
| count | 6497.000000 | 6497.000000 | 6497.000000 | 6497.000000 |
| mean  | 10.491801   | 5.443235    | 3.218501    | 0.753886    |
| std   | 1.192712    | 4.757804    | 0.160787    | 0.430779    |
| min   | 8.000000    | 0.600000    | 2.720000    | 0.000000    |
| 25%   | 9.500000    | 1.800000    | 3.110000    | 1.000000    |
| 50%   | 10.300000   | 3.000000    | 3.210000    | 1.000000    |
| 75%   | 11.300000   | 8.100000    | 3.320000    | 1.000000    |
| max   | 14.900000   | 65.800000   | 4.010000    | 1.000000    |




> 사분위수
- 데이터를 순서대로 4등분한 값
  - ex) 2사분위수(중간값) : 데이터를 일렬로 놓았을 때 정중앙의 값
    - 데이터의 수가 짝수라 중앙값을 선택할 수 없다면 가운데 2개 값의 평균을 사용


```python
data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()
```


```python
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42
)
```


```python
print(train_input.shape, test_input.shape)
```

    (5197, 3) (1300, 3)
    

- StandardScaler 클래스를 사용해 훈련 세트 전처리


```python
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)
```

- 표준점수로 변환된 train_scaled와 test_scaled를 사용해 로지스틱 회귀 모델 훈련


```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target))
print(lr.score(test_scaled, test_target))
```

    0.7808350971714451
    0.7776923076923077
    

### 설명하기 쉬운 모델과 어려운 모델

- 로지스틱 회귀가 학습한 계수와 절편을 출력


```python
print(lr.coef_, lr.intercept_)
```

    [[ 0.51270274  1.6733911  -0.68767781]] [1.81777902]
    

- 정확히 이 숫자가 어떤 의미를 가지는 지 설명하기 어려움.
- 더군다나 다항 특성을 추가한다면 설명하기 더 어려움.


## 결정 트리(Decision Tree)

- 이유를 설명하기 쉬움.
- 데이터를 잘 나눌 수 있는 질문을 찾는 다면 계속 질문을 추가해서 분류 정확도를 높일 수 있다.

> 결정 트리를 만들 때, random_state 지정 이유
- 사이킷런의 결정 트리 알고리즘은 노드에서 최적의 분할을 찾기 전에 특성의 순서를 섞음.
- 약간의 무작위성이 주입되는 데 실행할 때마다 점수가 조금씩 달라질 수 있기 때문.
- 실전에서는 필요하지 않음.


```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target)) # 훈련 세트
print(dt.score(test_scaled, test_target)) # 테스트 세트
```

    0.996921300750433
    0.8592307692307692
    

- 훈련 세트에 대한 점수가 높게 출력.
- 테스트 세트 점수는 훈련 세트에 비해 낮게 출력. -> 과적합된 모델

- plot_tree()함수를 사용해 결정 트리를 이해하기 쉬운 그림으로 출력


```python
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 7))
plot_tree(dt)
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter5_1/output_27_0.png)


- 맨 위의 노드(node) : 루트 노드(root node)
- 맨 아래의 노드 : 리프 노드(leaf node)

> 노드(node)
- 노드는 결정 트리를 구성하는 핵심 요소
- 훈련 데이터의 특성에 대한 테스트를 표현
- ex) 현재 샘플의 당도가 -0.239보다 작거나 같은지 테스트
  - 가지(branch)는 테스트의 결과(True, False)를 나타내며 일반적으로 하나의 노드는 2개의 가지를 가짐.

- plot_tree() 함수에서 깊이를 제한해서 출력
  - max_depth = 1 : 루트 노드를 제외하고 하나의 노드를 더 확장해서 그림
  - filled : 클래스에 맞게 노드의 색을 칠할 수 있음 


```python
plt.figure(figsize=(10, 7))
plot_tree(dt, max_depth=1, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter5_1/output_31_0.png)


- filled = True 로 지정하면 클래스마다 색을 지정하고, 어떤 클래스의 비율이 높아지면 점점 진한 색으로 표시

- 결정 트리에서 예측하는 방법은 간단
  - 리프 노드에서 가장 많은 클래스가 예측 클래스
    - k-최근접 이웃과 비슷
  - 만약 이 결정 트리의 성장을 여기서 멈춘다면 왼쪽 노드에 도달한 샘플과 오른쪽에 도달한 샘플은 모드 양성 클래스로 예측
    - 두 노드 모두 양성 클래스의 개수가 많기 때문

> 만약 결정 트리를 회귀 문제에 적용하면 리프 노드에 도달한 샘플의 타깃을 평균하여 예측값으로 사용
- 사이킷런의 결정 트리 회귀 모델 : DecisionTreeRegressor

### 불순도

- 지니 불순도(gini impurity)를 의미
- DecisionTreeClassifier 클래스의 criterion 매개변수의 기본값 = 'gini'
  - criterion : 노드에서 데이터를 분할할 기준을 정하는 것
- 지니 불순도
  - $$지니\,불순도 = 1 - (음성 \,클래스 \,비율^2+양성\, 클래스 \,비율^2)$$
- 루트 노드 : 총 5,197개의 샘플
  - 음성 클래스 : 1,258
  - 양성 클래스 : 3,939
  - 지니 불순도
    - $$1-((1258/5197)^2+(3939/5197)^2)=0.367$$
- 왼쪽과 오른쪽 노드의 지니 불순도
  - 만약 100개의 샘플
  - 두 클래스의 비율이 정확히 1/2 -> 불순도 = 0.5 -> 최악
  - 지니 불순도
    -$$1-((50/100)^2+(50/100)^2)=0.5$$
  - 노드에 하나의 클래스가 있다면 지님 불순도는 0이 되어 가장 작음.
    - $$1-((0/100)^2+(100/100)^2)=0$$
- 결정 트리의 모델은 부모 노드(parent node)와 자식 노드(child node)의 불순도 차이가 가능한 크도록 트리를 성장
- 부모 노드와 자식 노드의 불순도 차이를 계산
  - 자식 노드의 불순도를 샘플 개수에 비례하여 모두 더함.
  - 그 다음 부모 노드의 불순도에서 뺀다.
  - ex) 루트 노드를 부모 노드, 왼쪽 노드와 오른쪽 노드를 자식 노드
    - 왼쪽 노드 2,922개의 샘플
    - 오른쪽 노드 2,275개의 샘플
    - 불순도의 차이
      - $$부모의\ 불순도\ - (왼쪽\ 노드\ 샘플\ 수 / 부모\ 샘플\ 수)\times 왼쪽\ 노드\ 불순도-(오른쪽\ 노드\ 샘플\, 수/부모\ 샘플수)\times 오른쪽\ 노드\ 불순도 \\ = 0.367 - (2922/5197) \times 0.481- (2275/5197) \times 0.069=0.066$$
      - 이런 부모와 자식 노드 자식 노드의 불순도 차이를 정보 이득(information gain)

- DecisionTreeClassifier 클래스에서 criterion='entropy'를 지정, 엔트로피 불순도를 사용가능
- 엔트로피 불순도 노드의 클래스 비율을 사용, 밑이 2인 로그를 사용하여 곱함.
  - $$-음성\,클래스\,비율\times\log_{2}(음성\,클래스\,비율)-양성\,클래스\,비율\times \log_{2}(양성\,클래스\,비율) \\ =-(1258/5197/)\times \log_{2}(1258/5197)-(3939/5197)\times \log_{2}(3939/5197)=0.798$$
- 지니 불순도와 엔트로피 불순도의 차이는 크지 않음.

### 가지치기
- 가장 간단한 방법 : 트리의 최대 깊이 지정


```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_scaled, train_target)
print(dt.score(train_scaled, train_target))
print(dt.score(test_scaled, test_target))
```

    0.8454877814123533
    0.8415384615384616
    


```python
plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter5_1/output_40_0.png)



```python
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(train_input, train_target)
print(dt.score(train_input, train_target))
print(dt.score(test_input, test_target))
```

    0.8454877814123533
    0.8415384615384616
    


```python
plt.figure(figsize=(20, 15))
plot_tree(dt, filled=True, feature_names=['alcohol', 'sugar', 'pH'])
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter5_1/output_42_0.png)


- 결정 트리는 어떤 특성이 가장 중요한지 나타내는 특성 중요도를 계산
  - 특성 중요도는 결정 트리 모델의 feature_importances\_속성에 저장


```python
print(dt.feature_importances_)
```

    [0.12345626 0.86862934 0.0079144 ]
    

- 두 번째 속성의 중요도가 가장 크다.

## 마무리

### 키워드로 끝내는 핵심 포인트
- 결정 트리: 예 / 아니오에 대한 질무을 이어나가면서 정답을 찾아 학습하는 알고리즘.
  - 비교적 예측 과정을 이해하기 쉽고 성능도 뛰어남.
- 불순도 : 결정 트리가 최적의 질문을 찾기 위한 기준.
  - 지니 불순도와 엔트로피 불순도
- 정보 이득 : 부모 노드와 자식 노드의 불순도 차이
  - 결정 트리 알고리즘은 정보 이득이 최대화되도록 학습
- 가지치기 : 제한 없이 성장하면 훈련 세트에 과대적합되기 쉬움.
  - 결정 트리의 성장을 제한
- 특성 중요도 : 결정 트리에 사용된 특성이 불순도를 감소하는 데 기영한 정도를 나타내는 값
  - 특성 중요도를 계산할 수 있는 것이 결정 트리의 큰 장점

### 핵심 패키지와 함수

> pandas
- info() : 데이터프레임의 요약된 정보를 출력
  - 인덱스와 컬럼 타입을 출력, null이 아닌 값의 개수, 메모리 사용량을 제공
  - verbose : 기본값인 True를 False로 바꾸면 각 열에 대한 정보를 출력하지 않음.
- describe() : 데이터프레임 열의 통계 값을 출력.
  - 수치형일 경우 최소, 최대, 평균, 표준편차, 사분위값 등이 출력
  - 문자열 같은 객체 타입의 열은 가장 자주 등장하는 값과 횟수 등이 출력
  - percentiles : 백분위수 지정
    - 기본값 : [0.25, 0.5, 0.75]

> scikit-learn
- DecisionTreeClassifier : 결정 트리 분류 클래스
  - criterion : 불순도를 지정
    - 'gini' : 지니 불순도
    - 'entropy' : 엔트로피 불순도
  - splitter : 노드를 분할하는 전략 선택
    - 기본값 : None -> 리프 노드가 순수하거나 min_samples\_split보다 샘플 개수가 적어질 때까지 성장
  - min_samples\_split : 노드를 나누기 위한 최소 샘플 개수, 기본값 = 2
  - max_features : 최적의 분할을 위해 탐색할 특성의 개수를 지정, 기본값 = None -> 모든 특성 사용


- plot_tree() : 결정 트리 모델 시각화. 첫 번재 매개변수로 결정 트리 모델 객체를 전달
  - max_depth : 나타낼 트리의 깊이 지정.
    - 기본값 = None -> 모든 노드 출력
  - feature_names : 특성의 이름을 지정
  - filled : True로 지정하면 타깃값에 따라 노드 안에 색을 채움.

# Book : '혼자 공부하는 머신러닝 + 딥러닝', 박해선 지음, 한빛미디어