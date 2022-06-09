---
title: "[ML][Book][혼공 머신러닝+딥러닝] ch2-2 데이터 전처리"
description: ""
date: "2022-04-22T16:25:45+09:00"
thumbnail: ""
categories:
  - "ML"
tags:
  - "Python"
  - "ML"

---
<!--more-->
## 넘파이로 데이터 준비


```python
fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8, 
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7, 
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]
```


```python
import numpy as np
```

- column_stack() 함수는 전달받은 리스트를 일렬로 세운 다음 차례대로 나란히 연결
- 연결한 리스트는 튜플(tuple)로 전달


```python
np.column_stack(([1, 2, 3], [4, 5, 6]))
```




    array([[1, 4],
           [2, 5],
           [3, 6]])



> 튜플(tuple)
  - 리스트와 비슷
  - 수정 불가능
  - 함수로 전달한 값이 바뀌지 않는다는 것을 믿을 수 있기 때문에 매개변수 값으로 많이 사용


```python
fish_data = np.column_stack((fish_length, fish_weight))
```


```python
print(fish_data[:5])
```

    [[ 25.4 242. ]
     [ 26.3 290. ]
     [ 26.5 340. ]
     [ 29.  363. ]
     [ 29.  430. ]]
    

- np.ones()와 np.zeros() 함수 -> 각각 원하는 개수의 1과 0을 채운 배열을 만들어줌


```python
print(np.ones(5))
```

    [1. 1. 1. 1. 1.]
    

- np.concatenate() : 첫 번째 차원을 따라 배열을 연결
  - ex) 1 1 1,  0 0 0
  - np.colum_stack()


```
1 0
1 0
1 0
```
  - np.concatenate()


```
1 1 1 0 0 0
```


```python
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
```


```python
print(fish_target)
```

    [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
     1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
     0.]
    

- 데이터가 클수록 파이썬 리스트는 비효율적, 넘파이 배열을 사용하는 것이 좋음

## 사이킷런으로 훈련 세트와 테스트 세트 나누기
- train_test_split()


```python
from sklearn.model_selection import train_test_split
```


```python
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, random_state=42)
```


```python
print(train_input.shape, test_input.shape)
```

    (36, 2) (13, 2)
    


```python
print(test_input.shape, test_target.shape)
```

    (13, 2) (13,)
    

- 넘파이 배열의 크기는 파이썬의 튜플로 표현
- 튜플의 원소가 하마념 원소 위에 콤마를 추가


```python
print(test_target)
```

    [1. 0. 0. 0. 1. 1. 1. 1. 1. 1. 1. 1. 1.]
    

- train_test_split()의 매개변수 stratify -> 클래스 비율에 맞게 데이터를 나눔
- 훈련 데이터가 작거나 특정 클래스의 샘플 수가 적을 때 유용


```python
train_input, test_input, train_target, test_target = train_test_split(fish_data, fish_target, stratify = fish_target, random_state=42)
```


```python
print(test_target)
```

    [0. 0. 1. 0. 1. 0. 1. 1. 1. 1. 1. 1. 1.]
    

## 수상한 도미 한 마리
- k-최근접 이웃 훈련


```python
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
```




    1.0




```python
print(kn.predict([[25, 150]])) # 도미 데이터 예측 -> 결과: 빙어
```

    [0.]
    


```python
import matplotlib.pyplot as plt

plt.scatter(train_input[:,0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter2_2/output_29_0.png)


- k-최근접 이웃은 주변의 샘플 중에서 다수인 클래스를 예측으로 사용
- 주어진 샘플에서 가장 가까운 이웃을 찾아주는 kneighbor() 메서드
- 이웃까지의 거리와 이웃 샘플의 인덱스를 반환
- KNeighborsClassifier 클래스의 이웃 개수인 n_neighbors의 기본 값은 5 -> 5개의 이웃이 반환


```python
distance, indexes = kn.kneighbors([[25, 150]])
```


```python
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter2_2/output_32_0.png)



```python
print(train_input[indexes])
```

    [[[ 25.4 242. ]
      [ 15.   19.9]
      [ 14.3  19.7]
      [ 13.   12.2]
      [ 12.2  12.2]]]
    


```python
print(train_target[indexes])
```

    [[1. 0. 0. 0. 0.]]
    


```python
# kenighbors()에서 반환한 distances 배열을 출력
# 이 배열에는 이웃 샘플까지의 거리가 담겨져 있음
print(distance)
```

    [[ 92.00086956 130.48375378 130.73859415 138.32150953 138.39320793]]
    

## 기준을 맞춰라


```python
plt.scatter(train_input[:,0], train_input[:,1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlim((0, 1000))   # x축 범위 지정
                      # y축 범위 지정 시 ylim()
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter2_2/output_37_0.png)


- 두 특성의 값이 놓인 범위가 매우 다름. -> 스케일(scale)이 다름
- 데이터를 표현하는 기준이 다르면 알고리즘이 올바르게 예측 불가
- 특성값을 일정한 기준으로 맞춰야함 -> 데이터 전처리(data preprocessing)

- 표준점수(standard score, z 점수) : 가장 널리 사용하는 방법 중 하나. 표준점수는 각 특성값이 평균에서 표준편차의 몇 배만큼 떨어져 있는 지를 나타냄. 
- 이를 통해 실제 특성값의 크기와 상관없이 동일한 조건으로 비교 가능

> 표준점수와 표준편차
  - 분산은 데이터에서 평균을 뺀 값을 모두 제곱한 다음 평균을 내어 구함
  - 표준편차는 분산의 제곱근으로 데이터가 분산된 정도
  - 표준점수는 각 데이터가 원점에서 몇 표준편차만큼 떨어져 있는지를 나타내는 값


```python
mean = np.mean(train_input, axis = 0)
std = np.std(train_input, axis = 0)
```

- np.mean() : 평균 계산
- np.std() : 표준편차 계산
- 특성마다 값의 스케일이 다르므로 평균과 표준편차는 각 특성별로 계산 -> axis = 0 -> 행을 따라 각 열의 통계 값을 계산


```python
print(mean, std)
```

    [ 27.29722222 454.09722222] [  9.98244253 323.29893931]
    

- 원본 데이터에서 평균을 빼고 표준편차로 나누어 표준점수로 변환


```python
train_scaled = (train_input - mean) / std
```

- train_input의 모든 행에서 위 과정을 전부 적용 -> 넘파이의 브로드캐스팅(broadcasting) 
- 브로드캐스팅은 넘파이 배열 사이에서 일어남
- train_input, mean, std -> 넘파이 배열

## 전처리 데이터로 모델 훈련


```python
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(25, 150, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter2_2/output_48_0.png)



```python
new = ([25, 150] - mean) / std
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter2_2/output_49_0.png)



```python
kn.fit(train_scaled, train_target)
```




    KNeighborsClassifier()



- 훈련 후 테스트 세트를 평가할 때는 훈련 세트의 기준으로 테스트 세트를 변환해야 같은 스케일로 산점도를 그릴 수 있다.


```python
test_scaled = (test_input - mean) / std
```


```python
kn.score(test_scaled, test_target)
```




    1.0




```python
print(kn.predict([new]))
```

    [1.]
    


```python
# 특성을 표준점수로 변환 -> 산점도
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:,0], train_scaled[:,1])
plt.scatter(new[0], new[1], marker='^')
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter2_2/output_55_0.png)


## 마무리

### 키워드로 끝나는 핵심 포인트
  - 데이터 전처리 : 머신러닝 모델에 훈련 데이터를 주입하기 전에 가공하는 단계. 많은 시간 소모
  - 표준점수 : 훈련 세트의 스케일을 바꾸는 대표적인 방법 중 하나. 표준점수를 얻으려면 특성의 평균을 빼고 표준편차로 나눔. 반드시 훈련 세트의 평균과 표준편차로 테스트 세트를 바꿔야 함
  - 브로드캐스팅 : 크기가 다른 넘파이 배열에서 자동으로 사칙 연산을 모든 행이나 열로 확장하여 수행하는 기능
 

### 핵심 패키지와 함수
> scikit-learn
  - train_test_split() : 훈련 데이터를 훈련 세트와 테스트 세트로 나누는 함수. 여러 개의 배열을 전달 가능.    
    - 테스트 세트로 나눌 비율은 test_size 매개변수에서 지정할 수 있으며 기본값은 0.25   
    - shuffle 매개변수로 무작위로 섞을지 여부를 결정 가능. 기본값은 True.   
    - stratify 매개변수에 클래스에 레이블이 담긴 배열(일반적으로 타깃 데이터)을 전달하면 클래스 비율에 맞게 운련 세트와 테스트 세트로 나눔   
  - kneighbors() : k-최근접 이웃 객체의 메서드. 입력한 데이터레 가장 가까운 이웃을 찾아 거리와 이웃 샘플 인덱스를 반환. 
    - 기본적으로 이웃의 개수는 KNeighborsClassifier 클래스의 객체를 생성할 때 지정한 개수를 사용. n_neighbors 매개변수에서 다르게 지정 가능 
    - return_distance 매개변수를 False로 지정하면 이웃 샘플릐 인덱스만 반환하고 거리는 반환하지 않는다. 기본값 = True

# Book : '혼자 공부하는 머신러닝 + 딥러닝', 박해선 지음, 한빛미디어