---
title: "[ML][Book][혼공 머신러닝+딥러닝] ch6-2 k-평균"
description: ""
date: "2022-04-23T16:34:45+09:00"
thumbnail: ""
categories:
  - "ML"
tags:
  - "Python"
  - "ML"

---
<!--more-->

- k-평균(k-means) 군집 알고리즘이 평균값을 자동으로 찾아줌.
- 이 평균값이 클러스터의 중심에 위치, 클러스터 중심(cluster center) 또는 센트로이드(centroid)

## k-평균 알고리즘 소개

- k-평균 알고리즘의 작동 방식
  1. 무작위로 k개의 클러스터 중심을 정함.
  2. 각 샘플에서 가장 가까운 크러스터 중심을 찾아 해당 클러스터의 샘플로 지정.
  3. 클러스터에 속한 샘플의 평균값으로 클러스터 중심을 변경.
  4. 클러스터 중심에 변화가 없을 때까지 2번으로 돌아가 반복.

## KMeans 클래스



```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy
```

    --2022-04-18 03:00:33--  https://bit.ly/fruits_300_data
    Resolving bit.ly (bit.ly)... 67.199.248.11, 67.199.248.10
    Connecting to bit.ly (bit.ly)|67.199.248.11|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://github.com/rickiepark/hg-mldl/raw/master/fruits_300.npy [following]
    --2022-04-18 03:00:33--  https://github.com/rickiepark/hg-mldl/raw/master/fruits_300.npy
    Resolving github.com (github.com)... 140.82.112.3
    Connecting to github.com (github.com)|140.82.112.3|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fruits_300.npy [following]
    --2022-04-18 03:00:34--  https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fruits_300.npy
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3000128 (2.9M) [application/octet-stream]
    Saving to: ‘fruits_300.npy’
    
    fruits_300.npy      100%[===================>]   2.86M  --.-KB/s    in 0.04s   
    
    2022-04-18 03:00:34 (69.1 MB/s) - ‘fruits_300.npy’ saved [3000128/3000128]
    
    

- k-평균 모델을 훈련하기 위해 (샘플 개수, 너비, 높이) 크기의 3차원 배열을 (샘플 개수, 너비X높이) 크기를 가진 2차원 배열로 변경


```python
import numpy as np
fruits = np.load('fruits_300.npy')
fruits_2d = fruits.reshape(-1, 100*100)
```


```python
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, random_state = 42)
km.fit(fruits_2d)
```




    KMeans(n_clusters=3, random_state=42)



- 군집된 결과는 KMeans 클래스 객체의 labels_속성에 저장
- labels_배열의 길이는 샘플 개수와 같음.
- 이 배열은 각 샘플이 어떤 레이블에 해당되는지 나타냄.
- n_clusters=3으로 지정했기 때문에 배열의 값은 0, 1, 2 중 하나.


```python
print(km.labels_)
```

    [2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     2 2 2 2 2 0 2 0 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 2 0 0 2 2 2 2 2 2 2 2 0 2
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 0 2 2 2 2 2 2 2 2 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
     0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
     1 1 1 1]
    


```python
print(np.unique(km.labels_, return_counts=True))
```

    (array([0, 1, 2], dtype=int32), array([111,  98,  91]))
    


```python
import matplotlib.pyplot as plt

def draw_fruits(arr, ratio=1):
  n = len(arr) # 샘플의 개수
  # 한 줄에 10개씩 이미지를 그림.
  # 샘플 개수를 10으로 나누어 전체 행 개수를 계산
  rows = int(np.ceil(n/10))
  # 행이 1개이면 열의 개수는 샘플 개수. 그렇지 않으면 10개
  cols = n if rows < 2 else 10
  
  fig, axs = plt.subplots(rows, cols,
                          figsize=(cols*ratio, rows*ratio), squeeze=False)
  
  for i in range(rows):
    for j in range(cols):
      if i*10 + j < n: # n 개까지 그림.
        axs[i, j].imshow(arr[i*10+j], cmap='gray_r')
      axs[i, j].axis('off')
  plt.show()
```


```python
draw_fruits(fruits[km.labels_==0])
```


![png](/images/self_study_ml_dl_images/chapter6_2/output_14_0.png)



```python
draw_fruits(fruits[km.labels_==1])
```


![png](/images/self_study_ml_dl_images/chapter6_2/output_15_0.png)



```python
draw_fruits(fruits[km.labels_==2])
```


![png](/images/self_study_ml_dl_images/chapter6_2/output_16_0.png)


## 클러스터 중심

- KMeans 클래스가 최종적으로 찾은 클러스터 중심은 cluster_centers_ 속성에 저장


```python
draw_fruits(km.cluster_centers_.reshape(-1, 100, 100), ratio=3)
```


![png](/images/self_study_ml_dl_images/chapter6_2/output_19_0.png)


- transform() : 훈련 데이터 샘플에서 클러스터 중심까지 거리로 변환


```python
print(km.transform(fruits_2d[100:101]))
```

    [[3393.8136117  8837.37750892 5267.70439881]]
    

- 하나의 샘플을 전달했기 때문에 잔화된 배령은 크기가 (1, 클러스터 개수)인 2차원 배열.
- 첫 번째 클러스터(레이블 0), 두 번째 클러스터(레이블 1)가 각각 첫 번째 원소, 두 번째 원소의 값
- 세 번째 클러스터까지의 거리 : 5267.70

- predict() : 가장 가까운 클러스터 중신을 예측 클래스로 출력


```python
print(km.predict(fruits_2d[100:101]))
```

    [0]
    


```python
draw_fruits(fruits[100:101])
```


![png](/images/self_study_ml_dl_images/chapter6_2/output_25_0.png)


- k-평균 알고리즘은 반복적으로 클로스터 중심을 옮기변서 최적의 클러스터를 찾음.
- n_iter_속성에 반복 횟수 저장


```python
print(km.n_iter_)
```

    4
    

## 최적의 k 찾기

- 엘보우(elbow)
  - 이너셔(inertia) : 클러스터 중심과 클러스터에 속한 샘플 사이의 거리를 제곱한 합.
    - 클러스터에 속한 샘플이 얼마나 가깝게 모여 있는 지를 나타내는 값
    - 일반적으로 클러스터 개수가 늘어난면 크러스터 개개의 크기는 줄어들기 때문에 이너셔도 줄어듬.
  - 엘보우 방법은 클러스터 개수를 늘려가면서 이너셔의 변화를 관찰하여 최적의 클러스터 개수를 찾는 방법


```python
inertia = []
for k in range(2, 7):
  km = KMeans(n_clusters=k, random_state=42)
  km.fit(fruits_2d)
  inertia.append(km.inertia_)
plt.plot(range(2, 7), inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter6_2/output_30_0.png)


- 클러스터 개수를 증가 시키면서 이너셔를 그래프로 그리면서 감소라는 속도가 꺽이는 지점
  - 이 지점부터는 클러스터의 개수를 늘려도 크러스터에 잘 밀집된 정도가 크게 개선되지 않음.
  - 이너셔가 크게 줄어들지 않음.
  

## 마무리


### 키워드로 끝내는 핵심 포인트
- k-평균 : 처음에 랜덤하게 클러스터 중심을 정하고 클러스터를 만듦
  - 그 다음 클러스터의 중심을 이동하고 다시 클러스터를 만드는 식으로 반복해서 최적의 클러스터를 구성하는 알고리즘.
- 클러스터 중심 : k-평균 알고리즘이 만든 클로스터에 속한 샘플의 특성 평균값.
  - 센트로이드(centroid)
  - 가장 가까운 클러스터 중심을 샘플의 또 다른 특성으로 사용하거나 새로운 샘플에 대한 예측으로 활용
- 엘보우 방법 : 최적의 클러스터 개수를 정하는 방법 중 하나.
  - 이너션느 클러스터 중신과 샘플 사이 거리의 제곱 합
  - 클러스터 새수에 따라 이너셔 감소가 꺾이는 지점이 적절한 k가 될 수 있음

### 핵심 패키지와 함수

> scikit-learn
- KMeans : k-평균 알고리즘
  - n_clusters : 클러스터 개수 지정. 기본값 = 8
    - 처음에 랜덤하게 세트로이드를 초기화하기 때문에 여러 번 반복하여 이너셔를 기준으로 가장 좋은 결과를 건택
  - n_init : 반복횟수 지정. 기본값 = 10
  - max_iter : k-평균 알고리즘의 한 번 실행에서 최적의 센트로이드를 찾기 위해 반복할 수 있는 최대 횟수. 기본값 = 200

# Book : '혼자 공부하는 머신러닝 + 딥러닝', 박해선 지음, 한빛미디어