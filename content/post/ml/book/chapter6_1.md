---
title: "[ML][Book][혼공 머신러닝+딥러닝] ch6-1 군집 알고리즘"
description: ""
date: "2022-04-23T16:33:45+09:00"
thumbnail: ""
categories:
  - "ML"
tags:
  - "Python"
  - "ML"

---
<!--more-->

## 타깃을 모르는 비지도 학숩

- 타깃이 없을 때 사용하는 머신러닝 알고리즘 : 비지도 학습(unsupervised learning)

## 과일 사진 데이터 준비

- 사과, 바나나, 파인애플 사진
- 흑백 사진
- 넘파이 배열의 기본 저장 포맷인 npy 파일로 저장


```python
!wget https://bit.ly/fruits_300_data -O fruits_300.npy
```

    --2022-04-18 00:57:45--  https://bit.ly/fruits_300_data
    Resolving bit.ly (bit.ly)... 67.199.248.10, 67.199.248.11
    Connecting to bit.ly (bit.ly)|67.199.248.10|:443... connected.
    HTTP request sent, awaiting response... 301 Moved Permanently
    Location: https://github.com/rickiepark/hg-mldl/raw/master/fruits_300.npy [following]
    --2022-04-18 00:57:45--  https://github.com/rickiepark/hg-mldl/raw/master/fruits_300.npy
    Resolving github.com (github.com)... 140.82.121.4
    Connecting to github.com (github.com)|140.82.121.4|:443... connected.
    HTTP request sent, awaiting response... 302 Found
    Location: https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fruits_300.npy [following]
    --2022-04-18 00:57:45--  https://raw.githubusercontent.com/rickiepark/hg-mldl/master/fruits_300.npy
    Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.111.133, 185.199.110.133, ...
    Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 3000128 (2.9M) [application/octet-stream]
    Saving to: ‘fruits_300.npy’
    
    fruits_300.npy      100%[===================>]   2.86M  --.-KB/s    in 0.03s   
    
    2022-04-18 00:57:46 (92.4 MB/s) - ‘fruits_300.npy’ saved [3000128/3000128]
    
    

> !
- 리눅스 셀(shell) 명령으로 이해.
- wget : 원격 주소에서 데이터를 다운로드하여 저장
- -O : 저장할 파일 이름응 지정


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
fruits = np.load('fruits_300.npy')
```


```python
print(fruits.shape)
```

    (300, 100, 100)
    

- 첫 번째 차원 (300) : 샘플의 개수
- 두 번째 차원 (100) : 이미지 높이
- 세 번째 차원 (100) : 이미지 너비
  - 이미지 크기 : 100 X 100
  - 배열의 크기 : 100 X 100


```python
print(fruits[0, 0, :])
```

    [  1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   2   1
       2   2   2   2   2   2   1   1   1   1   1   1   1   1   2   3   2   1
       2   1   1   1   1   2   1   3   2   1   3   1   4   1   2   5   5   5
      19 148 192 117  28   1   1   2   1   4   1   1   3   1   1   1   1   1
       2   2   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1
       1   1   1   1   1   1   1   1   1   1]
    

- 첫 번째 행에 잇는 픽셀 100개에 들어 있는 값을 출력
- 이 넘파이 배열은 흑백 사진 : 0 ~ 255까지의 정숫값을 가짐.

- matplotlib의 imshow() 함수를 사용하면 넘파이 배열로 저장된 이미지를 쉽게 그릴 수 있음.
- 흑백 이미지이므로 cmap 매개변수 값은 'gray'로 지정


```python
plt.imshow(fruits[0], cmap='gray')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter6_1/output_15_0.png)


- 0에 가까울수록 검게 나타나고 높은 값은 밝게 표시됨.
- 이 흑백 이미지는 사진으로 찍은 이미지를 넘파이 배열로 변환할 때 반전시킨 것.


> 컴퓨터는 왜 255에 가까운 바탕에 집중하나?
- 알고리즘이 어떤 출력을 만들기 위해 곱셈, 덧셈을 수행
- 픽셀 값이 0이면 출력도 0이 되어 의미가 없음.
- 픽셀 값이 높으면 출력 값도 커지기 때문에 의미를 부여하기 좋음.

- 종종 흑백 이미지를 반전 시켜 사용.
- cmap을 'gray_r'로 지정하여 다시 반전


```python
plt.imshow(fruits[0], cmap='gray_r')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter6_1/output_19_0.png)


- 이 그림에서 밝은 부분이 0에 가까고 짙은 부분이 255에 가까운 값


```python
fig, axs = plt.subplots(1, 2)
axs[0].imshow(fruits[100], cmap='gray_r')
axs[1].imshow(fruits[200], cmap='gray_r')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter6_1/output_21_0.png)


## 픽셀 값 분석

- 넘파이 배열을 나눌 때 100 X 100 이미지를 펼쳐서 길이가 10,000인 1차원 배열로 만듦.


```python
# 첫 번째 차원을 -1로 지정하면 자동으로 남은 차원을 할당
apple = fruits[0:100].reshape(-1, 100*100)
pineapple = fruits[100:200].reshape(-1, 100*100)
banana = fruits[200:300].reshape(-1, 100*100)
```


```python
print(apple.shape)
```

    (100, 10000)
    

- 샘플의 픽셀 평균값




```python
print(apple.mean(axis=1))
```

    [ 88.3346  97.9249  87.3709  98.3703  92.8705  82.6439  94.4244  95.5999
      90.681   81.6226  87.0578  95.0745  93.8416  87.017   97.5078  87.2019
      88.9827 100.9158  92.7823 100.9184 104.9854  88.674   99.5643  97.2495
      94.1179  92.1935  95.1671  93.3322 102.8967  94.6695  90.5285  89.0744
      97.7641  97.2938 100.7564  90.5236 100.2542  85.8452  96.4615  97.1492
      90.711  102.3193  87.1629  89.8751  86.7327  86.3991  95.2865  89.1709
      96.8163  91.6604  96.1065  99.6829  94.9718  87.4812  89.2596  89.5268
      93.799   97.3983  87.151   97.825  103.22    94.4239  83.6657  83.5159
     102.8453  87.0379  91.2742 100.4848  93.8388  90.8568  97.4616  97.5022
      82.446   87.1789  96.9206  90.3135  90.565   97.6538  98.0919  93.6252
      87.3867  84.7073  89.1135  86.7646  88.7301  86.643   96.7323  97.2604
      81.9424  87.1687  97.2066  83.4712  95.9781  91.8096  98.4086 100.7823
     101.556  100.7027  91.6098  88.8976]
    


```python
plt.hist(np.mean(apple, axis=1), alpha=0.8)
plt.hist(np.mean(pineapple, axis=1), alpha=0.8)
plt.hist(np.mean(banana, axis=1), alpha=0.8)
plt.legend(['apple', 'pineappele', 'banana'])
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter6_1/output_28_0.png)


- 전체 샘플에 대해 픽셀의 평균을 계산
- 픽셀의 평균을 계산 (axis=0)


```python
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].bar(range(10000), np.mean(apple, axis=0))
axs[1].bar(range(10000), np.mean(pineapple, axis=0))
axs[2].bar(range(10000), np.mean(banana, axis=0))
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter6_1/output_30_0.png)


- 픽셀 평균값은 100X100 크기로 바꿔서 이미지처럼 출력하여 위 그래프와 비교
- 픽셀을 평균 낸 이미지를 모든 사진을 합쳐 놓은 대표 이미지


```python
apple_mean = np.mean(apple, axis=0).reshape(100, 100)
pineapple_mean = np.mean(pineapple, axis = 0).reshape(100, 100)
banana_mean = np.mean(banana, axis = 0).reshape(100, 100)
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].imshow(apple_mean, cmap='gray_r')
axs[1].imshow(pineapple_mean, cmap='gray_r')
axs[2].imshow(banana_mean, cmap='gray_r')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter6_1/output_32_0.png)


## 평균값과 가까운 사진 고르기

- 사과 사진의 평균값인 apple_mean과 가장 가짜운 사진을 고름.
- fruits 배열에 있는 모든 샘플에서 apple_mean을 뺀 절댓값의 평균을 계산.


```python
abs_diff = np.abs(fruits - apple_mean)
abs_mean = np.mean(abs_diff, axis=(1, 2))
print(abs_mean.shape)
```

    (300,)
    

- apple_mean과 오차가 가장 작은 샘플 100개
- np.argsort() -> 작은 것에서 큰 것으로 정렬


```python
apple_index = np.argsort(abs_mean)[:100]
fig, axs = plt.subplots(10, 10, figsize=(10, 10))
for i in range(10):
  for j in range(10):
    axs[i, j].imshow(fruits[apple_index[i*10 + j]], cmap='gray_r')
    axs[i, j].axis('off')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter6_1/output_37_0.png)


- 군집(clustering) : 비슷한 샘플까리 그룹으로 모으는 작업
- 클러스터(cluster) : 군집 알고리즘에서 만든 그룹

## 마무리

### 키워드로 끝내는 핵심 포인트
- 비지도 학습 : 머신러닝의 한 종류로 훈련 데이터에 타깃이 없음.
  - 타깃이 없기 때문에 외부의 도움 없이 스스로 유용한 무언가를 학습해야함.
  - 대표적으로 군집, 차원축소
- 히스토그램 : 구간별로 갑싱 발생한 빈도를 그래프로 표현.
  - 보통 x축 값의 구간(계급)이고 y축은 발생 빈도(도수)
- 군집 : 비슷한 샘플끼리 하나의 그룹으로 모으는 대표적인 비지도 학습 작업.
  - 군집 알고리즘으로 모은 샘플 그룹을 클러스터

# Book : '혼자 공부하는 머신러닝 + 딥러닝', 박해선 지음, 한빛미디어