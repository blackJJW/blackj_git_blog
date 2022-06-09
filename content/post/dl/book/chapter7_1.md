---
title: "[DL][Book][혼공 머신러닝+딥러닝] ch7-1 인공 신경망"
description: ""
date: "2022-04-23T16:50:45+09:00"
thumbnail: ""
categories:
  - "DL"
tags:
  - "Python"
  - "DL"

---
<!--more-->

## 패션 MNIST


```python
from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    32768/29515 [=================================] - 0s 0us/step
    40960/29515 [=========================================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26427392/26421880 [==============================] - 0s 0us/step
    26435584/26421880 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    16384/5148 [===============================================================================================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4423680/4422102 [==============================] - 0s 0us/step
    4431872/4422102 [==============================] - 0s 0us/step
    


```python
print(train_input.shape, train_target.shape)
```

    (60000, 28, 28) (60000,)
    


```python
print(test_input.shape, test_target.shape)
```

    (10000, 28, 28) (10000,)
    


```python
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 10, figsize = (10, 10))
for i in range(10):
  axs[i].imshow(train_input[i], cmap='gray_r')
  axs[i].axis('off')
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter7_1/output_6_0.png)



```python
print([train_target[i] for i in range(10)])
```

    [9, 0, 0, 3, 0, 2, 7, 2, 5, 5]
    


```python
import numpy as np
print(np.unique(train_target, return_counts=True))
```

    (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000]))
    

## 로지스틱 회귀로 패션 아이템 분류

- SDGClassifier는 2차원 입력을 다루지 못하기 때문에 각 샘플을 1차원 배열로 만들어야 한다. 



```python
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
```


```python
print(train_scaled.shape)
```

    (60000, 784)
    


```python
from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log', max_iter=5, random_state=42)
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))
```

    0.8195666666666668
    

## 인공 신경망

- 출력층(output layer) : 신경망의 최종 값을 만듦
- 뉴런(neuron) : z 값을 계산하는 단위
  - 뉴런에서 일어나는 일은 선형 계산이 전부
  - 뉴런이라는 표현 대신 유닛(unit)이라고 부르는 사람이 많아짐.
- 입력층(input layer) 

> 딥러닝
- 인공 신경망과 거의 동의어로 사용되는 경우가 많음.
- 혹은 심층 신경망(deep neural network, DNN)을 딥러닝이라고 부름
  - 여러 개의 층을 가진 신경망

### 텐서플로와 케라스


```python
import tensorflow as tf
```

- 딥러닝 라이브러리가 다른 머신러닝 라이브러리와 다른 점 중 하나는 그래픽 처리 장치인 GPU를 사용하여 인공 신경망을 훈련.
- GPU는 벡터와 행렬 연산에 매우 최적화되어 있기 때문에 곱셈과 덧셈이 많이 수행되는 인공 신경망에 큰 도움.

- 케라스 라이브러리는 직접 GPU 연산을 수행하지 않음. 대신 GPU 연산을 수행하는 다른 라이브러리를 백엔드(backend)로 사용
  - ex) 텐서플로가 케라스의 벡엔드 중 하나.
  - 이런 케라스를 멀티-백엔드 케라스라고 부름.

- 텐서플로에서 케라스를 사용하려면 다음과 같이 import


```python
from tensorflow import keras
```

## 인공 신경망으로 모델 만들기


```python
from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
```


```python
print(train_scaled.shape, train_target.shape)
```

    (48000, 784) (48000,)
    


```python
print(val_scaled.shape, val_target.shape)
```

    (12000, 784) (12000,)
    

- 케라스의 레이어(keras.layer) 패키지 안에는 다양한 층이 준비
- 밀집층(dense layer) : 가장 기본이 됨
- 완전 연결층(fully connected layer) : 양쪽의 뉴련이 모두 연결하고 있음.

- Dense 클래스
  - 매개변수 : 뉴런 개수, 뉴련의 출력에 적용할 함수, 입력의 크기


```python
dense = keras.layers.Dense(10, activation='softmax', input_shape=(784, ))
```

- 10개의 패션 아이템 분류 -> 뉴런 개수 10개
- 10개의 뉴런에서 출력되는 값을 확률로 바꾸기 위해서는 softmax함수 사용.
  - 만약 2개의 클래스를 분류하는 이진 분류라면 시그모이드 함수를 사용하기 위해 'sigmoid'로 설정.

- 이 밀집층이 가진 신결망 모델을 생성
- Sequential 클래스 사용


```python
model = keras.Sequential(dense)
```

## 인공 신경망으로 패션 아이템 분류


```python
model.compile(loss='sparse_categorical_crossentropy', metrics ='accuracy')
```

- 이진 분류 loss = 'binary_crossentropy'
- 다중 분류 loss = 'categorical_crossentropy'MBM

- 타깃값응 해당 클래스만 1이고 나머지는 모두 0인 배열로 만드는 것 원-핫 인코딩(one-hot encoding)
- 따라서 다중 분루에서 크로스 엔트로피 손실 함수를 사용하려면 0, 1, 2와 같이 정수로 된 타깃값을 원-핫 인코딩으로 변환해야 함.


```python
print(train_target[:10])
```

    [7 3 5 8 6 9 3 3 9 9]
    

- 'sparse_categorical_crossentropy' : 정수로된 타깃값을 사용해 크로스 엔트로피 손실을 계산하는 것

- 타깃를 원-핫 인코딩으로 준비했다면 compile()에 손실 함수를 loss='categorical_crossentropy'로 지정.

- compile()의 두 번째 매개변수인 metrics
  - 케라스는 모델이 훈련할 때 기본으로 에포크마다 손실 값을 출력.
  - 정확도를 함께 출력하기 위해 'accuracy'


```python
model.fit(train_scaled, train_target, epochs=5)
```

    Epoch 1/5
    1500/1500 [==============================] - 5s 2ms/step - loss: 0.6093 - accuracy: 0.7943
    Epoch 2/5
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.4785 - accuracy: 0.8399
    Epoch 3/5
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.4556 - accuracy: 0.8472
    Epoch 4/5
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.4448 - accuracy: 0.8511
    Epoch 5/5
    1500/1500 [==============================] - 3s 2ms/step - loss: 0.4363 - accuracy: 0.8555
    




    <keras.callbacks.History at 0x7f2630111f50>




```python
model.evaluate(val_scaled, val_target)
```

    375/375 [==============================] - 1s 2ms/step - loss: 0.4484 - accuracy: 0.8509
    




    [0.44841471314430237, 0.8509166836738586]



- 검증 세트의 점수는 훈련 세트 점수보다 조금 낮은 것이 일반적

## 마무리

### 키워드로 끝내는 핵심 포인트

- 인공 신경망 : 생물학적 뉴런네서 영감을 받아 만든 머신러닝 알고리즘. 딥러닝
- 텐서플로 : CPU, GPU 이용. 인공 신경망 모델을 효울적으로 하며 모델 구축과 서비스에 필요한 다양한 도구를 제공.
- 밀집층 : 가장 간단한 인공 신경망의 층
  - 양쪽 모두 연결 되어 있으면 ' 완전 연결 층'
  - 특별히 출력층에 밀집층을 사욯할 째는 분류하려는 클래스와 동일한 개수의 뉴런을 사용.
- 원-핫 인코딩 : 정숫값을 배열에서 해당 정수 위치의 원소만 1이고 나머지는 모두 0으로 변환.
  - 이런 변환이 필요한 이유는 다중 분류에서 출력층에서 만든 확률과 크로스 엔트로피 손실을 계산하기 위해.
- sparse_categorical_entropy 손실을 지정하면 이런 변환을 수행할 필요가 없음.

### 핵심 패키지와 함수

> TensorFlow
- Dense : 신경망에서 가장 기본 층인 밀집층을 만드는 클래스
  - activition : 사용할 활성화 함수를 지젖
    - sigmaoid, softmatx
  - 아무것도 지정하지 않으면 활성화 함수를 사용하지 않음
  -  Sequential : 맨 처음 추가되는 층에는 input_shape 입력 크기를 지정
- Sequential : 케라스에서 신경망 모델을 만드는 클래스
  - 이 클래스의 객체를 생성할 때 신경만 모델에 추가할 층을 지정 가능.
  - 추가할 층이 1개 이상일 경우 파이썬 리스트로 전달.
- compoile() : 모델 객체를 만든 후 훈려하기 전에 사용할 손실 함수와 측정 지표 등을 지정.
  - loss = 손실함수를 지정
    - 이진 분류 : 'binary_crossentropy'
    - 다중 분류 : 'sparse_categorical_crossentropy
  - 클래스 레이블이 정수일 경우 'sparse_categorical_crossentropy'로 지정.
  - 회귀 모델일 경우 'mean_square_error' 등으로 지정
- fit() : 모델을 훈련하는 메서드
  -  첫 번째와 두 번째 매개변수에 입려과 타긱 데이터를 전달
  - epochs : 에포크 횟수
- evaluate() : 모델의 성능을 평가
- 첫 번째와 두 번째 에 입려과 타깃으로 전달
- compile() 메서드에서 loss 매개변수에 지정한 손실 핢수 값과 metric 매개변수에서 지정한 측정 지표를 출력

# Book : '혼자 공부하는 머신러닝 + 딥러닝', 박해선 지음, 한빛미디어