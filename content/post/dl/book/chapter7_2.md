---
title: "[DL][Book][혼공 머신러닝+딥러닝] ch7-2 심층 신경망"
description: ""
date: "2022-04-23T16:51:45+09:00"
thumbnail: ""
categories:
  - "DL"
tags:
  - "Python"
  - "DL"

---
<!--more-->

## 2개의 층


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
    

- 이미지의 픽셀값을 0 ~ 255 범위에서 0 ~ 1 사이로 변환하고, 28 X 28 크기의 2차원 배열을 784 크기의 1차원 배열로 펼침.


```python
from sklearn.model_selection import train_test_split

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size=0.2, random_state=42)
```

- 은닉층(hidden layer) : 입력층과 출력층 사이에 있는 모든 층
- 활성화 함수 : 신경망 층의 선형 방정식의 계산 값에 적용하는 함수.
  - 출력층에 적용하는 활성화 함수는 종류가 제한되어 있음.
  - 이진 분류일 경우 시그모이드 함수
  - 다중 분류일 경우 소프트맥스 함수
- 은닉층의 활성화 함수는 비교적 자유로움.
  - 시그모이드 함수, 렐루(ReLU) 함수 등


```python
dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784, ))
dense2 = keras.layers.Dense(10, activation='softmax')
```

## 심층 신경망 만들기




- dense1과 dense2 객체를 Sequential 클래스에 추가하여 심층 신경망(deep neural network, DNN) 생성


```python
model = keras.Sequential([dense1, dense2])
```


```python
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense (Dense)               (None, 100)               78500     
                                                                     
     dense_1 (Dense)             (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    _________________________________________________________________
    

- 맨 첫 줄에 모델의 이름이 출력. 그 다음 이 모델에 들어 있는 층이 순서대로 나열.
  - 이 순서는 맨 처음 추가한 은닉층에서 출력층의 순서로 나열
- 층마다 층 이름, 클래스, 출력 크기, 모델 파라미터 개수가 출력.
  - 층을 만들 때 name 매개변수로 이름을 지정가능.
  - 층 이름을 지정하지 않으면 케라스가 자동으로 'dense'라고 이름을 붙임.
- 출력을 보면 (None, 100)
  - 첫 번째 차원은 샘플의 개수
    - 샘플 개수가 지정되지 않았기 때문에 None
    - 케사스 모델에서 fit()을 수행할 때 미니배치 경사 하강법 사용
      - 샘플 개수를 고정하지 않고 어떤 배치 크기에도 유연하게 대응할 수 있도록 None으로 지정
    - 신경망 층에 입력되거나 출력되는 배열의 첫 번째 차원을 배치 차원이라 부름.
  - 두 번째 차원
    - 은닉층의 뉴런 개수를 100로 지정, 100개의 출력 생성.
      - 샘플마다 784개의 픽셀값이 은닉층을 통과하면서 100개의 특성으로 압축됨.
- 마지막으로 모델 파라미터 개수가 출력.


## 층을 추가하는 방법

- Sequential 클래스의 생성자 안에서 바로 Dense 클래스의 객체를 만드는 경우가 많음.


```python
model = keras.Sequential([
                          keras.layers.Dense(100, activation='sigmoid', input_shape=(784,), name='hidden'),
                          keras.layers.Dense(10, activation='softmax', name='output')
                          ], name = '패션 MNIST 모델')
```


```python
model.summary()
```

    Model: "패션 MNIST 모델"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     hidden (Dense)              (None, 100)               78500     
                                                                     
     output (Dense)              (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    _________________________________________________________________
    

- add() 메서드를 이용한 층 추가


```python
model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784, )))
model.add(keras.layers.Dense(10, activation='softmax'))
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     dense_2 (Dense)             (None, 100)               78500     
                                                                     
     dense_3 (Dense)             (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    _________________________________________________________________
    


```python
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
```

    Epoch 1/5
    1500/1500 [==============================] - 8s 3ms/step - loss: 0.5638 - accuracy: 0.8071
    Epoch 2/5
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.4084 - accuracy: 0.8539
    Epoch 3/5
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.3742 - accuracy: 0.8638
    Epoch 4/5
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.3516 - accuracy: 0.8729
    Epoch 5/5
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.3331 - accuracy: 0.8783
    




    <keras.callbacks.History at 0x7fa78fde4490>



## 렐루 함수

- 렐루(ReLU) 함수
  - 아주 간단.
  - 입력이 양수일 경우 마치 활성화 함수가 없는 것처럼 그냥 입력을 통과시키고 음수일 경우에는 0으로 만듦.
  - $max(0, z)$ : z가 0보다 크면 z를 출력하고 z가 0보다 작으면 0을 출력
  - 이미지 처리에서 좋은 성능


- Flatten 클래스
  - 배치 차원을 제외하고 나머지 입력 차원을 모두 일렬로 펼치는 역할.
  - 입력에 곱해지는 가중치나 절편이 없음.
    - 인공 신경망의 성능을 위해 기여하는 바는 없음.
  - Flatten 클래스를 층처럼 입력층과 은닉층 사이에 추가하기 때문에 이를 층이라 부름.
  - Flatten 층은 입력층 다음에 추가.


```python
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
```


```python
model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_1 (Flatten)         (None, 784)               0         
                                                                     
     dense_4 (Dense)             (None, 100)               78500     
                                                                     
     dense_5 (Dense)             (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 79,510
    Trainable params: 79,510
    Non-trainable params: 0
    _________________________________________________________________
    

- 첫 번째 등장하는 Flatten 클래스에 포함된 모델 파라미터는 0개.
- 케라스의 Flatten 층을 신경망 모델에 추가하면 입력값의 차원을 짐작 가능.



```python
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
train_scaled = train_input / 255.0
train_scaled, val_scaled, train_target, val_target = train_test_split(train_scaled, train_target, test_size = 0.2, random_state = 42)
```


```python
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
```

    Epoch 1/5
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.5293 - accuracy: 0.8148
    Epoch 2/5
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.3936 - accuracy: 0.8585
    Epoch 3/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3569 - accuracy: 0.8721
    Epoch 4/5
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.3342 - accuracy: 0.8789
    Epoch 5/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3184 - accuracy: 0.8857
    




    <keras.callbacks.History at 0x7fa78fc0c3d0>




```python
model.evaluate(val_scaled, val_target)
```

    375/375 [==============================] - 2s 4ms/step - loss: 0.3602 - accuracy: 0.8755
    




    [0.3602057695388794, 0.8755000233650208]



## 옵티마이저

- 은닉층, 활성화 함수, 층의 종류, batch_size, epochs 등 -> 하이퍼 파라미터 

- compile() 메서드에서는 케라스의 기본 경사 하강법 알고리즘인 RMSprop을 사용.
  - 케라스는 다양한 종류의 경사 하강법 알고리즘을 제공.
    - 옵티마이저(optimizer)
    

- 가장 기본적인 optimizer -> 확률적 경사 하강법(SGD)


```python
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')
```


```python
# 위 코드와 동일

sgd = keras.optimizers.SGD()
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')
```

- 만약 SGD 클래스이 학습률 기본값이 0.01일 때 이를 바꾸고 싶다면 learning_rate를 지정.


```python
sgd = keras.optimizers.SGD(learning_rate=0.1)
```

- 기본 경사 하강법 옵티마이저

SGD(learing_rate=0.01) -> 모멘텀(momentum > 0) -> 네스테로프 모멘텀(nesterov = True)

- 적응적 학습률 옵티마이저

RMSprop -> Adam

Adagrad

- SGD 클래스의 momentum 매개변수의 기본값은 0
  - 이를 0보다 큰 값으로 지정하면 마치 이전의 그레이디언트를 가속도처럼 사용하는 모멘텀 최적화(momentum optimization)를 사용.
    - 보통 momentum은 0.9이상을 지정
- SGD 클랫의 nesterov 매개변수를 False에서 True로 바꾸면 네스테로프 모멘텀 최적화(nesterov momentum optimizer 또는 네스테로프 가속 경사)를 사용


```python
sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)
```

- 네스테로프 모멘텀은 모멘텀 최적화를 2번 반복하여 구현.
- 대부분의 경우 네스테로프 모멘텀 최적화가 기본 확률적 경사 하강법보다 더 나은 성능을 제공.

- 모델이 최적점에 가까이 갈수록 학습률을 낮출 수 있음.
  - 이렇게 하면 안정적으로 최적점에 수렴할 가능성이 높음.
  - 이런 학습률을 적응적 학습률(adaptive learning rate)라고 함
  - 학습률 매개변수를 튜닝하는 수고를 덜 수 있는 것이 장점
- 대표적인 옵티마이저는 Adagrad, PMSprop


```python
adagrad = keras.optimizers.Adagrad()
model.compile(optimizer = adagrad, loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
```


```python
rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer = rmsprop, loss = 'sparse_categorical_crossentropy', metrics = 'accuracy')
```

- 모멘텀 최적화와 RMSprop의 장점을 접목한 것이 Adam


```python
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))
```


```python
model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics='accuracy')
model.fit(train_scaled, train_target, epochs=5)
```

    Epoch 1/5
    1500/1500 [==============================] - 5s 3ms/step - loss: 0.5216 - accuracy: 0.8180
    Epoch 2/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3914 - accuracy: 0.8591
    Epoch 3/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3508 - accuracy: 0.8718
    Epoch 4/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3246 - accuracy: 0.8817
    Epoch 5/5
    1500/1500 [==============================] - 4s 3ms/step - loss: 0.3048 - accuracy: 0.8883
    




    <keras.callbacks.History at 0x7fa78fa64d50>




```python
model.evaluate(val_scaled, val_target)
```

    375/375 [==============================] - 2s 4ms/step - loss: 0.3498 - accuracy: 0.8761
    




    [0.34981411695480347, 0.8760833144187927]



## 마무리

### 키워드로 끝내는 핵심 포인트

- 심층 신경망 : 2개 이상의 층을 포함한 신경망. 종종 다층 인공 신경망, 심층 신경망, 딥러닝을 같은 의미로 사옹
- 렐루 함수 : 이미지 분류 모델의 은닉층에 만힝 사요하는 활성화 함수.
  - 시그모이드 함수는 층이 많을 수로 활성화 함수의 양쪽 끝에서 변화가 작기 때문에 학습이 어려워짐.
  - 렐루 함수는 이런 문제가없으며 계산도 간단.
- 옵티마이저 : 신경망의가중치와 절편을 학습하기 위한 알고리즘 또는 방법을 말함.
  - 케라스에는 다양한 경사 하강버빙 구현
    - 대표적으로 SGD, 네스테로프 모멘텀, RMSprop, Adam 등이 있음

### 핵심 패키지와 함수

> TensorFlow
- add() : 케라스 모델에 층을 추가.
  - 케라스 모델의 add() 메서드는 keras.layers 패키지 아래에 있는 객체를 입력받아 신경망 모델을 추가.
  - add() 메서드를 호출하여 전달한 순서대로 층이 차례대로 늘어남.
- summary() : 케라스 모델의 정보를 출력하는 메서드.
  - 모델에 추가된 층의 종류와 순서, 모델 파라미터 개수를 출력.
  - 층을 만들 때 name 매개변수로 이름을 지정하먄 summary() 메서드 출력에서 구분하기 쉬움.
- SGD : 기본 경사 하강법 옵티마이저 클래스.
  - learning_rate : 학습률 지정, 기본값 = 0.01
  - momentum : 0 이상의 값을 지정하겸 모멘텀 최적화를 수행.
  - nesterov : True로 설정하면, 네트테로브 모멘텀 최적화를 수행.
- Adagrad : Adagrad 옵티마이저 클래스
  - learnig_rate : 학습률을 지정하며 기본값 = 0.001
  - Adagrad : 그레이디언트 제곱을 누적하여 학습률을 나눔.
    - initial_accumulator_value 매개변수에서 누적 초깃값을 지정할 수 있으며 기본값 = 0.1
- RMSprop : RMSprop 옵티마이저 클래스.
  - learnig_rate : 학습률 지정, 기본값 = 0.001
  - Adagrad 처럼 그레이디언트 제곱으로 학습률을 나누지만 최근의 그레이디언트를 사용하기 위해 지수 감소를 사옹.
    - rho : 감소 비율 지정, 기본값 = 0.9
- Adam : Adam 옵티마이저 클래스
  - learning_rate : 학습률 지정, 기본값 = 0.001
  - 모멘텀 최적화에 있는 그레이디언트의 지수 감소 평균을 조절하기 위해 beta_1 매개변수가 있으며, 기본값 = 0.9
  - RMSprop에 있는 그레이디언트의 지수 감소 평균을 조절하기 위해 beta_2 매개변수가 있으며, 기본값 = 0.999

# Book : '혼자 공부하는 머신러닝 + 딥러닝', 박해선 지음, 한빛미디어