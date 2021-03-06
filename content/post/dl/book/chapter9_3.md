---
title: "[DL][Book][혼공 머신러닝+딥러닝] ch9-3 LSTM과 GRU 셀"
description: ""
date: "2022-05-06T17:30:45+09:00"
thumbnail: ""
categories:
  - "DL"
tags:
  - "Python"
  - "DL"

---
<!--more-->

## LSTM 구조

- Long Short-Term Memory
- 단기 기억을 오래 기억하기 위해 고안
- 입력과 가중치를 곱하고 절편을 더해 활성화 함수를 통과시키는 구조를 여러 개 가지고 있음
  - 이런 계산 결과는 다음 타임스텝에 재사용됨

- 은닉상태를 만드는 방법
  - 입력과 이전 타임스텝의 은닉 상태를 가중치에 곱한 후 활성화 함수를 통과시켜 다음 은닉 상태를 생성
    - 이때 기본 순환층과는 달리 시그모이드 활성화 함수를 사용
    - tanh 활성화 함수를 통과한 어떤 값과 곱해져서 은닉 상태를 만듦

- LSTM에는 순환되는 상태가 2개
  - 은닉 상태
  - 셀 상태(cell state) : 은닉 상태와 달리 다음 층으로 전달되지 않고 LSTM 셀에서 순환만 되는 값

- 셀 상태 계산
  1. 입력과 은닉 상태를 또 다른 가중치 $w_{i}$에 곱한 다음 시그모이드 함수를 통과
  2. 이전 타입스텝의 셀 사태와 곱하여 새로운 셀 상태를 생성
    - 이 셀 상태가 tanh 함수를 통과하여 새로운 은닉 상태를 만드는 데 기여

- LSTM은 작은 셀을 여러 개 포함하고 있는 큰 셀과 같음
- 중요한 것은 입력과 은닉 상태에 곱해지는 가중치 $w_{o}$와 $w_{f}$가 다르다는 점
  - 이 두 작은 셀은 각기 다른 기능을 위해 훈련됨

- 여기에 2개의 작은 셀이 더 추가되어 셀 상태를 만드는 데 기여\
  - LSTM은 총 4개의 셀이 있음
- 입력과 은닉 상태를 각기 다른 가중체에 곱한 다음, 하나는 시그모이드 함수를 통과시키고 다른 하나는 tanh 함수를 통과시킴
- 두 결과를 곱한 후 이전 셀 상태와 더함
  - 이 결과가 최종적인 다음 셀 상태가 됨


- 삭제 게이트, 입력 게이트, 출력 게이트
  - 삭제 게이트 : 셀 상태에 있는 정보를 제거하는 역할
  - 입력 게이트 : 새로운 정보를 셀 상태에 추가
  - 출력 게이트 : 이 게이트를 통해 셀 상태가 다음 은닉 상태로 출력

## LSTM 신경망 훈력


```python
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)
```

- 그 다음 케라스의 pad_sequences() 함수로 각 샘플의 길이를 100에 맞추고 부족할 때는 패딩을 추가


```python
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)
```

- LSTM 셀을 상요한 순환층을 생성


```python
from tensorflow import keras
model = keras.Sequential()
model.add(keras.layers.Embedding(500, 16, input_length=100))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))
```


```python
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding_2 (Embedding)     (None, 100, 16)           8000      
                                                                     
     lstm_2 (LSTM)               (None, 8)                 800       
                                                                     
     dense_2 (Dense)             (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 8,809
    Trainable params: 8,809
    Non-trainable params: 0
    _________________________________________________________________
    

- SimpleRNN 클래스의 모델 파라미터의 개수는 200개
- LSTM 셀에는 작은 셀이 4개 있으므로 정확히 4배가 늘어 모델 파라미터 개수는 800개가 됨

- 배치 크기 = 64
- 에포크 회수 = 100


```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss = 'binary_crossentropy',
              metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5',
                                                 save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)
history = model.fit(train_seq, train_target, epochs=100, batch_size=64,
                    validation_data=(val_seq, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

    Epoch 1/100
    313/313 [==============================] - 13s 36ms/step - loss: 0.6926 - accuracy: 0.5258 - val_loss: 0.6917 - val_accuracy: 0.5680
    Epoch 2/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.6904 - accuracy: 0.5907 - val_loss: 0.6886 - val_accuracy: 0.6204
    Epoch 3/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.6845 - accuracy: 0.6497 - val_loss: 0.6792 - val_accuracy: 0.6664
    Epoch 4/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.6555 - accuracy: 0.7054 - val_loss: 0.6158 - val_accuracy: 0.7126
    Epoch 5/100
    313/313 [==============================] - 12s 39ms/step - loss: 0.5900 - accuracy: 0.7181 - val_loss: 0.5802 - val_accuracy: 0.7160
    Epoch 6/100
    313/313 [==============================] - 11s 37ms/step - loss: 0.5650 - accuracy: 0.7347 - val_loss: 0.5579 - val_accuracy: 0.7402
    Epoch 7/100
    313/313 [==============================] - 11s 35ms/step - loss: 0.5449 - accuracy: 0.7499 - val_loss: 0.5432 - val_accuracy: 0.7474
    Epoch 8/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.5278 - accuracy: 0.7621 - val_loss: 0.5250 - val_accuracy: 0.7604
    Epoch 9/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.5118 - accuracy: 0.7721 - val_loss: 0.5124 - val_accuracy: 0.7668
    Epoch 10/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4964 - accuracy: 0.7800 - val_loss: 0.4961 - val_accuracy: 0.7746
    Epoch 11/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4822 - accuracy: 0.7861 - val_loss: 0.4853 - val_accuracy: 0.7820
    Epoch 12/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4696 - accuracy: 0.7926 - val_loss: 0.4745 - val_accuracy: 0.7836
    Epoch 13/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4588 - accuracy: 0.7969 - val_loss: 0.4644 - val_accuracy: 0.7920
    Epoch 14/100
    313/313 [==============================] - 13s 40ms/step - loss: 0.4497 - accuracy: 0.8008 - val_loss: 0.4583 - val_accuracy: 0.7926
    Epoch 15/100
    313/313 [==============================] - 13s 41ms/step - loss: 0.4424 - accuracy: 0.8044 - val_loss: 0.4518 - val_accuracy: 0.7986
    Epoch 16/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4365 - accuracy: 0.8061 - val_loss: 0.4480 - val_accuracy: 0.7978
    Epoch 17/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4315 - accuracy: 0.8086 - val_loss: 0.4445 - val_accuracy: 0.8004
    Epoch 18/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4280 - accuracy: 0.8102 - val_loss: 0.4423 - val_accuracy: 0.8010
    Epoch 19/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4245 - accuracy: 0.8108 - val_loss: 0.4426 - val_accuracy: 0.8024
    Epoch 20/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4219 - accuracy: 0.8127 - val_loss: 0.4391 - val_accuracy: 0.7974
    Epoch 21/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4197 - accuracy: 0.8134 - val_loss: 0.4355 - val_accuracy: 0.8052
    Epoch 22/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4174 - accuracy: 0.8130 - val_loss: 0.4351 - val_accuracy: 0.7972
    Epoch 23/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4160 - accuracy: 0.8135 - val_loss: 0.4341 - val_accuracy: 0.8020
    Epoch 24/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4141 - accuracy: 0.8140 - val_loss: 0.4328 - val_accuracy: 0.8048
    Epoch 25/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4128 - accuracy: 0.8134 - val_loss: 0.4332 - val_accuracy: 0.8062
    Epoch 26/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4121 - accuracy: 0.8146 - val_loss: 0.4306 - val_accuracy: 0.8034
    Epoch 27/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4111 - accuracy: 0.8135 - val_loss: 0.4312 - val_accuracy: 0.8068
    Epoch 28/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4095 - accuracy: 0.8143 - val_loss: 0.4309 - val_accuracy: 0.7994
    Epoch 29/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4089 - accuracy: 0.8142 - val_loss: 0.4294 - val_accuracy: 0.8008
    Epoch 30/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4078 - accuracy: 0.8164 - val_loss: 0.4289 - val_accuracy: 0.8022
    Epoch 31/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4072 - accuracy: 0.8164 - val_loss: 0.4339 - val_accuracy: 0.8048
    Epoch 32/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4069 - accuracy: 0.8155 - val_loss: 0.4308 - val_accuracy: 0.8054
    Epoch 33/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4057 - accuracy: 0.8174 - val_loss: 0.4284 - val_accuracy: 0.8006
    Epoch 34/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4049 - accuracy: 0.8152 - val_loss: 0.4334 - val_accuracy: 0.8056
    Epoch 35/100
    313/313 [==============================] - 11s 35ms/step - loss: 0.4044 - accuracy: 0.8184 - val_loss: 0.4286 - val_accuracy: 0.8064
    Epoch 36/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4040 - accuracy: 0.8156 - val_loss: 0.4269 - val_accuracy: 0.8052
    Epoch 37/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4031 - accuracy: 0.8156 - val_loss: 0.4271 - val_accuracy: 0.8038
    Epoch 38/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4029 - accuracy: 0.8168 - val_loss: 0.4263 - val_accuracy: 0.8038
    Epoch 39/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4017 - accuracy: 0.8159 - val_loss: 0.4267 - val_accuracy: 0.8046
    Epoch 40/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4014 - accuracy: 0.8159 - val_loss: 0.4269 - val_accuracy: 0.8074
    Epoch 41/100
    313/313 [==============================] - 11s 34ms/step - loss: 0.4008 - accuracy: 0.8173 - val_loss: 0.4266 - val_accuracy: 0.8028
    


```python
import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter9_3/output_19_0.png)


- 기본 순환층보다 LSTM이 과대적합을 억제하면서 훈련을 잘 수행
- 하지만 경우에 따라서는 과대적합을 더 강하게 제어할 필요가 있음

## 순환층에 드롭아웃 적용하기

- 완전 연결 실졍망과 합성곱 신경망에서는 Dropput 클래스를 사용해 드롭아웃을 적용
  - 드롭아웃은 은닉층에 있는 뉴런의 출력을 랜덤하게 꺼서 과대적합을 막는 기법
  - 이를 통해 모델이 훈련 세트에 너무 과대 적합된는 것을 막음
- 순환층은 자체적으로 드롭아웃 기능을 제공
- SimpleRNN과 LSTM 클래스 모두 dropout 매개변수와 recurrnet_dropout 매개변수를 가지고 있음
  

- dropout 매개변수는 셀의 입력에 드롭아웃을 적용하고 recurrent_dropout은 순환되는 은닉 상태에 드롭아웃을 적용
- 기술적인 문제로 인해 recurrent_dropout을 사용하면 GPU를 사용하여 모델을 훈련하지 못함
  - 이 때문에 모델의 훈련 속도가 크게 느려짐

- LSTM 클래스에 dropout 매개변수를 0.3으로 지정하여 30%의 입력을 드롭아웃


```python
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.LSTM(8, dropout=0.3))
model2.add(keras.layers.Dense(1, activation='sigmoid'))
```


```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss = 'binary_crossentropy',
              metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-dropout-model.h5',
                                                 save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)
history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                    validation_data=(val_seq, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

    Epoch 1/100
    313/313 [==============================] - 14s 37ms/step - loss: 0.6925 - accuracy: 0.5213 - val_loss: 0.6918 - val_accuracy: 0.5640
    Epoch 2/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.6903 - accuracy: 0.5796 - val_loss: 0.6890 - val_accuracy: 0.6126
    Epoch 3/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.6858 - accuracy: 0.6264 - val_loss: 0.6829 - val_accuracy: 0.6426
    Epoch 4/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.6750 - accuracy: 0.6647 - val_loss: 0.6645 - val_accuracy: 0.6822
    Epoch 5/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.6329 - accuracy: 0.6978 - val_loss: 0.5980 - val_accuracy: 0.7220
    Epoch 6/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.5804 - accuracy: 0.7309 - val_loss: 0.5634 - val_accuracy: 0.7380
    Epoch 7/100
    313/313 [==============================] - 11s 35ms/step - loss: 0.5560 - accuracy: 0.7448 - val_loss: 0.5456 - val_accuracy: 0.7472
    Epoch 8/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.5388 - accuracy: 0.7548 - val_loss: 0.5303 - val_accuracy: 0.7622
    Epoch 9/100
    313/313 [==============================] - 12s 40ms/step - loss: 0.5235 - accuracy: 0.7628 - val_loss: 0.5168 - val_accuracy: 0.7676
    Epoch 10/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.5095 - accuracy: 0.7718 - val_loss: 0.5044 - val_accuracy: 0.7742
    Epoch 11/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4960 - accuracy: 0.7782 - val_loss: 0.5001 - val_accuracy: 0.7712
    Epoch 12/100
    313/313 [==============================] - 11s 35ms/step - loss: 0.4878 - accuracy: 0.7828 - val_loss: 0.4872 - val_accuracy: 0.7774
    Epoch 13/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4772 - accuracy: 0.7879 - val_loss: 0.4775 - val_accuracy: 0.7836
    Epoch 14/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4703 - accuracy: 0.7889 - val_loss: 0.4710 - val_accuracy: 0.7872
    Epoch 15/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4637 - accuracy: 0.7941 - val_loss: 0.4659 - val_accuracy: 0.7866
    Epoch 16/100
    313/313 [==============================] - 11s 37ms/step - loss: 0.4560 - accuracy: 0.7973 - val_loss: 0.4608 - val_accuracy: 0.7888
    Epoch 17/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4513 - accuracy: 0.7970 - val_loss: 0.4572 - val_accuracy: 0.7912
    Epoch 18/100
    313/313 [==============================] - 11s 37ms/step - loss: 0.4491 - accuracy: 0.7961 - val_loss: 0.4533 - val_accuracy: 0.7900
    Epoch 19/100
    313/313 [==============================] - 16s 52ms/step - loss: 0.4441 - accuracy: 0.7988 - val_loss: 0.4502 - val_accuracy: 0.7930
    Epoch 20/100
    313/313 [==============================] - 14s 45ms/step - loss: 0.4411 - accuracy: 0.8015 - val_loss: 0.4477 - val_accuracy: 0.7940
    Epoch 21/100
    313/313 [==============================] - 14s 44ms/step - loss: 0.4377 - accuracy: 0.8013 - val_loss: 0.4494 - val_accuracy: 0.7912
    Epoch 22/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4361 - accuracy: 0.8037 - val_loss: 0.4446 - val_accuracy: 0.7964
    Epoch 23/100
    313/313 [==============================] - 12s 37ms/step - loss: 0.4317 - accuracy: 0.8062 - val_loss: 0.4442 - val_accuracy: 0.7926
    Epoch 24/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4311 - accuracy: 0.8047 - val_loss: 0.4419 - val_accuracy: 0.7976
    Epoch 25/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4271 - accuracy: 0.8074 - val_loss: 0.4399 - val_accuracy: 0.7954
    Epoch 26/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4254 - accuracy: 0.8084 - val_loss: 0.4389 - val_accuracy: 0.7954
    Epoch 27/100
    313/313 [==============================] - 12s 37ms/step - loss: 0.4254 - accuracy: 0.8090 - val_loss: 0.4372 - val_accuracy: 0.7966
    Epoch 28/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4247 - accuracy: 0.8091 - val_loss: 0.4364 - val_accuracy: 0.7978
    Epoch 29/100
    313/313 [==============================] - 11s 37ms/step - loss: 0.4222 - accuracy: 0.8091 - val_loss: 0.4354 - val_accuracy: 0.7990
    Epoch 30/100
    313/313 [==============================] - 16s 51ms/step - loss: 0.4216 - accuracy: 0.8080 - val_loss: 0.4348 - val_accuracy: 0.7974
    Epoch 31/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4200 - accuracy: 0.8097 - val_loss: 0.4338 - val_accuracy: 0.7996
    Epoch 32/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4192 - accuracy: 0.8109 - val_loss: 0.4335 - val_accuracy: 0.7972
    Epoch 33/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4184 - accuracy: 0.8105 - val_loss: 0.4319 - val_accuracy: 0.7986
    Epoch 34/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4180 - accuracy: 0.8109 - val_loss: 0.4315 - val_accuracy: 0.7996
    Epoch 35/100
    313/313 [==============================] - 12s 40ms/step - loss: 0.4161 - accuracy: 0.8116 - val_loss: 0.4310 - val_accuracy: 0.7992
    Epoch 36/100
    313/313 [==============================] - 11s 36ms/step - loss: 0.4147 - accuracy: 0.8115 - val_loss: 0.4315 - val_accuracy: 0.8016
    Epoch 37/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4137 - accuracy: 0.8135 - val_loss: 0.4323 - val_accuracy: 0.7964
    Epoch 38/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4133 - accuracy: 0.8105 - val_loss: 0.4319 - val_accuracy: 0.7972
    

- 검증 손실이 약간 향상


```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter9_3/output_28_0.png)


- LSTM 층에 적용한 드롭아웃이 효과를 발휘
- 훈련 손실과 검증 손실 간의 차이가 좁혀진 것을 확인

## 2개의 층을 연결하기

- 순환층을 연결할 때는 한 가지 주의할 점이 존재
- 순환층의 은닉 상태는 샘플의 마지막 타임스탭에 대한 은닉 상태만 다음 층으로 전달
- 하지만 순환층을 쌓게 되면 모든 순환층에 순차 데이터가 필요
  - 앞쪽의 순환층이 모든 타임스텝에 대한 은닉 상태를 출력해야함
- 오직 마지막 순환층만 마지막 타임스텝의 은닉 상태를 출력해야함

- 케라스의 순환층에서 모든 타입스텝의 은닉 상태를 출력하려면 마지막을 제외한 다른 모든 순환층에서 return_sequences 매개변수를 True로 지정하면 됨


```python
model3 = keras.Sequential()
model3.add(keras.layers.Embedding(500, 16, input_length=100))
model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
model3.add(keras.layers.LSTM(8, dropout=0.3))
model3.add(keras.layers.Dense(1, activation='sigmoid'))
```

- 2개의 LSTM 층을 쌓았고 모두 드롭아웃을 0.3으로 지정
- 첫 번째 LSTM 클래스에는 return_sequences 매개변수를 True로 지정한 것을 확인


```python
model3.summary()
```

    Model: "sequential_4"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding_4 (Embedding)     (None, 100, 16)           8000      
                                                                     
     lstm_4 (LSTM)               (None, 100, 8)            800       
                                                                     
     lstm_5 (LSTM)               (None, 8)                 544       
                                                                     
     dense_4 (Dense)             (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 9,353
    Trainable params: 9,353
    Non-trainable params: 0
    _________________________________________________________________
    

- 첫 번째 LSTM 층이 모든 타임스텝(100개)의 은닉 상태를 출력하기 때문에 출력 크기가 (None, 100, 8)로 표시
- 두 번째 LSTM 층의 출력 크기는 마지막 타임스텝의 은닉 상태만 출력하기 때문에 (None, 8)


```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model3.compile(optimizer=rmsprop, loss = 'binary_crossentropy',
              metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-2rnn-model.h5',
                                                 save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)
history = model3.fit(train_seq, train_target, epochs=100, batch_size=64,
                    validation_data=(val_seq, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

    Epoch 1/100
    313/313 [==============================] - 26s 71ms/step - loss: 0.6920 - accuracy: 0.5326 - val_loss: 0.6902 - val_accuracy: 0.5734
    Epoch 2/100
    313/313 [==============================] - 22s 70ms/step - loss: 0.6832 - accuracy: 0.6094 - val_loss: 0.6722 - val_accuracy: 0.6584
    Epoch 3/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.6362 - accuracy: 0.6755 - val_loss: 0.5861 - val_accuracy: 0.7112
    Epoch 4/100
    313/313 [==============================] - 22s 69ms/step - loss: 0.5543 - accuracy: 0.7275 - val_loss: 0.5317 - val_accuracy: 0.7412
    Epoch 5/100
    313/313 [==============================] - 22s 71ms/step - loss: 0.5223 - accuracy: 0.7464 - val_loss: 0.5032 - val_accuracy: 0.7610
    Epoch 6/100
    313/313 [==============================] - 22s 71ms/step - loss: 0.5031 - accuracy: 0.7592 - val_loss: 0.4942 - val_accuracy: 0.7644
    Epoch 7/100
    313/313 [==============================] - 21s 67ms/step - loss: 0.4896 - accuracy: 0.7674 - val_loss: 0.4777 - val_accuracy: 0.7744
    Epoch 8/100
    313/313 [==============================] - 21s 67ms/step - loss: 0.4796 - accuracy: 0.7750 - val_loss: 0.4734 - val_accuracy: 0.7820
    Epoch 9/100
    313/313 [==============================] - 21s 67ms/step - loss: 0.4718 - accuracy: 0.7793 - val_loss: 0.4650 - val_accuracy: 0.7822
    Epoch 10/100
    313/313 [==============================] - 22s 69ms/step - loss: 0.4702 - accuracy: 0.7779 - val_loss: 0.4602 - val_accuracy: 0.7858
    Epoch 11/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4622 - accuracy: 0.7808 - val_loss: 0.4586 - val_accuracy: 0.7860
    Epoch 12/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4578 - accuracy: 0.7868 - val_loss: 0.4582 - val_accuracy: 0.7876
    Epoch 13/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4533 - accuracy: 0.7907 - val_loss: 0.4608 - val_accuracy: 0.7864
    Epoch 14/100
    313/313 [==============================] - 23s 74ms/step - loss: 0.4507 - accuracy: 0.7904 - val_loss: 0.4531 - val_accuracy: 0.7908
    Epoch 15/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4486 - accuracy: 0.7933 - val_loss: 0.4475 - val_accuracy: 0.7920
    Epoch 16/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4442 - accuracy: 0.7966 - val_loss: 0.4496 - val_accuracy: 0.7924
    Epoch 17/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4406 - accuracy: 0.7963 - val_loss: 0.4441 - val_accuracy: 0.7942
    Epoch 18/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4405 - accuracy: 0.7958 - val_loss: 0.4455 - val_accuracy: 0.7912
    Epoch 19/100
    313/313 [==============================] - 22s 69ms/step - loss: 0.4388 - accuracy: 0.8004 - val_loss: 0.4442 - val_accuracy: 0.7946
    Epoch 20/100
    313/313 [==============================] - 22s 69ms/step - loss: 0.4367 - accuracy: 0.7997 - val_loss: 0.4403 - val_accuracy: 0.7944
    Epoch 21/100
    313/313 [==============================] - 23s 73ms/step - loss: 0.4343 - accuracy: 0.8012 - val_loss: 0.4400 - val_accuracy: 0.7976
    Epoch 22/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4320 - accuracy: 0.8034 - val_loss: 0.4408 - val_accuracy: 0.7930
    Epoch 23/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4319 - accuracy: 0.8009 - val_loss: 0.4377 - val_accuracy: 0.7944
    Epoch 24/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4322 - accuracy: 0.8007 - val_loss: 0.4363 - val_accuracy: 0.7986
    Epoch 25/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4291 - accuracy: 0.8039 - val_loss: 0.4361 - val_accuracy: 0.7928
    Epoch 26/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4289 - accuracy: 0.8033 - val_loss: 0.4483 - val_accuracy: 0.7938
    Epoch 27/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4266 - accuracy: 0.8053 - val_loss: 0.4342 - val_accuracy: 0.7972
    Epoch 28/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4240 - accuracy: 0.8063 - val_loss: 0.4338 - val_accuracy: 0.7988
    Epoch 29/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4244 - accuracy: 0.8072 - val_loss: 0.4340 - val_accuracy: 0.8028
    Epoch 30/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4215 - accuracy: 0.8069 - val_loss: 0.4321 - val_accuracy: 0.7996
    Epoch 31/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4230 - accuracy: 0.8083 - val_loss: 0.4312 - val_accuracy: 0.8014
    Epoch 32/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4219 - accuracy: 0.8077 - val_loss: 0.4318 - val_accuracy: 0.8042
    Epoch 33/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4206 - accuracy: 0.8080 - val_loss: 0.4306 - val_accuracy: 0.8030
    Epoch 34/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4198 - accuracy: 0.8094 - val_loss: 0.4310 - val_accuracy: 0.8016
    Epoch 35/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4185 - accuracy: 0.8081 - val_loss: 0.4343 - val_accuracy: 0.7984
    Epoch 36/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4169 - accuracy: 0.8095 - val_loss: 0.4294 - val_accuracy: 0.8048
    Epoch 37/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4185 - accuracy: 0.8072 - val_loss: 0.4287 - val_accuracy: 0.8028
    Epoch 38/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4164 - accuracy: 0.8104 - val_loss: 0.4288 - val_accuracy: 0.8032
    Epoch 39/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4152 - accuracy: 0.8116 - val_loss: 0.4282 - val_accuracy: 0.8070
    Epoch 40/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4139 - accuracy: 0.8084 - val_loss: 0.4273 - val_accuracy: 0.8058
    Epoch 41/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4150 - accuracy: 0.8121 - val_loss: 0.4276 - val_accuracy: 0.8042
    Epoch 42/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4127 - accuracy: 0.8145 - val_loss: 0.4264 - val_accuracy: 0.8048
    Epoch 43/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4137 - accuracy: 0.8127 - val_loss: 0.4274 - val_accuracy: 0.8060
    Epoch 44/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4104 - accuracy: 0.8126 - val_loss: 0.4267 - val_accuracy: 0.8056
    Epoch 45/100
    313/313 [==============================] - 21s 68ms/step - loss: 0.4108 - accuracy: 0.8105 - val_loss: 0.4290 - val_accuracy: 0.8028
    

- 일반적으로 순환층을 쌓으면 성능이 높아짐


```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter9_3/output_39_0.png)


- 과대적합을 제어하면서 손실을 최대한 낮춤

## GRU 구조

- Gated Recurrent Unit
- 이 셀은 LSTM을 간소화한 버전으로 생각 가능
  - LSTM처럼 셀 상태를 계산하지 않고 은닉 상태 하나만 포함

- GRU 셀에는 은닉 상태와 입력에 가중치를 곱하고 절편을 더하는 작은 셀이 3개 들어 있음
  - 2개는 시그모이드 활성화 함수를 사용하고 하나는 tanh 활성화 함수를 사용
- 여기에서도 은닉 상태화 입력에 곱해지는 가중치를 합쳐서 나타냄

- 맨 왼쪽에서 $w_{z}$를 사용하는 셀의 출력이 은닉 상테에 바로 곱해져 삭제 게이트 역할을 수행
- 이와 똑같은 출력을 1에서 뺀 다음에 가장 오른쪽 $w_{g}$를 사요하는 셀의 출력에 곱함
  - 이는 입력되는 정보를 제어하는 역할을 수행
- 가운데 $w_{r}$을 사용하는 셀에서 출력된 값은 $w_{g}$셀이 사용할 은닉 상태의 정보를 제어

- GRU 셀은 LSTM보다 가중치가 적기 때문에 계산량이 적지만 LSTM 못지않은 좋은 성능을 내는 것으로 알려짐

## GRU 신경망 훈련하기


```python
model4 = keras.Sequential()
model4.add(keras.layers.Embedding(500, 16, input_length=100))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation='sigmoid'))
```


```python
model4.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     embedding_5 (Embedding)     (None, 100, 16)           8000      
                                                                     
     gru (GRU)                   (None, 8)                 624       
                                                                     
     dense_5 (Dense)             (None, 1)                 9         
                                                                     
    =================================================================
    Total params: 8,633
    Trainable params: 8,633
    Non-trainable params: 0
    _________________________________________________________________
    

- GRU 층의 모델 파라미터 개수를 계산
  - GRU 셀에는 3개의 작은 셀이 존재
- 작은 셀에는 입력과 은닉 상태에 곱하는 가중치와 절편이 존재
- 입력에 곱하는 가중치는 $16 \times 8 =128$이고 은닉상태에 곱하는 가중치는 $8 \times 8 =64$개
- 절편은 뉴런마다 하나씩이므로 8개
- 모두 더하면 $128 + 64+8=200$개
  - 이런 작은 셀이 3개이므로 모두 600개의 모델 파라미터가 필요

- But) summary() 메서드의 출력은 624개

- GRU 셀의 초기 버전 계산
  - 은닉 상태가 먼저 가중치와 곱해진 다음 가운데 셀의 출력과 곱해짐
- 이전에는 입력과 은닉 상태에 곱해지는 가중치를 $w_{g}$로 별도로 표기했지만, 이번에는 $w_{x}$와 $w_{h}$로 나눔
- 이렇게 나누어 계산하면 은닉 상태에 곱해지는 가중치 외에 절편이 별도로 필요
  - 따라서, 작은 셀마다 하나씩 절편이 추가되고 8개의 뉴런이 있으므로 총 24개의 모델 파라미터가 더해짐
  - GRU 층의 총 모델 파라미터 개수는 624개가 됨


```python
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model4.compile(optimizer=rmsprop, loss = 'binary_crossentropy',
              metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-gru-model.h5',
                                                 save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)
history = model4.fit(train_seq, train_target, epochs=100, batch_size=64,
                    validation_data=(val_seq, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

    Epoch 1/100
    313/313 [==============================] - 14s 39ms/step - loss: 0.6926 - accuracy: 0.5252 - val_loss: 0.6921 - val_accuracy: 0.5400
    Epoch 2/100
    313/313 [==============================] - 12s 37ms/step - loss: 0.6911 - accuracy: 0.5662 - val_loss: 0.6904 - val_accuracy: 0.5648
    Epoch 3/100
    313/313 [==============================] - 14s 46ms/step - loss: 0.6886 - accuracy: 0.5810 - val_loss: 0.6876 - val_accuracy: 0.5776
    Epoch 4/100
    313/313 [==============================] - 13s 40ms/step - loss: 0.6845 - accuracy: 0.5996 - val_loss: 0.6831 - val_accuracy: 0.5930
    Epoch 5/100
    313/313 [==============================] - 13s 41ms/step - loss: 0.6782 - accuracy: 0.6112 - val_loss: 0.6763 - val_accuracy: 0.6054
    Epoch 6/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.6686 - accuracy: 0.6296 - val_loss: 0.6658 - val_accuracy: 0.6186
    Epoch 7/100
    313/313 [==============================] - 12s 39ms/step - loss: 0.6541 - accuracy: 0.6432 - val_loss: 0.6495 - val_accuracy: 0.6402
    Epoch 8/100
    313/313 [==============================] - 12s 37ms/step - loss: 0.6310 - accuracy: 0.6634 - val_loss: 0.6235 - val_accuracy: 0.6618
    Epoch 9/100
    313/313 [==============================] - 12s 37ms/step - loss: 0.5925 - accuracy: 0.6913 - val_loss: 0.5769 - val_accuracy: 0.7014
    Epoch 10/100
    313/313 [==============================] - 12s 37ms/step - loss: 0.5347 - accuracy: 0.7347 - val_loss: 0.5331 - val_accuracy: 0.7396
    Epoch 11/100
    313/313 [==============================] - 12s 37ms/step - loss: 0.5046 - accuracy: 0.7569 - val_loss: 0.5103 - val_accuracy: 0.7536
    Epoch 12/100
    313/313 [==============================] - 13s 40ms/step - loss: 0.4884 - accuracy: 0.7667 - val_loss: 0.4978 - val_accuracy: 0.7606
    Epoch 13/100
    313/313 [==============================] - 12s 37ms/step - loss: 0.4758 - accuracy: 0.7762 - val_loss: 0.4853 - val_accuracy: 0.7720
    Epoch 14/100
    313/313 [==============================] - 13s 42ms/step - loss: 0.4653 - accuracy: 0.7840 - val_loss: 0.4798 - val_accuracy: 0.7770
    Epoch 15/100
    313/313 [==============================] - 13s 40ms/step - loss: 0.4571 - accuracy: 0.7893 - val_loss: 0.4696 - val_accuracy: 0.7778
    Epoch 16/100
    313/313 [==============================] - 12s 39ms/step - loss: 0.4496 - accuracy: 0.7930 - val_loss: 0.4642 - val_accuracy: 0.7828
    Epoch 17/100
    313/313 [==============================] - 12s 39ms/step - loss: 0.4439 - accuracy: 0.7972 - val_loss: 0.4614 - val_accuracy: 0.7850
    Epoch 18/100
    313/313 [==============================] - 13s 41ms/step - loss: 0.4390 - accuracy: 0.7997 - val_loss: 0.4551 - val_accuracy: 0.7860
    Epoch 19/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4347 - accuracy: 0.8025 - val_loss: 0.4513 - val_accuracy: 0.7908
    Epoch 20/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4311 - accuracy: 0.8042 - val_loss: 0.4490 - val_accuracy: 0.7916
    Epoch 21/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4282 - accuracy: 0.8077 - val_loss: 0.4471 - val_accuracy: 0.7920
    Epoch 22/100
    313/313 [==============================] - 12s 37ms/step - loss: 0.4257 - accuracy: 0.8086 - val_loss: 0.4454 - val_accuracy: 0.7946
    Epoch 23/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4239 - accuracy: 0.8091 - val_loss: 0.4442 - val_accuracy: 0.7942
    Epoch 24/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4221 - accuracy: 0.8108 - val_loss: 0.4453 - val_accuracy: 0.7908
    Epoch 25/100
    313/313 [==============================] - 12s 37ms/step - loss: 0.4206 - accuracy: 0.8122 - val_loss: 0.4429 - val_accuracy: 0.7954
    Epoch 26/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4190 - accuracy: 0.8138 - val_loss: 0.4443 - val_accuracy: 0.7962
    Epoch 27/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4183 - accuracy: 0.8122 - val_loss: 0.4412 - val_accuracy: 0.7976
    Epoch 28/100
    313/313 [==============================] - 12s 37ms/step - loss: 0.4170 - accuracy: 0.8144 - val_loss: 0.4410 - val_accuracy: 0.8002
    Epoch 29/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4164 - accuracy: 0.8136 - val_loss: 0.4424 - val_accuracy: 0.7928
    Epoch 30/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4158 - accuracy: 0.8145 - val_loss: 0.4399 - val_accuracy: 0.7984
    Epoch 31/100
    313/313 [==============================] - 12s 39ms/step - loss: 0.4152 - accuracy: 0.8144 - val_loss: 0.4412 - val_accuracy: 0.7984
    Epoch 32/100
    313/313 [==============================] - 12s 39ms/step - loss: 0.4142 - accuracy: 0.8156 - val_loss: 0.4392 - val_accuracy: 0.7980
    Epoch 33/100
    313/313 [==============================] - 12s 39ms/step - loss: 0.4140 - accuracy: 0.8148 - val_loss: 0.4428 - val_accuracy: 0.7966
    Epoch 34/100
    313/313 [==============================] - 12s 40ms/step - loss: 0.4129 - accuracy: 0.8158 - val_loss: 0.4406 - val_accuracy: 0.7966
    Epoch 35/100
    313/313 [==============================] - 12s 38ms/step - loss: 0.4125 - accuracy: 0.8156 - val_loss: 0.4409 - val_accuracy: 0.7970
    

- LSTM과 비슷한 성능


```python
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()
```


![png](/images/self_study_ml_dl_images/chapter9_3/output_53_0.png)


- 드롭아웃을 사용하지 않았기 때문에 이전보다 훈련 손실과 검증 손실 사이에 차이가 있지만 훈련 과정이 잘 수렴되고 있는 것을 확인 가능

## 마무리

### 키워드로 끝내는 핵심 포인트

- LSTM : 셀은 타임스텝이 긴 데이터를 효과적으로 학습하기 위해 고안된 순환층
  - 입력 게이트, 삭제 게이트, 출력 게이트 역할을 하는 작은 셀이 포함되어 있음
- LSTM 셀은 은닉 상태 외에 **셀 상태**를 출력.
  - 셀 상태는 다음 층으로 전달되지 않으며 현재 셀에서만 순환됨
- GRU 셀 : LSTM 셀의 간소화 버전으로 생각할 수 있지만 LSTM 셀에 못지않은 성능을 냄

### 핵심 패키지와 함수

> TensorFlow
- LSTM : LSTM 셀을 사용한 순환층 클래스
  - 첫 번째 매개변수 : 뉴런의 개수를 지정
  - dropout : 입력에 대한 드롭아웃 비율을 지정
  - return_sequences : 모든 타임스텝의 은닉 상태를 출력할 지 결정
    - 기본값 : False
- GRU : GRU 셀을 사용한 순환층 클래스
  - 첫 번째 매개변수 : 뉴런의 개수를 지정
  - dropout : 입력에 대한 드롭아웃 비율을 지정
  - return_sequences : 모든 타임스텝의 은닉 상태를 출력할 지 결정
    - 기본값 : False

# Book : '혼자 공부하는 머신러닝 + 딥러닝', 박해선 지음, 한빛미디어