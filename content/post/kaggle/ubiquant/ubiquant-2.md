---
title: "[Kaggle]Ubiquant Market Prediction - 2"
description: ""
date: "2022-04-24T18:33:45+09:00"
thumbnail: ""
categories:
  - "Kaggle"
tags:
  - "Python"
  - "ML"
  - "Kaggle"

---
### Prediction and Score
<!--more-->
## < Score Improvement History >
|#|> std * x|k-fold|score|rank|
|:-:|:-:|:-:|:-:|:-:|
|**1**|> std * 70|3|**0.1388**|**1795/2715**|
|2|> std * 60|3|0.1387|
|3|> std * 50|3|0.1378|
|4|> std * 40|3|0.1366|
|5|> std * 30|3|0.1368|
|6|> std * 80|3|0.1381|
|7|> std * 90|3|0.1375|
|8|> std * 70|5|0.1387|

|#|> std * x or(>std * y and < len(z))|k-fold|score|rank|
|:-:|:-:|:-:|:-:|:-:|
|9|> std * 70 or(>std * 35 and < len(6))|3|0.1383|
|**10**|> std * 70 or(>std * 35 and < len(6))|5|**0.1396**|**1757/2714**|


## **score : 0.1388 ->**<span style='color:blue'>0.1396</span>    
## **rank : 1795/2715 ->**<span style='color:blue'>1757/2714</span>

# Step 1 ~ 3 :    
https://www.kaggle.com/code/blackjjw/ubiquant-eda
## dataset :   
https://www.kaggle.com/datasets/blackjjw/ubiquant-train-df-low-memory-pkl

# 라이브러리 불러오기


```python
import numpy as np
import pandas as pd

import gc
import joblib

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import scipy as stats

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold 
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgbm
from lightgbm import *
```


<style type='text/css'>
.datatable table.frame { margin-bottom: 0; }
.datatable table.frame thead { border-bottom: none; }
.datatable table.frame tr.coltypes td {  color: #FFFFFF;  line-height: 6px;  padding: 0 0.5em;}
.datatable .bool    { background: #DDDD99; }
.datatable .object  { background: #565656; }
.datatable .int     { background: #5D9E5D; }
.datatable .float   { background: #4040CC; }
.datatable .str     { background: #CC4040; }
.datatable .time    { background: #40CC40; }
.datatable .row_index {  background: var(--jp-border-color3);  border-right: 1px solid var(--jp-border-color0);  color: var(--jp-ui-font-color3);  font-size: 9px;}
.datatable .frame tbody td { text-align: left; }
.datatable .frame tr.coltypes .row_index {  background: var(--jp-border-color0);}
.datatable th:nth-child(2) { padding-left: 12px; }
.datatable .hellipsis {  color: var(--jp-cell-editor-border-color);}
.datatable .vellipsis {  background: var(--jp-layout-color0);  color: var(--jp-cell-editor-border-color);}
.datatable .na {  color: var(--jp-cell-editor-border-color);  font-size: 80%;}
.datatable .sp {  opacity: 0.25;}
.datatable .footer { font-size: 9px; }
.datatable .frame_dimensions {  background: var(--jp-border-color3);  border-top: 1px solid var(--jp-border-color0);  color: var(--jp-ui-font-color3);  display: inline-block;  opacity: 0.6;  padding: 1px 10px 1px 5px;}
</style>



# 데이터 불러오기
- Step 1 ~ 3에서 진행하고 생성한 train_df.pkl를 read.


```python
%%time
train_df = pd.read_pickle('../input/ubiquant-train-df-low-memory-pkl/train_df.pkl')
train_df.head()
```

    CPU times: user 1.01 s, sys: 2.29 s, total: 3.3 s
    Wall time: 20.3 s
    




# Step 4. 데이터 전처리

## 필요없는 컬럼 제거
- row_id는 단순 행 번호를 나타내므로 제거.
- 컬럼 리스트를 따로 생성하고 'row_id' 제거.
- 'time_id' 제거
- 종속변수인 'target' 제거.
- 필요없는 컬럼이 생성된 리스트를 통해 필요한 컬럼들만 추출하여 사용.


```python
feature_cols = train_df.columns.unique()
feature_cols = feature_cols.drop(['row_id', 'target', 'time_id'])
feature_cols
```




    Index(['investment_id', 'f_0', 'f_1', 'f_2', 'f_3', 'f_4', 'f_5', 'f_6', 'f_7',
           'f_8',
           ...
           'f_290', 'f_291', 'f_292', 'f_293', 'f_294', 'f_295', 'f_296', 'f_297',
           'f_298', 'f_299'],
          dtype='object', length=301)



## 결측치 제거
- dataset 내에 결측치가 존재하지 않음.

## 이상치 제거
- 평균에서 멀리 떨어져 있는 값들을 제거 (> std*num)

## mean > std * x
- 이상치의 개수와 리스트를 반환
|x|outlier count|
|:-----:|:-----:|
|90|21|
|80|42|
|70|128|
|60|325|
|50|788|
|40|2232|
|30|5707|


```python
'''
outlier_list = []
num = 70

for col in feature_cols :
    trans_dtype_df = train_df[col].astype(np.float32)
    
    temp_df = train_df[(trans_dtype_df > trans_dtype_df.mean() + trans_dtype_df.std() * num) |
                       (trans_dtype_df < trans_dtype_df.mean() - trans_dtype_df.std() * num) ]
    if len(temp_df) >0 :
        outliers = temp_df.index.to_list()
        outlier_list.extend(outliers)
        print(col, len(temp_df))
        gc.collect()

outlier_list = list(set(outlier_list))
print(len(outlier_list))
gc.collect()
'''
```




    '\noutlier_list = []\nnum = 70\n\nfor col in feature_cols :\n    trans_dtype_df = train_df[col].astype(np.float32)\n    \n    temp_df = train_df[(trans_dtype_df > trans_dtype_df.mean() + trans_dtype_df.std() * num) |\n                       (trans_dtype_df < trans_dtype_df.mean() - trans_dtype_df.std() * num) ]\n    if len(temp_df) >0 :\n        outliers = temp_df.index.to_list()\n        outlier_list.extend(outliers)\n        print(col, len(temp_df))\n        gc.collect()\n\noutlier_list = list(set(outlier_list))\nprint(len(outlier_list))\ngc.collect()\n'



## mean > std * x or (> std * y and len < z)
- 조건 범위의 이상치의 개수와 리스트를 반환

|x|y|z|col_count|outlier_count|
|:-:|:-:|:-:|:-:|:-:|
|70|35|6|48|174|


```python
outlier_list = []
outlier_col = []
x = 70
y = 35
z = 6

for col in feature_cols :
    trans_dtype_df = train_df[col].astype(np.float32)
    
    temp_df = train_df[(trans_dtype_df > trans_dtype_df.mean() + trans_dtype_df.std() * x) |
                       (trans_dtype_df < trans_dtype_df.mean() - trans_dtype_df.std() * x) ]
    temp2_df = train_df[(trans_dtype_df > trans_dtype_df.mean() + trans_dtype_df.std() * y) |
                        (trans_dtype_df < trans_dtype_df.mean() - trans_dtype_df.std() * y) ]
    if len(temp_df) > 0 : 
        outliers = temp_df.index.to_list()
        outlier_list.extend(outliers)
        outlier_col.append(col)
        print(col, len(temp_df))
    elif len(temp2_df) > 0 and len(temp2_df) < z :
        outliers = temp2_df.index.to_list()
        outlier_list.extend(outliers)
        outlier_col.append(col)
        print(col, len(temp2_df))

outlier_list = list(set(outlier_list))
print(len(outlier_col), len(outlier_list))
```

    f_4 6
    f_10 1
    f_12 1
    f_13 1
    f_37 3
    f_49 1
    f_55 2
    f_62 1
    f_77 5
    f_78 1
    f_87 3
    f_99 4
    f_104 1
    f_108 5
    f_115 14
    f_117 16
    f_118 2
    f_122 2
    f_124 17
    f_127 16
    f_128 5
    f_136 2
    f_137 1
    f_145 2
    f_149 2
    f_155 2
    f_162 1
    f_165 1
    f_172 2
    f_174 1
    f_175 30
    f_179 4
    f_193 3
    f_196 1
    f_197 1
    f_200 37
    f_209 5
    f_214 1
    f_215 1
    f_219 1
    f_233 3
    f_249 1
    f_250 4
    f_265 1
    f_277 1
    f_280 3
    f_289 6
    f_295 1
    48 174
    

- 이상치 제거


```python
train_df.drop(train_df.index[outlier_list], inplace = True)
train_df
gc.collect()
```




    68



# Step 5. 머신러닝 모형 개발

- 독립변수와 종속변수를 구분.


```python
#X = train_df[feature_cols]
#y = train_df['target']
```

- X, y 데이터 분리


```python
#X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 42, shuffle = False)
#X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
```

- RAM 확보를 위해 X, y 제거


```python
'''
del X
del y
gc.collect()
'''
```




    '\ndel X\ndel y\ngc.collect()\n'



- LightGBM 클래스를 부른 후 모형을 학습


```python
'''
model = lgbm.LGBMRegressor(
        objective="regression",
        metric="rmse",
        n_estimators=500 )

model.fit(X_train, y_train)
'''
```




    '\nmodel = lgbm.LGBMRegressor(\n        objective="regression",\n        metric="rmse",\n        n_estimators=500 )\n\nmodel.fit(X_train, y_train)\n'




```python
"""
for j in [200, 300, 400, 500, 600]:
        model = lgbm.LGBMRegressor(
            objective="regression",
            metric="rmse",
            n_estimators= j )

        model.fit(X_train, y_train)

        score = model.score(X_train, y_train)
        print(f"n_e:{j}, Training Score : {score}")

        y_pred = model.predict(X_valid)
        mse = mean_squared_error(y_valid, y_pred)
        print(f"n_e:{j},MSE : {mse:.2f}")
"""
```




    '\nfor j in [200, 300, 400, 500, 600]:\n        model = lgbm.LGBMRegressor(\n            objective="regression",\n            metric="rmse",\n            n_estimators= j )\n\n        model.fit(X_train, y_train)\n\n        score = model.score(X_train, y_train)\n        print(f"n_e:{j}, Training Score : {score}")\n\n        y_pred = model.predict(X_valid)\n        mse = mean_squared_error(y_valid, y_pred)\n        print(f"n_e:{j},MSE : {mse:.2f}")\n'



### Training_Score & MSE 계산
|test_size|n_estimators|delete outliers|Training_Score|MSE|delete outliers|Training_Score|MSE|delete outliers|Training_Score|MSE|   
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|                              
|0.2|200|X|0.054069993842590924|0.81|>std\*70|0.05405621833673957(<span style="color:blue">$\downarrow$</span>)|0.81|>std\*60|0.05404028505271219(<span style="color:blue">$\downarrow$</span>)|0.81|
|0.3|200|X|0.05846368663812751|0.83|>std\*70|0.058449805435160984(<span style="color:blue">$\downarrow$</span>)|0.82|>std\*60|||
|0.2|300|X|0.06668339494814646|0.82|>std\*70|0.06660813482520711(<span style="color:blue">$\downarrow$</span>)|0.81|>std\*60|0.06678748573971982(<span style="color:red">$\uparrow$</span>)|0.81|
|0.3|300|X|0.07216780004832235|0.83|>std\*70|0.07194148416323642(<span style="color:blue">$\downarrow$</span>)|0.83|>std\*60|||
|0.2|400|X|0.07730909212135473|0.82|>std\*70|0.0772426438888425(<span style="color:blue">$\downarrow$</span>)|0.82|>std\*60|0.07730612556862904(<span style="color:red">$\uparrow$</span>)|0.82|
|0.3|400|X|0.08363594720433998|0.83|>std\*70|0.08389302333398285(<span style="color:red">$\uparrow$</span>)|0.83|>std\*60|||
|0.2|500|X|0.08671736858043233|0.82|>std\*70|0.08636877075690108(<span style="color:blue">$\downarrow$</span>)|0.82|>std\*60|0.08657127960859345(<span style="color:red">$\uparrow$</span>)|0.82|
|0.3|500|X|0.09377199406580616|0.83|>std\*70|0.0941912571685275(<span style="color:red">$\uparrow$</span>)|0.83|>std\*60|||
|0.2|600|X|0.09585158203024369|0.82|>std\*70|0.09531511104149593(<span style="color:blue">$\downarrow$</span>)|0.82|>std\*60|0.09508766558702908(<span style="color:blue">$\downarrow$</span>)|0.82|
|0.3|600|X|0.10299723306322961|0.83|>std\*70|0.10381084559412301(<span style="color:red">$\uparrow$</span>)|0.83|>std\*60|||
|0.2|700|X|0.10387433272279445|0.82|>std\*70|0.10403926054146928(<span style="color:blue">$\downarrow$</span>)|0.82|>std\*60|X|X|

- 이상치를 제거해도 전반적으로 training score가 항상 오르지 않는다.
- 오히려 떨어지는 것으로 보임

- trianing score, MSE 계산


```python
'''
score = model.score(X_train, y_train)
print(f"Training Score : {score}")

y_pred = model.predict(X_valid)
mse = mean_squared_error(y_valid, y_pred)
print(f"MSE : {mse:.2f}")
'''

```




    '\nscore = model.score(X_train, y_train)\nprint(f"Training Score : {score}")\n\ny_pred = model.predict(X_valid)\nmse = mean_squared_error(y_valid, y_pred)\nprint(f"MSE : {mse:.2f}")\n'



- LightGBM의 세부 파라미터 적용
    - 참고 자료를 조사하여 적용


```python
#https://www.kaggle.com/valleyzw/ubiquant-lgbm-baseline
params = {
        'learning_rate':0.1,
        "objective": "regression",
        "metric": "rmse",
        'boosting_type': "gbdt",
        'verbosity': -1,
        'n_jobs': -1, 
        'seed': 21,
        'lambda_l1': 1.1895057699067542, 
        'lambda_l2': 1.9079686837880768e-08, 
        'num_leaves': 112, 
        'subsample':None,
        'feature_fraction': 0.6259927292757151, 
        'bagging_fraction': 0.9782210574588895, 
        'bagging_freq': 1, 
        'n_estimators': 306, 
        'max_depth': 12, 
        'max_bin': 494, 
        'min_data_in_leaf': 366,
        'colsample_bytree': None,
        'subsample_freq': None,
        'min_child_samples': None,
        'reg_lambda': None,
        'reg_alpha': None,
    }
gc.collect()
```




    0



- StratifiedkFold를 통한 교차 검증을 통해 훈련 


```python
seed = 0
folds = 5
models = []
skf = StratifiedKFold(folds, shuffle = True, random_state = seed)

for train_index, test_index in skf.split(train_df, train_df['investment_id']):
    train = train_df.iloc[train_index]
    valid = train_df.iloc[test_index]
    
    lgbm = LGBMRegressor(**params)
    '''
        num_leaves=2 ** np.random.randint(3, 8),
        learning_rate = 10 ** (-np.random.uniform(0.1,2)),
        n_estimators = 500,
        min_child_samples = 1000, 
        subsample=np.random.uniform(0.5,1.0), 
        subsample_freq=1,
        n_jobs= -1,
        tree_method ='gpu_hist' ,
    '''
    
    lgbm.fit(train[feature_cols], train['target'], eval_set = (valid[feature_cols], valid['target']))
    models.append(lgbm)
    gc.collect()
gc.collect()
```

    /opt/conda/lib/python3.7/site-packages/sklearn/model_selection/_split.py:680: UserWarning: The least populated class in y has only 2 members, which is less than n_splits=5.
      UserWarning,
    

    [1]	valid_0's rmse: 0.920276
    [2]	valid_0's rmse: 0.919447
    [3]	valid_0's rmse: 0.918681
    [4]	valid_0's rmse: 0.918043
    [5]	valid_0's rmse: 0.917446
    [6]	valid_0's rmse: 0.916926
    [7]	valid_0's rmse: 0.916436
    [8]	valid_0's rmse: 0.916037
    [9]	valid_0's rmse: 0.915646
    [10]	valid_0's rmse: 0.915325
    [11]	valid_0's rmse: 0.915022
    [12]	valid_0's rmse: 0.914761
    [13]	valid_0's rmse: 0.914493
    [14]	valid_0's rmse: 0.91428
    [15]	valid_0's rmse: 0.914053
    [16]	valid_0's rmse: 0.913849
    [17]	valid_0's rmse: 0.913637
    [18]	valid_0's rmse: 0.913399
    [19]	valid_0's rmse: 0.913188
    [20]	valid_0's rmse: 0.912967
    [21]	valid_0's rmse: 0.912766
    [22]	valid_0's rmse: 0.912581
    [23]	valid_0's rmse: 0.912386
    [24]	valid_0's rmse: 0.912148
    [25]	valid_0's rmse: 0.911985
    [26]	valid_0's rmse: 0.911828
    [27]	valid_0's rmse: 0.911669
    [28]	valid_0's rmse: 0.911507
    [29]	valid_0's rmse: 0.911368
    [30]	valid_0's rmse: 0.911213
    [31]	valid_0's rmse: 0.911094
    [32]	valid_0's rmse: 0.910956
    [33]	valid_0's rmse: 0.910813
    [34]	valid_0's rmse: 0.910649
    [35]	valid_0's rmse: 0.91053
    [36]	valid_0's rmse: 0.91037
    [37]	valid_0's rmse: 0.910238
    [38]	valid_0's rmse: 0.910101
    [39]	valid_0's rmse: 0.909975
    [40]	valid_0's rmse: 0.909852
    [41]	valid_0's rmse: 0.909634
    [42]	valid_0's rmse: 0.90948
    [43]	valid_0's rmse: 0.909356
    [44]	valid_0's rmse: 0.909205
    [45]	valid_0's rmse: 0.909112
    [46]	valid_0's rmse: 0.909019
    [47]	valid_0's rmse: 0.908908
    [48]	valid_0's rmse: 0.908808
    [49]	valid_0's rmse: 0.908713
    [50]	valid_0's rmse: 0.908584
    [51]	valid_0's rmse: 0.908507
    [52]	valid_0's rmse: 0.908417
    [53]	valid_0's rmse: 0.908309
    [54]	valid_0's rmse: 0.90821
    [55]	valid_0's rmse: 0.908108
    [56]	valid_0's rmse: 0.90803
    [57]	valid_0's rmse: 0.90794
    [58]	valid_0's rmse: 0.907848
    [59]	valid_0's rmse: 0.907759
    [60]	valid_0's rmse: 0.907671
    [61]	valid_0's rmse: 0.907513
    [62]	valid_0's rmse: 0.907401
    [63]	valid_0's rmse: 0.907347
    [64]	valid_0's rmse: 0.907274
    [65]	valid_0's rmse: 0.907181
    [66]	valid_0's rmse: 0.907081
    [67]	valid_0's rmse: 0.906995
    [68]	valid_0's rmse: 0.906913
    [69]	valid_0's rmse: 0.906765
    [70]	valid_0's rmse: 0.906692
    [71]	valid_0's rmse: 0.90663
    [72]	valid_0's rmse: 0.906512
    [73]	valid_0's rmse: 0.906445
    [74]	valid_0's rmse: 0.906349
    [75]	valid_0's rmse: 0.906281
    [76]	valid_0's rmse: 0.906213
    [77]	valid_0's rmse: 0.90604
    [78]	valid_0's rmse: 0.906001
    [79]	valid_0's rmse: 0.90592
    [80]	valid_0's rmse: 0.905853
    [81]	valid_0's rmse: 0.905692
    [82]	valid_0's rmse: 0.905643
    [83]	valid_0's rmse: 0.905584
    [84]	valid_0's rmse: 0.905498
    [85]	valid_0's rmse: 0.905437
    [86]	valid_0's rmse: 0.905398
    [87]	valid_0's rmse: 0.905259
    [88]	valid_0's rmse: 0.905201
    [89]	valid_0's rmse: 0.905175
    [90]	valid_0's rmse: 0.905108
    [91]	valid_0's rmse: 0.905048
    [92]	valid_0's rmse: 0.904991
    [93]	valid_0's rmse: 0.904844
    [94]	valid_0's rmse: 0.904798
    [95]	valid_0's rmse: 0.904734
    [96]	valid_0's rmse: 0.904657
    [97]	valid_0's rmse: 0.904603
    [98]	valid_0's rmse: 0.904543
    [99]	valid_0's rmse: 0.904492
    [100]	valid_0's rmse: 0.90445
    [101]	valid_0's rmse: 0.904374
    [102]	valid_0's rmse: 0.904321
    [103]	valid_0's rmse: 0.904266
    [104]	valid_0's rmse: 0.904163
    [105]	valid_0's rmse: 0.904096
    [106]	valid_0's rmse: 0.90399
    [107]	valid_0's rmse: 0.90391
    [108]	valid_0's rmse: 0.903862
    [109]	valid_0's rmse: 0.90366
    [110]	valid_0's rmse: 0.903592
    [111]	valid_0's rmse: 0.903516
    [112]	valid_0's rmse: 0.903398
    [113]	valid_0's rmse: 0.903363
    [114]	valid_0's rmse: 0.903287
    [115]	valid_0's rmse: 0.903231
    [116]	valid_0's rmse: 0.903163
    [117]	valid_0's rmse: 0.903083
    [118]	valid_0's rmse: 0.903026
    [119]	valid_0's rmse: 0.902981
    [120]	valid_0's rmse: 0.90293
    [121]	valid_0's rmse: 0.902881
    [122]	valid_0's rmse: 0.902808
    [123]	valid_0's rmse: 0.902733
    [124]	valid_0's rmse: 0.902683
    [125]	valid_0's rmse: 0.902632
    [126]	valid_0's rmse: 0.902604
    [127]	valid_0's rmse: 0.90252
    [128]	valid_0's rmse: 0.90242
    [129]	valid_0's rmse: 0.902319
    [130]	valid_0's rmse: 0.902296
    [131]	valid_0's rmse: 0.902219
    [132]	valid_0's rmse: 0.902164
    [133]	valid_0's rmse: 0.902129
    [134]	valid_0's rmse: 0.902102
    [135]	valid_0's rmse: 0.902061
    [136]	valid_0's rmse: 0.901999
    [137]	valid_0's rmse: 0.901964
    [138]	valid_0's rmse: 0.901915
    [139]	valid_0's rmse: 0.901836
    [140]	valid_0's rmse: 0.901821
    [141]	valid_0's rmse: 0.901785
    [142]	valid_0's rmse: 0.901757
    [143]	valid_0's rmse: 0.901719
    [144]	valid_0's rmse: 0.901675
    [145]	valid_0's rmse: 0.901644
    [146]	valid_0's rmse: 0.90157
    [147]	valid_0's rmse: 0.901538
    [148]	valid_0's rmse: 0.901518
    [149]	valid_0's rmse: 0.901467
    [150]	valid_0's rmse: 0.901449
    [151]	valid_0's rmse: 0.901366
    [152]	valid_0's rmse: 0.901289
    [153]	valid_0's rmse: 0.901238
    [154]	valid_0's rmse: 0.901136
    [155]	valid_0's rmse: 0.901067
    [156]	valid_0's rmse: 0.900979
    [157]	valid_0's rmse: 0.900871
    [158]	valid_0's rmse: 0.900785
    [159]	valid_0's rmse: 0.900741
    [160]	valid_0's rmse: 0.900714
    [161]	valid_0's rmse: 0.900676
    [162]	valid_0's rmse: 0.900668
    [163]	valid_0's rmse: 0.900622
    [164]	valid_0's rmse: 0.900597
    [165]	valid_0's rmse: 0.900561
    [166]	valid_0's rmse: 0.90052
    [167]	valid_0's rmse: 0.900506
    [168]	valid_0's rmse: 0.900411
    [169]	valid_0's rmse: 0.900378
    [170]	valid_0's rmse: 0.900352
    [171]	valid_0's rmse: 0.900319
    [172]	valid_0's rmse: 0.900257
    [173]	valid_0's rmse: 0.90024
    [174]	valid_0's rmse: 0.900158
    [175]	valid_0's rmse: 0.900111
    [176]	valid_0's rmse: 0.900097
    [177]	valid_0's rmse: 0.900078
    [178]	valid_0's rmse: 0.899998
    [179]	valid_0's rmse: 0.899972
    [180]	valid_0's rmse: 0.899934
    [181]	valid_0's rmse: 0.899909
    [182]	valid_0's rmse: 0.899881
    [183]	valid_0's rmse: 0.899844
    [184]	valid_0's rmse: 0.899809
    [185]	valid_0's rmse: 0.899783
    [186]	valid_0's rmse: 0.899718
    [187]	valid_0's rmse: 0.899672
    [188]	valid_0's rmse: 0.899641
    [189]	valid_0's rmse: 0.899614
    [190]	valid_0's rmse: 0.899544
    [191]	valid_0's rmse: 0.899502
    [192]	valid_0's rmse: 0.899473
    [193]	valid_0's rmse: 0.899457
    [194]	valid_0's rmse: 0.899401
    [195]	valid_0's rmse: 0.899377
    [196]	valid_0's rmse: 0.899336
    [197]	valid_0's rmse: 0.899292
    [198]	valid_0's rmse: 0.899285
    [199]	valid_0's rmse: 0.899263
    [200]	valid_0's rmse: 0.899176
    [201]	valid_0's rmse: 0.899166
    [202]	valid_0's rmse: 0.899154
    [203]	valid_0's rmse: 0.899139
    [204]	valid_0's rmse: 0.899125
    [205]	valid_0's rmse: 0.899092
    [206]	valid_0's rmse: 0.899069
    [207]	valid_0's rmse: 0.899048
    [208]	valid_0's rmse: 0.899002
    [209]	valid_0's rmse: 0.898905
    [210]	valid_0's rmse: 0.898816
    [211]	valid_0's rmse: 0.898774
    [212]	valid_0's rmse: 0.898763
    [213]	valid_0's rmse: 0.898737
    [214]	valid_0's rmse: 0.898731
    [215]	valid_0's rmse: 0.898715
    [216]	valid_0's rmse: 0.89868
    [217]	valid_0's rmse: 0.898633
    [218]	valid_0's rmse: 0.898601
    [219]	valid_0's rmse: 0.898598
    [220]	valid_0's rmse: 0.898518
    [221]	valid_0's rmse: 0.898506
    [222]	valid_0's rmse: 0.898476
    [223]	valid_0's rmse: 0.898462
    [224]	valid_0's rmse: 0.898407
    [225]	valid_0's rmse: 0.898406
    [226]	valid_0's rmse: 0.898353
    [227]	valid_0's rmse: 0.898342
    [228]	valid_0's rmse: 0.898334
    [229]	valid_0's rmse: 0.898328
    [230]	valid_0's rmse: 0.898246
    [231]	valid_0's rmse: 0.898191
    [232]	valid_0's rmse: 0.898179
    [233]	valid_0's rmse: 0.89816
    [234]	valid_0's rmse: 0.898071
    [235]	valid_0's rmse: 0.898078
    [236]	valid_0's rmse: 0.898025
    [237]	valid_0's rmse: 0.898018
    [238]	valid_0's rmse: 0.897952
    [239]	valid_0's rmse: 0.897939
    [240]	valid_0's rmse: 0.897915
    [241]	valid_0's rmse: 0.897877
    [242]	valid_0's rmse: 0.897868
    [243]	valid_0's rmse: 0.897817
    [244]	valid_0's rmse: 0.897795
    [245]	valid_0's rmse: 0.897737
    [246]	valid_0's rmse: 0.897726
    [247]	valid_0's rmse: 0.897708
    [248]	valid_0's rmse: 0.897697
    [249]	valid_0's rmse: 0.897687
    [250]	valid_0's rmse: 0.897582
    [251]	valid_0's rmse: 0.897563
    [252]	valid_0's rmse: 0.897547
    [253]	valid_0's rmse: 0.897533
    [254]	valid_0's rmse: 0.897504
    [255]	valid_0's rmse: 0.897491
    [256]	valid_0's rmse: 0.897455
    [257]	valid_0's rmse: 0.897422
    [258]	valid_0's rmse: 0.897412
    [259]	valid_0's rmse: 0.897381
    [260]	valid_0's rmse: 0.89738
    [261]	valid_0's rmse: 0.897382
    [262]	valid_0's rmse: 0.897351
    [263]	valid_0's rmse: 0.897336
    [264]	valid_0's rmse: 0.897336
    [265]	valid_0's rmse: 0.897312
    [266]	valid_0's rmse: 0.897295
    [267]	valid_0's rmse: 0.897272
    [268]	valid_0's rmse: 0.897264
    [269]	valid_0's rmse: 0.897212
    [270]	valid_0's rmse: 0.897199
    [271]	valid_0's rmse: 0.897193
    [272]	valid_0's rmse: 0.897181
    [273]	valid_0's rmse: 0.897178
    [274]	valid_0's rmse: 0.897173
    [275]	valid_0's rmse: 0.897096
    [276]	valid_0's rmse: 0.89709
    [277]	valid_0's rmse: 0.897064
    [278]	valid_0's rmse: 0.897038
    [279]	valid_0's rmse: 0.897043
    [280]	valid_0's rmse: 0.897035
    [281]	valid_0's rmse: 0.896997
    [282]	valid_0's rmse: 0.896984
    [283]	valid_0's rmse: 0.896972
    [284]	valid_0's rmse: 0.89694
    [285]	valid_0's rmse: 0.896907
    [286]	valid_0's rmse: 0.89687
    [287]	valid_0's rmse: 0.896875
    [288]	valid_0's rmse: 0.896871
    [289]	valid_0's rmse: 0.896859
    [290]	valid_0's rmse: 0.896828
    [291]	valid_0's rmse: 0.896775
    [292]	valid_0's rmse: 0.896775
    [293]	valid_0's rmse: 0.896773
    [294]	valid_0's rmse: 0.896748
    [295]	valid_0's rmse: 0.896741
    [296]	valid_0's rmse: 0.896677
    [297]	valid_0's rmse: 0.896669
    [298]	valid_0's rmse: 0.896655
    [299]	valid_0's rmse: 0.896635
    [300]	valid_0's rmse: 0.896614
    [301]	valid_0's rmse: 0.896606
    [302]	valid_0's rmse: 0.896554
    [303]	valid_0's rmse: 0.896538
    [304]	valid_0's rmse: 0.89654
    [305]	valid_0's rmse: 0.896522
    [306]	valid_0's rmse: 0.896501
    [1]	valid_0's rmse: 0.918777
    [2]	valid_0's rmse: 0.91793
    [3]	valid_0's rmse: 0.917173
    [4]	valid_0's rmse: 0.916527
    [5]	valid_0's rmse: 0.915915
    [6]	valid_0's rmse: 0.915355
    [7]	valid_0's rmse: 0.914879
    [8]	valid_0's rmse: 0.914472
    [9]	valid_0's rmse: 0.914056
    [10]	valid_0's rmse: 0.91372
    [11]	valid_0's rmse: 0.913411
    [12]	valid_0's rmse: 0.91313
    [13]	valid_0's rmse: 0.91286
    [14]	valid_0's rmse: 0.91261
    [15]	valid_0's rmse: 0.912379
    [16]	valid_0's rmse: 0.912136
    [17]	valid_0's rmse: 0.911928
    [18]	valid_0's rmse: 0.91172
    [19]	valid_0's rmse: 0.911497
    [20]	valid_0's rmse: 0.911297
    [21]	valid_0's rmse: 0.911103
    [22]	valid_0's rmse: 0.910858
    [23]	valid_0's rmse: 0.910659
    [24]	valid_0's rmse: 0.910504
    [25]	valid_0's rmse: 0.91033
    [26]	valid_0's rmse: 0.910175
    [27]	valid_0's rmse: 0.909897
    [28]	valid_0's rmse: 0.909745
    [29]	valid_0's rmse: 0.909608
    [30]	valid_0's rmse: 0.909485
    [31]	valid_0's rmse: 0.90929
    [32]	valid_0's rmse: 0.909133
    [33]	valid_0's rmse: 0.908942
    [34]	valid_0's rmse: 0.908811
    [35]	valid_0's rmse: 0.908696
    [36]	valid_0's rmse: 0.908511
    [37]	valid_0's rmse: 0.908427
    [38]	valid_0's rmse: 0.908291
    [39]	valid_0's rmse: 0.908144
    [40]	valid_0's rmse: 0.908038
    [41]	valid_0's rmse: 0.907885
    [42]	valid_0's rmse: 0.907756
    [43]	valid_0's rmse: 0.907554
    [44]	valid_0's rmse: 0.907394
    [45]	valid_0's rmse: 0.907269
    [46]	valid_0's rmse: 0.90717
    [47]	valid_0's rmse: 0.907039
    [48]	valid_0's rmse: 0.906894
    [49]	valid_0's rmse: 0.906814
    [50]	valid_0's rmse: 0.906708
    [51]	valid_0's rmse: 0.906625
    [52]	valid_0's rmse: 0.906527
    [53]	valid_0's rmse: 0.906394
    [54]	valid_0's rmse: 0.906257
    [55]	valid_0's rmse: 0.906158
    [56]	valid_0's rmse: 0.906072
    [57]	valid_0's rmse: 0.905966
    [58]	valid_0's rmse: 0.905868
    [59]	valid_0's rmse: 0.905776
    [60]	valid_0's rmse: 0.905694
    [61]	valid_0's rmse: 0.905553
    [62]	valid_0's rmse: 0.90541
    [63]	valid_0's rmse: 0.905348
    [64]	valid_0's rmse: 0.905252
    [65]	valid_0's rmse: 0.905147
    [66]	valid_0's rmse: 0.905064
    [67]	valid_0's rmse: 0.904955
    [68]	valid_0's rmse: 0.904884
    [69]	valid_0's rmse: 0.904807
    [70]	valid_0's rmse: 0.90474
    [71]	valid_0's rmse: 0.904666
    [72]	valid_0's rmse: 0.904561
    [73]	valid_0's rmse: 0.904511
    [74]	valid_0's rmse: 0.904426
    [75]	valid_0's rmse: 0.90423
    [76]	valid_0's rmse: 0.904166
    [77]	valid_0's rmse: 0.904096
    [78]	valid_0's rmse: 0.904037
    [79]	valid_0's rmse: 0.903973
    [80]	valid_0's rmse: 0.903912
    [81]	valid_0's rmse: 0.903793
    [82]	valid_0's rmse: 0.903639
    [83]	valid_0's rmse: 0.903571
    [84]	valid_0's rmse: 0.903508
    [85]	valid_0's rmse: 0.903417
    [86]	valid_0's rmse: 0.90337
    [87]	valid_0's rmse: 0.903294
    [88]	valid_0's rmse: 0.903167
    [89]	valid_0's rmse: 0.9031
    [90]	valid_0's rmse: 0.903038
    [91]	valid_0's rmse: 0.90298
    [92]	valid_0's rmse: 0.902827
    [93]	valid_0's rmse: 0.902698
    [94]	valid_0's rmse: 0.902645
    [95]	valid_0's rmse: 0.902566
    [96]	valid_0's rmse: 0.902495
    [97]	valid_0's rmse: 0.902439
    [98]	valid_0's rmse: 0.90236
    [99]	valid_0's rmse: 0.902308
    [100]	valid_0's rmse: 0.902258
    [101]	valid_0's rmse: 0.902218
    [102]	valid_0's rmse: 0.902075
    [103]	valid_0's rmse: 0.901992
    [104]	valid_0's rmse: 0.901933
    [105]	valid_0's rmse: 0.901892
    [106]	valid_0's rmse: 0.90185
    [107]	valid_0's rmse: 0.901795
    [108]	valid_0's rmse: 0.901728
    [109]	valid_0's rmse: 0.901675
    [110]	valid_0's rmse: 0.901629
    [111]	valid_0's rmse: 0.90156
    [112]	valid_0's rmse: 0.9015
    [113]	valid_0's rmse: 0.901458
    [114]	valid_0's rmse: 0.90144
    [115]	valid_0's rmse: 0.901372
    [116]	valid_0's rmse: 0.90132
    [117]	valid_0's rmse: 0.90128
    [118]	valid_0's rmse: 0.901224
    [119]	valid_0's rmse: 0.90116
    [120]	valid_0's rmse: 0.90114
    [121]	valid_0's rmse: 0.901087
    [122]	valid_0's rmse: 0.901057
    [123]	valid_0's rmse: 0.90101
    [124]	valid_0's rmse: 0.900904
    [125]	valid_0's rmse: 0.900853
    [126]	valid_0's rmse: 0.900827
    [127]	valid_0's rmse: 0.900742
    [128]	valid_0's rmse: 0.900697
    [129]	valid_0's rmse: 0.900685
    [130]	valid_0's rmse: 0.900591
    [131]	valid_0's rmse: 0.900537
    [132]	valid_0's rmse: 0.900478
    [133]	valid_0's rmse: 0.900448
    [134]	valid_0's rmse: 0.900331
    [135]	valid_0's rmse: 0.900284
    [136]	valid_0's rmse: 0.9002
    [137]	valid_0's rmse: 0.900101
    [138]	valid_0's rmse: 0.90001
    [139]	valid_0's rmse: 0.899962
    [140]	valid_0's rmse: 0.899934
    [141]	valid_0's rmse: 0.899915
    [142]	valid_0's rmse: 0.899867
    [143]	valid_0's rmse: 0.899819
    [144]	valid_0's rmse: 0.899687
    [145]	valid_0's rmse: 0.89961
    [146]	valid_0's rmse: 0.899527
    [147]	valid_0's rmse: 0.899488
    [148]	valid_0's rmse: 0.899463
    [149]	valid_0's rmse: 0.899353
    [150]	valid_0's rmse: 0.899283
    [151]	valid_0's rmse: 0.899254
    [152]	valid_0's rmse: 0.89918
    [153]	valid_0's rmse: 0.899152
    [154]	valid_0's rmse: 0.899076
    [155]	valid_0's rmse: 0.899054
    [156]	valid_0's rmse: 0.899034
    [157]	valid_0's rmse: 0.898996
    [158]	valid_0's rmse: 0.898955
    [159]	valid_0's rmse: 0.898892
    [160]	valid_0's rmse: 0.898878
    [161]	valid_0's rmse: 0.898844
    [162]	valid_0's rmse: 0.898828
    [163]	valid_0's rmse: 0.898788
    [164]	valid_0's rmse: 0.898762
    [165]	valid_0's rmse: 0.898701
    [166]	valid_0's rmse: 0.898684
    [167]	valid_0's rmse: 0.898654
    [168]	valid_0's rmse: 0.898605
    [169]	valid_0's rmse: 0.898508
    [170]	valid_0's rmse: 0.898448
    [171]	valid_0's rmse: 0.898397
    [172]	valid_0's rmse: 0.898351
    [173]	valid_0's rmse: 0.898328
    [174]	valid_0's rmse: 0.898298
    [175]	valid_0's rmse: 0.898234
    [176]	valid_0's rmse: 0.898189
    [177]	valid_0's rmse: 0.898162
    [178]	valid_0's rmse: 0.898134
    [179]	valid_0's rmse: 0.898071
    [180]	valid_0's rmse: 0.898054
    [181]	valid_0's rmse: 0.898029
    [182]	valid_0's rmse: 0.897983
    [183]	valid_0's rmse: 0.897955
    [184]	valid_0's rmse: 0.897928
    [185]	valid_0's rmse: 0.89786
    [186]	valid_0's rmse: 0.897853
    [187]	valid_0's rmse: 0.897794
    [188]	valid_0's rmse: 0.897775
    [189]	valid_0's rmse: 0.897719
    [190]	valid_0's rmse: 0.897704
    [191]	valid_0's rmse: 0.897682
    [192]	valid_0's rmse: 0.897665
    [193]	valid_0's rmse: 0.897622
    [194]	valid_0's rmse: 0.897556
    [195]	valid_0's rmse: 0.897554
    [196]	valid_0's rmse: 0.897535
    [197]	valid_0's rmse: 0.897522
    [198]	valid_0's rmse: 0.897503
    [199]	valid_0's rmse: 0.897469
    [200]	valid_0's rmse: 0.897426
    [201]	valid_0's rmse: 0.897395
    [202]	valid_0's rmse: 0.897369
    [203]	valid_0's rmse: 0.89734
    [204]	valid_0's rmse: 0.897318
    [205]	valid_0's rmse: 0.897256
    [206]	valid_0's rmse: 0.897247
    [207]	valid_0's rmse: 0.897161
    [208]	valid_0's rmse: 0.897119
    [209]	valid_0's rmse: 0.897119
    [210]	valid_0's rmse: 0.897059
    [211]	valid_0's rmse: 0.897039
    [212]	valid_0's rmse: 0.897007
    [213]	valid_0's rmse: 0.896935
    [214]	valid_0's rmse: 0.896876
    [215]	valid_0's rmse: 0.89685
    [216]	valid_0's rmse: 0.896842
    [217]	valid_0's rmse: 0.896805
    [218]	valid_0's rmse: 0.896743
    [219]	valid_0's rmse: 0.896732
    [220]	valid_0's rmse: 0.896706
    [221]	valid_0's rmse: 0.896646
    [222]	valid_0's rmse: 0.896622
    [223]	valid_0's rmse: 0.896589
    [224]	valid_0's rmse: 0.896563
    [225]	valid_0's rmse: 0.896539
    [226]	valid_0's rmse: 0.896532
    [227]	valid_0's rmse: 0.896516
    [228]	valid_0's rmse: 0.896524
    [229]	valid_0's rmse: 0.896504
    [230]	valid_0's rmse: 0.896487
    [231]	valid_0's rmse: 0.896439
    [232]	valid_0's rmse: 0.896403
    [233]	valid_0's rmse: 0.896366
    [234]	valid_0's rmse: 0.896336
    [235]	valid_0's rmse: 0.89632
    [236]	valid_0's rmse: 0.896249
    [237]	valid_0's rmse: 0.896222
    [238]	valid_0's rmse: 0.896195
    [239]	valid_0's rmse: 0.896184
    [240]	valid_0's rmse: 0.896173
    [241]	valid_0's rmse: 0.89618
    [242]	valid_0's rmse: 0.896135
    [243]	valid_0's rmse: 0.896139
    [244]	valid_0's rmse: 0.896117
    [245]	valid_0's rmse: 0.89608
    [246]	valid_0's rmse: 0.896023
    [247]	valid_0's rmse: 0.896016
    [248]	valid_0's rmse: 0.895979
    [249]	valid_0's rmse: 0.895962
    [250]	valid_0's rmse: 0.895949
    [251]	valid_0's rmse: 0.895837
    [252]	valid_0's rmse: 0.895828
    [253]	valid_0's rmse: 0.895795
    [254]	valid_0's rmse: 0.895745
    [255]	valid_0's rmse: 0.895651
    [256]	valid_0's rmse: 0.895638
    [257]	valid_0's rmse: 0.895621
    [258]	valid_0's rmse: 0.895574
    [259]	valid_0's rmse: 0.895562
    [260]	valid_0's rmse: 0.895546
    [261]	valid_0's rmse: 0.895531
    [262]	valid_0's rmse: 0.895522
    [263]	valid_0's rmse: 0.895507
    [264]	valid_0's rmse: 0.895493
    [265]	valid_0's rmse: 0.895465
    [266]	valid_0's rmse: 0.895422
    [267]	valid_0's rmse: 0.89539
    [268]	valid_0's rmse: 0.895342
    [269]	valid_0's rmse: 0.895328
    [270]	valid_0's rmse: 0.895323
    [271]	valid_0's rmse: 0.895301
    [272]	valid_0's rmse: 0.895258
    [273]	valid_0's rmse: 0.895241
    [274]	valid_0's rmse: 0.895235
    [275]	valid_0's rmse: 0.895229
    [276]	valid_0's rmse: 0.895191
    [277]	valid_0's rmse: 0.895161
    [278]	valid_0's rmse: 0.895106
    [279]	valid_0's rmse: 0.895093
    [280]	valid_0's rmse: 0.895078
    [281]	valid_0's rmse: 0.895068
    [282]	valid_0's rmse: 0.895043
    [283]	valid_0's rmse: 0.894994
    [284]	valid_0's rmse: 0.894964
    [285]	valid_0's rmse: 0.894959
    [286]	valid_0's rmse: 0.894922
    [287]	valid_0's rmse: 0.89493
    [288]	valid_0's rmse: 0.894894
    [289]	valid_0's rmse: 0.894884
    [290]	valid_0's rmse: 0.894875
    [291]	valid_0's rmse: 0.89487
    [292]	valid_0's rmse: 0.894851
    [293]	valid_0's rmse: 0.894842
    [294]	valid_0's rmse: 0.894822
    [295]	valid_0's rmse: 0.894804
    [296]	valid_0's rmse: 0.894799
    [297]	valid_0's rmse: 0.894776
    [298]	valid_0's rmse: 0.894765
    [299]	valid_0's rmse: 0.894755
    [300]	valid_0's rmse: 0.894664
    [301]	valid_0's rmse: 0.894628
    [302]	valid_0's rmse: 0.894605
    [303]	valid_0's rmse: 0.894574
    [304]	valid_0's rmse: 0.894553
    [305]	valid_0's rmse: 0.894519
    [306]	valid_0's rmse: 0.894516
    [1]	valid_0's rmse: 0.919976
    [2]	valid_0's rmse: 0.919058
    [3]	valid_0's rmse: 0.91826
    [4]	valid_0's rmse: 0.917576
    [5]	valid_0's rmse: 0.916948
    [6]	valid_0's rmse: 0.916393
    [7]	valid_0's rmse: 0.915873
    [8]	valid_0's rmse: 0.915444
    [9]	valid_0's rmse: 0.915022
    [10]	valid_0's rmse: 0.914656
    [11]	valid_0's rmse: 0.914337
    [12]	valid_0's rmse: 0.914036
    [13]	valid_0's rmse: 0.913727
    [14]	valid_0's rmse: 0.913456
    [15]	valid_0's rmse: 0.913214
    [16]	valid_0's rmse: 0.91298
    [17]	valid_0's rmse: 0.912739
    [18]	valid_0's rmse: 0.912529
    [19]	valid_0's rmse: 0.912274
    [20]	valid_0's rmse: 0.912067
    [21]	valid_0's rmse: 0.911865
    [22]	valid_0's rmse: 0.911659
    [23]	valid_0's rmse: 0.911332
    [24]	valid_0's rmse: 0.911107
    [25]	valid_0's rmse: 0.910939
    [26]	valid_0's rmse: 0.910779
    [27]	valid_0's rmse: 0.910621
    [28]	valid_0's rmse: 0.910476
    [29]	valid_0's rmse: 0.910282
    [30]	valid_0's rmse: 0.910129
    [31]	valid_0's rmse: 0.909937
    [32]	valid_0's rmse: 0.909804
    [33]	valid_0's rmse: 0.909608
    [34]	valid_0's rmse: 0.909466
    [35]	valid_0's rmse: 0.909297
    [36]	valid_0's rmse: 0.909142
    [37]	valid_0's rmse: 0.909021
    [38]	valid_0's rmse: 0.908828
    [39]	valid_0's rmse: 0.908748
    [40]	valid_0's rmse: 0.908627
    [41]	valid_0's rmse: 0.908524
    [42]	valid_0's rmse: 0.908361
    [43]	valid_0's rmse: 0.908232
    [44]	valid_0's rmse: 0.908144
    [45]	valid_0's rmse: 0.908013
    [46]	valid_0's rmse: 0.907873
    [47]	valid_0's rmse: 0.907755
    [48]	valid_0's rmse: 0.907673
    [49]	valid_0's rmse: 0.907474
    [50]	valid_0's rmse: 0.907386
    [51]	valid_0's rmse: 0.907293
    [52]	valid_0's rmse: 0.907131
    [53]	valid_0's rmse: 0.907023
    [54]	valid_0's rmse: 0.906922
    [55]	valid_0's rmse: 0.906866
    [56]	valid_0's rmse: 0.906788
    [57]	valid_0's rmse: 0.906677
    [58]	valid_0's rmse: 0.906567
    [59]	valid_0's rmse: 0.906409
    [60]	valid_0's rmse: 0.906231
    [61]	valid_0's rmse: 0.906132
    [62]	valid_0's rmse: 0.906044
    [63]	valid_0's rmse: 0.905931
    [64]	valid_0's rmse: 0.90583
    [65]	valid_0's rmse: 0.905776
    [66]	valid_0's rmse: 0.90569
    [67]	valid_0's rmse: 0.90561
    [68]	valid_0's rmse: 0.905532
    [69]	valid_0's rmse: 0.905435
    [70]	valid_0's rmse: 0.905372
    [71]	valid_0's rmse: 0.905303
    [72]	valid_0's rmse: 0.905228
    [73]	valid_0's rmse: 0.905149
    [74]	valid_0's rmse: 0.905035
    [75]	valid_0's rmse: 0.904944
    [76]	valid_0's rmse: 0.904884
    [77]	valid_0's rmse: 0.904765
    [78]	valid_0's rmse: 0.90473
    [79]	valid_0's rmse: 0.904684
    [80]	valid_0's rmse: 0.904503
    [81]	valid_0's rmse: 0.904346
    [82]	valid_0's rmse: 0.904273
    [83]	valid_0's rmse: 0.904127
    [84]	valid_0's rmse: 0.904076
    [85]	valid_0's rmse: 0.903949
    [86]	valid_0's rmse: 0.903864
    [87]	valid_0's rmse: 0.903802
    [88]	valid_0's rmse: 0.90368
    [89]	valid_0's rmse: 0.903594
    [90]	valid_0's rmse: 0.903539
    [91]	valid_0's rmse: 0.903435
    [92]	valid_0's rmse: 0.903318
    [93]	valid_0's rmse: 0.903235
    [94]	valid_0's rmse: 0.903131
    [95]	valid_0's rmse: 0.903051
    [96]	valid_0's rmse: 0.902976
    [97]	valid_0's rmse: 0.90291
    [98]	valid_0's rmse: 0.902828
    [99]	valid_0's rmse: 0.902767
    [100]	valid_0's rmse: 0.902711
    [101]	valid_0's rmse: 0.902656
    [102]	valid_0's rmse: 0.902607
    [103]	valid_0's rmse: 0.902557
    [104]	valid_0's rmse: 0.902478
    [105]	valid_0's rmse: 0.902397
    [106]	valid_0's rmse: 0.902303
    [107]	valid_0's rmse: 0.902236
    [108]	valid_0's rmse: 0.902184
    [109]	valid_0's rmse: 0.90214
    [110]	valid_0's rmse: 0.902049
    [111]	valid_0's rmse: 0.902011
    [112]	valid_0's rmse: 0.901917
    [113]	valid_0's rmse: 0.901817
    [114]	valid_0's rmse: 0.901731
    [115]	valid_0's rmse: 0.901674
    [116]	valid_0's rmse: 0.901612
    [117]	valid_0's rmse: 0.90158
    [118]	valid_0's rmse: 0.901542
    [119]	valid_0's rmse: 0.901519
    [120]	valid_0's rmse: 0.901463
    [121]	valid_0's rmse: 0.9014
    [122]	valid_0's rmse: 0.901254
    [123]	valid_0's rmse: 0.901187
    [124]	valid_0's rmse: 0.901139
    [125]	valid_0's rmse: 0.901106
    [126]	valid_0's rmse: 0.901062
    [127]	valid_0's rmse: 0.901023
    [128]	valid_0's rmse: 0.900895
    [129]	valid_0's rmse: 0.90086
    [130]	valid_0's rmse: 0.900803
    [131]	valid_0's rmse: 0.90078
    [132]	valid_0's rmse: 0.900742
    [133]	valid_0's rmse: 0.900694
    [134]	valid_0's rmse: 0.900553
    [135]	valid_0's rmse: 0.9005
    [136]	valid_0's rmse: 0.90046
    [137]	valid_0's rmse: 0.900435
    [138]	valid_0's rmse: 0.900394
    [139]	valid_0's rmse: 0.900305
    [140]	valid_0's rmse: 0.900216
    [141]	valid_0's rmse: 0.900187
    [142]	valid_0's rmse: 0.900126
    [143]	valid_0's rmse: 0.900087
    [144]	valid_0's rmse: 0.900014
    [145]	valid_0's rmse: 0.899929
    [146]	valid_0's rmse: 0.899891
    [147]	valid_0's rmse: 0.89987
    [148]	valid_0's rmse: 0.899757
    [149]	valid_0's rmse: 0.899738
    [150]	valid_0's rmse: 0.899628
    [151]	valid_0's rmse: 0.899602
    [152]	valid_0's rmse: 0.899541
    [153]	valid_0's rmse: 0.899523
    [154]	valid_0's rmse: 0.89948
    [155]	valid_0's rmse: 0.899416
    [156]	valid_0's rmse: 0.899307
    [157]	valid_0's rmse: 0.899256
    [158]	valid_0's rmse: 0.899207
    [159]	valid_0's rmse: 0.899177
    [160]	valid_0's rmse: 0.89914
    [161]	valid_0's rmse: 0.899094
    [162]	valid_0's rmse: 0.899081
    [163]	valid_0's rmse: 0.899009
    [164]	valid_0's rmse: 0.898992
    [165]	valid_0's rmse: 0.898964
    [166]	valid_0's rmse: 0.898892
    [167]	valid_0's rmse: 0.89881
    [168]	valid_0's rmse: 0.898788
    [169]	valid_0's rmse: 0.898746
    [170]	valid_0's rmse: 0.89866
    [171]	valid_0's rmse: 0.898661
    [172]	valid_0's rmse: 0.898571
    [173]	valid_0's rmse: 0.898513
    [174]	valid_0's rmse: 0.89849
    [175]	valid_0's rmse: 0.898449
    [176]	valid_0's rmse: 0.898414
    [177]	valid_0's rmse: 0.89838
    [178]	valid_0's rmse: 0.898351
    [179]	valid_0's rmse: 0.898326
    [180]	valid_0's rmse: 0.898308
    [181]	valid_0's rmse: 0.898274
    [182]	valid_0's rmse: 0.898246
    [183]	valid_0's rmse: 0.898208
    [184]	valid_0's rmse: 0.89812
    [185]	valid_0's rmse: 0.898021
    [186]	valid_0's rmse: 0.897997
    [187]	valid_0's rmse: 0.897945
    [188]	valid_0's rmse: 0.897924
    [189]	valid_0's rmse: 0.897887
    [190]	valid_0's rmse: 0.897853
    [191]	valid_0's rmse: 0.897821
    [192]	valid_0's rmse: 0.897795
    [193]	valid_0's rmse: 0.897758
    [194]	valid_0's rmse: 0.897684
    [195]	valid_0's rmse: 0.897636
    [196]	valid_0's rmse: 0.89763
    [197]	valid_0's rmse: 0.89759
    [198]	valid_0's rmse: 0.897563
    [199]	valid_0's rmse: 0.897513
    [200]	valid_0's rmse: 0.8975
    [201]	valid_0's rmse: 0.897444
    [202]	valid_0's rmse: 0.897417
    [203]	valid_0's rmse: 0.897363
    [204]	valid_0's rmse: 0.897334
    [205]	valid_0's rmse: 0.897304
    [206]	valid_0's rmse: 0.897269
    [207]	valid_0's rmse: 0.897259
    [208]	valid_0's rmse: 0.897228
    [209]	valid_0's rmse: 0.897182
    [210]	valid_0's rmse: 0.897177
    [211]	valid_0's rmse: 0.897162
    [212]	valid_0's rmse: 0.897101
    [213]	valid_0's rmse: 0.897087
    [214]	valid_0's rmse: 0.897012
    [215]	valid_0's rmse: 0.896997
    [216]	valid_0's rmse: 0.896992
    [217]	valid_0's rmse: 0.89699
    [218]	valid_0's rmse: 0.896985
    [219]	valid_0's rmse: 0.896917
    [220]	valid_0's rmse: 0.896897
    [221]	valid_0's rmse: 0.896842
    [222]	valid_0's rmse: 0.896836
    [223]	valid_0's rmse: 0.896824
    [224]	valid_0's rmse: 0.896808
    [225]	valid_0's rmse: 0.896772
    [226]	valid_0's rmse: 0.896753
    [227]	valid_0's rmse: 0.896744
    [228]	valid_0's rmse: 0.89668
    [229]	valid_0's rmse: 0.896654
    [230]	valid_0's rmse: 0.89662
    [231]	valid_0's rmse: 0.896583
    [232]	valid_0's rmse: 0.896549
    [233]	valid_0's rmse: 0.896538
    [234]	valid_0's rmse: 0.896502
    [235]	valid_0's rmse: 0.896459
    [236]	valid_0's rmse: 0.896407
    [237]	valid_0's rmse: 0.896339
    [238]	valid_0's rmse: 0.896299
    [239]	valid_0's rmse: 0.896259
    [240]	valid_0's rmse: 0.89622
    [241]	valid_0's rmse: 0.896188
    [242]	valid_0's rmse: 0.896136
    [243]	valid_0's rmse: 0.896098
    [244]	valid_0's rmse: 0.896094
    [245]	valid_0's rmse: 0.896079
    [246]	valid_0's rmse: 0.896079
    [247]	valid_0's rmse: 0.89606
    [248]	valid_0's rmse: 0.896009
    [249]	valid_0's rmse: 0.895997
    [250]	valid_0's rmse: 0.895954
    [251]	valid_0's rmse: 0.895955
    [252]	valid_0's rmse: 0.895948
    [253]	valid_0's rmse: 0.895903
    [254]	valid_0's rmse: 0.8959
    [255]	valid_0's rmse: 0.895891
    [256]	valid_0's rmse: 0.895864
    [257]	valid_0's rmse: 0.895852
    [258]	valid_0's rmse: 0.895842
    [259]	valid_0's rmse: 0.895809
    [260]	valid_0's rmse: 0.895771
    [261]	valid_0's rmse: 0.895707
    [262]	valid_0's rmse: 0.895643
    [263]	valid_0's rmse: 0.895597
    [264]	valid_0's rmse: 0.895517
    [265]	valid_0's rmse: 0.895509
    [266]	valid_0's rmse: 0.8955
    [267]	valid_0's rmse: 0.895479
    [268]	valid_0's rmse: 0.895483
    [269]	valid_0's rmse: 0.895471
    [270]	valid_0's rmse: 0.895447
    [271]	valid_0's rmse: 0.895398
    [272]	valid_0's rmse: 0.895381
    [273]	valid_0's rmse: 0.895373
    [274]	valid_0's rmse: 0.895369
    [275]	valid_0's rmse: 0.895319
    [276]	valid_0's rmse: 0.895307
    [277]	valid_0's rmse: 0.895276
    [278]	valid_0's rmse: 0.895269
    [279]	valid_0's rmse: 0.895266
    [280]	valid_0's rmse: 0.895244
    [281]	valid_0's rmse: 0.895241
    [282]	valid_0's rmse: 0.895215
    [283]	valid_0's rmse: 0.895178
    [284]	valid_0's rmse: 0.895144
    [285]	valid_0's rmse: 0.895136
    [286]	valid_0's rmse: 0.895104
    [287]	valid_0's rmse: 0.895074
    [288]	valid_0's rmse: 0.895073
    [289]	valid_0's rmse: 0.895024
    [290]	valid_0's rmse: 0.895021
    [291]	valid_0's rmse: 0.895015
    [292]	valid_0's rmse: 0.895007
    [293]	valid_0's rmse: 0.895008
    [294]	valid_0's rmse: 0.894992
    [295]	valid_0's rmse: 0.894991
    [296]	valid_0's rmse: 0.894974
    [297]	valid_0's rmse: 0.894943
    [298]	valid_0's rmse: 0.894931
    [299]	valid_0's rmse: 0.894895
    [300]	valid_0's rmse: 0.894899
    [301]	valid_0's rmse: 0.894874
    [302]	valid_0's rmse: 0.894876
    [303]	valid_0's rmse: 0.894842
    [304]	valid_0's rmse: 0.894811
    [305]	valid_0's rmse: 0.894817
    [306]	valid_0's rmse: 0.894809
    [1]	valid_0's rmse: 0.919233
    [2]	valid_0's rmse: 0.91839
    [3]	valid_0's rmse: 0.917646
    [4]	valid_0's rmse: 0.917019
    [5]	valid_0's rmse: 0.916408
    [6]	valid_0's rmse: 0.915847
    [7]	valid_0's rmse: 0.915377
    [8]	valid_0's rmse: 0.914957
    [9]	valid_0's rmse: 0.914555
    [10]	valid_0's rmse: 0.914211
    [11]	valid_0's rmse: 0.913891
    [12]	valid_0's rmse: 0.913617
    [13]	valid_0's rmse: 0.913342
    [14]	valid_0's rmse: 0.913083
    [15]	valid_0's rmse: 0.912788
    [16]	valid_0's rmse: 0.912533
    [17]	valid_0's rmse: 0.912312
    [18]	valid_0's rmse: 0.912121
    [19]	valid_0's rmse: 0.911918
    [20]	valid_0's rmse: 0.91171
    [21]	valid_0's rmse: 0.911539
    [22]	valid_0's rmse: 0.911372
    [23]	valid_0's rmse: 0.91119
    [24]	valid_0's rmse: 0.911043
    [25]	valid_0's rmse: 0.910872
    [26]	valid_0's rmse: 0.910699
    [27]	valid_0's rmse: 0.910548
    [28]	valid_0's rmse: 0.910424
    [29]	valid_0's rmse: 0.910296
    [30]	valid_0's rmse: 0.910149
    [31]	valid_0's rmse: 0.910029
    [32]	valid_0's rmse: 0.909818
    [33]	valid_0's rmse: 0.909591
    [34]	valid_0's rmse: 0.90947
    [35]	valid_0's rmse: 0.909341
    [36]	valid_0's rmse: 0.909213
    [37]	valid_0's rmse: 0.909097
    [38]	valid_0's rmse: 0.908962
    [39]	valid_0's rmse: 0.908849
    [40]	valid_0's rmse: 0.908716
    [41]	valid_0's rmse: 0.908609
    [42]	valid_0's rmse: 0.908409
    [43]	valid_0's rmse: 0.908289
    [44]	valid_0's rmse: 0.908133
    [45]	valid_0's rmse: 0.908016
    [46]	valid_0's rmse: 0.907892
    [47]	valid_0's rmse: 0.90773
    [48]	valid_0's rmse: 0.907621
    [49]	valid_0's rmse: 0.907518
    [50]	valid_0's rmse: 0.907434
    [51]	valid_0's rmse: 0.907327
    [52]	valid_0's rmse: 0.907214
    [53]	valid_0's rmse: 0.907106
    [54]	valid_0's rmse: 0.907
    [55]	valid_0's rmse: 0.906892
    [56]	valid_0's rmse: 0.906767
    [57]	valid_0's rmse: 0.906677
    [58]	valid_0's rmse: 0.906592
    [59]	valid_0's rmse: 0.906479
    [60]	valid_0's rmse: 0.906327
    [61]	valid_0's rmse: 0.906251
    [62]	valid_0's rmse: 0.906174
    [63]	valid_0's rmse: 0.906078
    [64]	valid_0's rmse: 0.906009
    [65]	valid_0's rmse: 0.905851
    [66]	valid_0's rmse: 0.905805
    [67]	valid_0's rmse: 0.905723
    [68]	valid_0's rmse: 0.905636
    [69]	valid_0's rmse: 0.905557
    [70]	valid_0's rmse: 0.905487
    [71]	valid_0's rmse: 0.905412
    [72]	valid_0's rmse: 0.90531
    [73]	valid_0's rmse: 0.905236
    [74]	valid_0's rmse: 0.905104
    [75]	valid_0's rmse: 0.905055
    [76]	valid_0's rmse: 0.904977
    [77]	valid_0's rmse: 0.904894
    [78]	valid_0's rmse: 0.904834
    [79]	valid_0's rmse: 0.904763
    [80]	valid_0's rmse: 0.904696
    [81]	valid_0's rmse: 0.904565
    [82]	valid_0's rmse: 0.904485
    [83]	valid_0's rmse: 0.90438
    [84]	valid_0's rmse: 0.904323
    [85]	valid_0's rmse: 0.904255
    [86]	valid_0's rmse: 0.904183
    [87]	valid_0's rmse: 0.904114
    [88]	valid_0's rmse: 0.903954
    [89]	valid_0's rmse: 0.903885
    [90]	valid_0's rmse: 0.903815
    [91]	valid_0's rmse: 0.903683
    [92]	valid_0's rmse: 0.903621
    [93]	valid_0's rmse: 0.903531
    [94]	valid_0's rmse: 0.903485
    [95]	valid_0's rmse: 0.903429
    [96]	valid_0's rmse: 0.903401
    [97]	valid_0's rmse: 0.903358
    [98]	valid_0's rmse: 0.903285
    [99]	valid_0's rmse: 0.903182
    [100]	valid_0's rmse: 0.903087
    [101]	valid_0's rmse: 0.903031
    [102]	valid_0's rmse: 0.90295
    [103]	valid_0's rmse: 0.902907
    [104]	valid_0's rmse: 0.902725
    [105]	valid_0's rmse: 0.902652
    [106]	valid_0's rmse: 0.902565
    [107]	valid_0's rmse: 0.902517
    [108]	valid_0's rmse: 0.902446
    [109]	valid_0's rmse: 0.902365
    [110]	valid_0's rmse: 0.90231
    [111]	valid_0's rmse: 0.902259
    [112]	valid_0's rmse: 0.902176
    [113]	valid_0's rmse: 0.90204
    [114]	valid_0's rmse: 0.901987
    [115]	valid_0's rmse: 0.901958
    [116]	valid_0's rmse: 0.901904
    [117]	valid_0's rmse: 0.901877
    [118]	valid_0's rmse: 0.901848
    [119]	valid_0's rmse: 0.9018
    [120]	valid_0's rmse: 0.901674
    [121]	valid_0's rmse: 0.901636
    [122]	valid_0's rmse: 0.901601
    [123]	valid_0's rmse: 0.901534
    [124]	valid_0's rmse: 0.90142
    [125]	valid_0's rmse: 0.901364
    [126]	valid_0's rmse: 0.901337
    [127]	valid_0's rmse: 0.901222
    [128]	valid_0's rmse: 0.901185
    [129]	valid_0's rmse: 0.901117
    [130]	valid_0's rmse: 0.901046
    [131]	valid_0's rmse: 0.901018
    [132]	valid_0's rmse: 0.900979
    [133]	valid_0's rmse: 0.900909
    [134]	valid_0's rmse: 0.900801
    [135]	valid_0's rmse: 0.90074
    [136]	valid_0's rmse: 0.900692
    [137]	valid_0's rmse: 0.900547
    [138]	valid_0's rmse: 0.90048
    [139]	valid_0's rmse: 0.900401
    [140]	valid_0's rmse: 0.900335
    [141]	valid_0's rmse: 0.900311
    [142]	valid_0's rmse: 0.900232
    [143]	valid_0's rmse: 0.900215
    [144]	valid_0's rmse: 0.900172
    [145]	valid_0's rmse: 0.900057
    [146]	valid_0's rmse: 0.899947
    [147]	valid_0's rmse: 0.899925
    [148]	valid_0's rmse: 0.899837
    [149]	valid_0's rmse: 0.899778
    [150]	valid_0's rmse: 0.899708
    [151]	valid_0's rmse: 0.89967
    [152]	valid_0's rmse: 0.89964
    [153]	valid_0's rmse: 0.899481
    [154]	valid_0's rmse: 0.899467
    [155]	valid_0's rmse: 0.899452
    [156]	valid_0's rmse: 0.899424
    [157]	valid_0's rmse: 0.899388
    [158]	valid_0's rmse: 0.899349
    [159]	valid_0's rmse: 0.899319
    [160]	valid_0's rmse: 0.899301
    [161]	valid_0's rmse: 0.899274
    [162]	valid_0's rmse: 0.899203
    [163]	valid_0's rmse: 0.899171
    [164]	valid_0's rmse: 0.899159
    [165]	valid_0's rmse: 0.89912
    [166]	valid_0's rmse: 0.899044
    [167]	valid_0's rmse: 0.898932
    [168]	valid_0's rmse: 0.898845
    [169]	valid_0's rmse: 0.898826
    [170]	valid_0's rmse: 0.898793
    [171]	valid_0's rmse: 0.898771
    [172]	valid_0's rmse: 0.898664
    [173]	valid_0's rmse: 0.898623
    [174]	valid_0's rmse: 0.898605
    [175]	valid_0's rmse: 0.898599
    [176]	valid_0's rmse: 0.898542
    [177]	valid_0's rmse: 0.898532
    [178]	valid_0's rmse: 0.898505
    [179]	valid_0's rmse: 0.898488
    [180]	valid_0's rmse: 0.898464
    [181]	valid_0's rmse: 0.898416
    [182]	valid_0's rmse: 0.898355
    [183]	valid_0's rmse: 0.898339
    [184]	valid_0's rmse: 0.898215
    [185]	valid_0's rmse: 0.898169
    [186]	valid_0's rmse: 0.898122
    [187]	valid_0's rmse: 0.898088
    [188]	valid_0's rmse: 0.898066
    [189]	valid_0's rmse: 0.898016
    [190]	valid_0's rmse: 0.898007
    [191]	valid_0's rmse: 0.897946
    [192]	valid_0's rmse: 0.897931
    [193]	valid_0's rmse: 0.89785
    [194]	valid_0's rmse: 0.897782
    [195]	valid_0's rmse: 0.897756
    [196]	valid_0's rmse: 0.897728
    [197]	valid_0's rmse: 0.897678
    [198]	valid_0's rmse: 0.897645
    [199]	valid_0's rmse: 0.897584
    [200]	valid_0's rmse: 0.897583
    [201]	valid_0's rmse: 0.897542
    [202]	valid_0's rmse: 0.897523
    [203]	valid_0's rmse: 0.89751
    [204]	valid_0's rmse: 0.897498
    [205]	valid_0's rmse: 0.897479
    [206]	valid_0's rmse: 0.89745
    [207]	valid_0's rmse: 0.897397
    [208]	valid_0's rmse: 0.897326
    [209]	valid_0's rmse: 0.897316
    [210]	valid_0's rmse: 0.897299
    [211]	valid_0's rmse: 0.897223
    [212]	valid_0's rmse: 0.897183
    [213]	valid_0's rmse: 0.897181
    [214]	valid_0's rmse: 0.897148
    [215]	valid_0's rmse: 0.897145
    [216]	valid_0's rmse: 0.897131
    [217]	valid_0's rmse: 0.897129
    [218]	valid_0's rmse: 0.897071
    [219]	valid_0's rmse: 0.897034
    [220]	valid_0's rmse: 0.896999
    [221]	valid_0's rmse: 0.896965
    [222]	valid_0's rmse: 0.896937
    [223]	valid_0's rmse: 0.89691
    [224]	valid_0's rmse: 0.896871
    [225]	valid_0's rmse: 0.896856
    [226]	valid_0's rmse: 0.896833
    [227]	valid_0's rmse: 0.896822
    [228]	valid_0's rmse: 0.89677
    [229]	valid_0's rmse: 0.89675
    [230]	valid_0's rmse: 0.896732
    [231]	valid_0's rmse: 0.896719
    [232]	valid_0's rmse: 0.89668
    [233]	valid_0's rmse: 0.896655
    [234]	valid_0's rmse: 0.896647
    [235]	valid_0's rmse: 0.896644
    [236]	valid_0's rmse: 0.896576
    [237]	valid_0's rmse: 0.896565
    [238]	valid_0's rmse: 0.896516
    [239]	valid_0's rmse: 0.896457
    [240]	valid_0's rmse: 0.896443
    [241]	valid_0's rmse: 0.896415
    [242]	valid_0's rmse: 0.896325
    [243]	valid_0's rmse: 0.896302
    [244]	valid_0's rmse: 0.896283
    [245]	valid_0's rmse: 0.896279
    [246]	valid_0's rmse: 0.896259
    [247]	valid_0's rmse: 0.896244
    [248]	valid_0's rmse: 0.896213
    [249]	valid_0's rmse: 0.896192
    [250]	valid_0's rmse: 0.896189
    [251]	valid_0's rmse: 0.896172
    [252]	valid_0's rmse: 0.896048
    [253]	valid_0's rmse: 0.896029
    [254]	valid_0's rmse: 0.896008
    [255]	valid_0's rmse: 0.895979
    [256]	valid_0's rmse: 0.895971
    [257]	valid_0's rmse: 0.895931
    [258]	valid_0's rmse: 0.895875
    [259]	valid_0's rmse: 0.895878
    [260]	valid_0's rmse: 0.895867
    [261]	valid_0's rmse: 0.895867
    [262]	valid_0's rmse: 0.895858
    [263]	valid_0's rmse: 0.895858
    [264]	valid_0's rmse: 0.895849
    [265]	valid_0's rmse: 0.895829
    [266]	valid_0's rmse: 0.895796
    [267]	valid_0's rmse: 0.895787
    [268]	valid_0's rmse: 0.895779
    [269]	valid_0's rmse: 0.89574
    [270]	valid_0's rmse: 0.895732
    [271]	valid_0's rmse: 0.895724
    [272]	valid_0's rmse: 0.895707
    [273]	valid_0's rmse: 0.895706
    [274]	valid_0's rmse: 0.8957
    [275]	valid_0's rmse: 0.89567
    [276]	valid_0's rmse: 0.895638
    [277]	valid_0's rmse: 0.895626
    [278]	valid_0's rmse: 0.895618
    [279]	valid_0's rmse: 0.895567
    [280]	valid_0's rmse: 0.895465
    [281]	valid_0's rmse: 0.895419
    [282]	valid_0's rmse: 0.895411
    [283]	valid_0's rmse: 0.895384
    [284]	valid_0's rmse: 0.895372
    [285]	valid_0's rmse: 0.89537
    [286]	valid_0's rmse: 0.895356
    [287]	valid_0's rmse: 0.895344
    [288]	valid_0's rmse: 0.895349
    [289]	valid_0's rmse: 0.895332
    [290]	valid_0's rmse: 0.895325
    [291]	valid_0's rmse: 0.895238
    [292]	valid_0's rmse: 0.895177
    [293]	valid_0's rmse: 0.895158
    [294]	valid_0's rmse: 0.895119
    [295]	valid_0's rmse: 0.895061
    [296]	valid_0's rmse: 0.895027
    [297]	valid_0's rmse: 0.894951
    [298]	valid_0's rmse: 0.894955
    [299]	valid_0's rmse: 0.894939
    [300]	valid_0's rmse: 0.894927
    [301]	valid_0's rmse: 0.89489
    [302]	valid_0's rmse: 0.894854
    [303]	valid_0's rmse: 0.894845
    [304]	valid_0's rmse: 0.894828
    [305]	valid_0's rmse: 0.894816
    [306]	valid_0's rmse: 0.894816
    [1]	valid_0's rmse: 0.914877
    [2]	valid_0's rmse: 0.914011
    [3]	valid_0's rmse: 0.913235
    [4]	valid_0's rmse: 0.912568
    [5]	valid_0's rmse: 0.911966
    [6]	valid_0's rmse: 0.911439
    [7]	valid_0's rmse: 0.910917
    [8]	valid_0's rmse: 0.910482
    [9]	valid_0's rmse: 0.910094
    [10]	valid_0's rmse: 0.909749
    [11]	valid_0's rmse: 0.909439
    [12]	valid_0's rmse: 0.909143
    [13]	valid_0's rmse: 0.90887
    [14]	valid_0's rmse: 0.908618
    [15]	valid_0's rmse: 0.908398
    [16]	valid_0's rmse: 0.908201
    [17]	valid_0's rmse: 0.907986
    [18]	valid_0's rmse: 0.907771
    [19]	valid_0's rmse: 0.907554
    [20]	valid_0's rmse: 0.907363
    [21]	valid_0's rmse: 0.907155
    [22]	valid_0's rmse: 0.906979
    [23]	valid_0's rmse: 0.906748
    [24]	valid_0's rmse: 0.906561
    [25]	valid_0's rmse: 0.906411
    [26]	valid_0's rmse: 0.906226
    [27]	valid_0's rmse: 0.906072
    [28]	valid_0's rmse: 0.905937
    [29]	valid_0's rmse: 0.905748
    [30]	valid_0's rmse: 0.905566
    [31]	valid_0's rmse: 0.905416
    [32]	valid_0's rmse: 0.905201
    [33]	valid_0's rmse: 0.905061
    [34]	valid_0's rmse: 0.90491
    [35]	valid_0's rmse: 0.904807
    [36]	valid_0's rmse: 0.904691
    [37]	valid_0's rmse: 0.904578
    [38]	valid_0's rmse: 0.904473
    [39]	valid_0's rmse: 0.904327
    [40]	valid_0's rmse: 0.904215
    [41]	valid_0's rmse: 0.90411
    [42]	valid_0's rmse: 0.903962
    [43]	valid_0's rmse: 0.903776
    [44]	valid_0's rmse: 0.903646
    [45]	valid_0's rmse: 0.903506
    [46]	valid_0's rmse: 0.903418
    [47]	valid_0's rmse: 0.903355
    [48]	valid_0's rmse: 0.903246
    [49]	valid_0's rmse: 0.9031
    [50]	valid_0's rmse: 0.90303
    [51]	valid_0's rmse: 0.902898
    [52]	valid_0's rmse: 0.902738
    [53]	valid_0's rmse: 0.902628
    [54]	valid_0's rmse: 0.902458
    [55]	valid_0's rmse: 0.902336
    [56]	valid_0's rmse: 0.90222
    [57]	valid_0's rmse: 0.902166
    [58]	valid_0's rmse: 0.902054
    [59]	valid_0's rmse: 0.901949
    [60]	valid_0's rmse: 0.90183
    [61]	valid_0's rmse: 0.901753
    [62]	valid_0's rmse: 0.901663
    [63]	valid_0's rmse: 0.90156
    [64]	valid_0's rmse: 0.901444
    [65]	valid_0's rmse: 0.901366
    [66]	valid_0's rmse: 0.901291
    [67]	valid_0's rmse: 0.901242
    [68]	valid_0's rmse: 0.901159
    [69]	valid_0's rmse: 0.901032
    [70]	valid_0's rmse: 0.90091
    [71]	valid_0's rmse: 0.900786
    [72]	valid_0's rmse: 0.900709
    [73]	valid_0's rmse: 0.900588
    [74]	valid_0's rmse: 0.900518
    [75]	valid_0's rmse: 0.900422
    [76]	valid_0's rmse: 0.900301
    [77]	valid_0's rmse: 0.900246
    [78]	valid_0's rmse: 0.9002
    [79]	valid_0's rmse: 0.900153
    [80]	valid_0's rmse: 0.900076
    [81]	valid_0's rmse: 0.900025
    [82]	valid_0's rmse: 0.899945
    [83]	valid_0's rmse: 0.899859
    [84]	valid_0's rmse: 0.899767
    [85]	valid_0's rmse: 0.89965
    [86]	valid_0's rmse: 0.899589
    [87]	valid_0's rmse: 0.899532
    [88]	valid_0's rmse: 0.899407
    [89]	valid_0's rmse: 0.899341
    [90]	valid_0's rmse: 0.899275
    [91]	valid_0's rmse: 0.899223
    [92]	valid_0's rmse: 0.89918
    [93]	valid_0's rmse: 0.899094
    [94]	valid_0's rmse: 0.899023
    [95]	valid_0's rmse: 0.898971
    [96]	valid_0's rmse: 0.89891
    [97]	valid_0's rmse: 0.898855
    [98]	valid_0's rmse: 0.898774
    [99]	valid_0's rmse: 0.89861
    [100]	valid_0's rmse: 0.89844
    [101]	valid_0's rmse: 0.898405
    [102]	valid_0's rmse: 0.898237
    [103]	valid_0's rmse: 0.898165
    [104]	valid_0's rmse: 0.898107
    [105]	valid_0's rmse: 0.898081
    [106]	valid_0's rmse: 0.898022
    [107]	valid_0's rmse: 0.897859
    [108]	valid_0's rmse: 0.897812
    [109]	valid_0's rmse: 0.897764
    [110]	valid_0's rmse: 0.897684
    [111]	valid_0's rmse: 0.897616
    [112]	valid_0's rmse: 0.897534
    [113]	valid_0's rmse: 0.897499
    [114]	valid_0's rmse: 0.897387
    [115]	valid_0's rmse: 0.897287
    [116]	valid_0's rmse: 0.897244
    [117]	valid_0's rmse: 0.897204
    [118]	valid_0's rmse: 0.897146
    [119]	valid_0's rmse: 0.897027
    [120]	valid_0's rmse: 0.896981
    [121]	valid_0's rmse: 0.896942
    [122]	valid_0's rmse: 0.896901
    [123]	valid_0's rmse: 0.896845
    [124]	valid_0's rmse: 0.896808
    [125]	valid_0's rmse: 0.896755
    [126]	valid_0's rmse: 0.896716
    [127]	valid_0's rmse: 0.896673
    [128]	valid_0's rmse: 0.896574
    [129]	valid_0's rmse: 0.896512
    [130]	valid_0's rmse: 0.896425
    [131]	valid_0's rmse: 0.896392
    [132]	valid_0's rmse: 0.896341
    [133]	valid_0's rmse: 0.896295
    [134]	valid_0's rmse: 0.896258
    [135]	valid_0's rmse: 0.896192
    [136]	valid_0's rmse: 0.896096
    [137]	valid_0's rmse: 0.896026
    [138]	valid_0's rmse: 0.895917
    [139]	valid_0's rmse: 0.895884
    [140]	valid_0's rmse: 0.895859
    [141]	valid_0's rmse: 0.895797
    [142]	valid_0's rmse: 0.895726
    [143]	valid_0's rmse: 0.895709
    [144]	valid_0's rmse: 0.895673
    [145]	valid_0's rmse: 0.895608
    [146]	valid_0's rmse: 0.895543
    [147]	valid_0's rmse: 0.895533
    [148]	valid_0's rmse: 0.895513
    [149]	valid_0's rmse: 0.895456
    [150]	valid_0's rmse: 0.895425
    [151]	valid_0's rmse: 0.895364
    [152]	valid_0's rmse: 0.895343
    [153]	valid_0's rmse: 0.895247
    [154]	valid_0's rmse: 0.895222
    [155]	valid_0's rmse: 0.895185
    [156]	valid_0's rmse: 0.895158
    [157]	valid_0's rmse: 0.895142
    [158]	valid_0's rmse: 0.895102
    [159]	valid_0's rmse: 0.895075
    [160]	valid_0's rmse: 0.894994
    [161]	valid_0's rmse: 0.894967
    [162]	valid_0's rmse: 0.894895
    [163]	valid_0's rmse: 0.89485
    [164]	valid_0's rmse: 0.894817
    [165]	valid_0's rmse: 0.8948
    [166]	valid_0's rmse: 0.89475
    [167]	valid_0's rmse: 0.894664
    [168]	valid_0's rmse: 0.894616
    [169]	valid_0's rmse: 0.894589
    [170]	valid_0's rmse: 0.894523
    [171]	valid_0's rmse: 0.894513
    [172]	valid_0's rmse: 0.894482
    [173]	valid_0's rmse: 0.894464
    [174]	valid_0's rmse: 0.894438
    [175]	valid_0's rmse: 0.894339
    [176]	valid_0's rmse: 0.894313
    [177]	valid_0's rmse: 0.89425
    [178]	valid_0's rmse: 0.894236
    [179]	valid_0's rmse: 0.894121
    [180]	valid_0's rmse: 0.894005
    [181]	valid_0's rmse: 0.893971
    [182]	valid_0's rmse: 0.893924
    [183]	valid_0's rmse: 0.893891
    [184]	valid_0's rmse: 0.893853
    [185]	valid_0's rmse: 0.893806
    [186]	valid_0's rmse: 0.893765
    [187]	valid_0's rmse: 0.89376
    [188]	valid_0's rmse: 0.893719
    [189]	valid_0's rmse: 0.893651
    [190]	valid_0's rmse: 0.893582
    [191]	valid_0's rmse: 0.893546
    [192]	valid_0's rmse: 0.893526
    [193]	valid_0's rmse: 0.893498
    [194]	valid_0's rmse: 0.893426
    [195]	valid_0's rmse: 0.893396
    [196]	valid_0's rmse: 0.893381
    [197]	valid_0's rmse: 0.893344
    [198]	valid_0's rmse: 0.893298
    [199]	valid_0's rmse: 0.893286
    [200]	valid_0's rmse: 0.893267
    [201]	valid_0's rmse: 0.893248
    [202]	valid_0's rmse: 0.893219
    [203]	valid_0's rmse: 0.893213
    [204]	valid_0's rmse: 0.893184
    [205]	valid_0's rmse: 0.893181
    [206]	valid_0's rmse: 0.893156
    [207]	valid_0's rmse: 0.893146
    [208]	valid_0's rmse: 0.893112
    [209]	valid_0's rmse: 0.893065
    [210]	valid_0's rmse: 0.89303
    [211]	valid_0's rmse: 0.893018
    [212]	valid_0's rmse: 0.892964
    [213]	valid_0's rmse: 0.89296
    [214]	valid_0's rmse: 0.892926
    [215]	valid_0's rmse: 0.892897
    [216]	valid_0's rmse: 0.892855
    [217]	valid_0's rmse: 0.892836
    [218]	valid_0's rmse: 0.892841
    [219]	valid_0's rmse: 0.892766
    [220]	valid_0's rmse: 0.892752
    [221]	valid_0's rmse: 0.892733
    [222]	valid_0's rmse: 0.892723
    [223]	valid_0's rmse: 0.892692
    [224]	valid_0's rmse: 0.892688
    [225]	valid_0's rmse: 0.892617
    [226]	valid_0's rmse: 0.89259
    [227]	valid_0's rmse: 0.892588
    [228]	valid_0's rmse: 0.892567
    [229]	valid_0's rmse: 0.892457
    [230]	valid_0's rmse: 0.892436
    [231]	valid_0's rmse: 0.892428
    [232]	valid_0's rmse: 0.892424
    [233]	valid_0's rmse: 0.892388
    [234]	valid_0's rmse: 0.892337
    [235]	valid_0's rmse: 0.892336
    [236]	valid_0's rmse: 0.89232
    [237]	valid_0's rmse: 0.892296
    [238]	valid_0's rmse: 0.892229
    [239]	valid_0's rmse: 0.892157
    [240]	valid_0's rmse: 0.892148
    [241]	valid_0's rmse: 0.892144
    [242]	valid_0's rmse: 0.892104
    [243]	valid_0's rmse: 0.892097
    [244]	valid_0's rmse: 0.892081
    [245]	valid_0's rmse: 0.892073
    [246]	valid_0's rmse: 0.892079
    [247]	valid_0's rmse: 0.892073
    [248]	valid_0's rmse: 0.892038
    [249]	valid_0's rmse: 0.892026
    [250]	valid_0's rmse: 0.891995
    [251]	valid_0's rmse: 0.891947
    [252]	valid_0's rmse: 0.891892
    [253]	valid_0's rmse: 0.891836
    [254]	valid_0's rmse: 0.891824
    [255]	valid_0's rmse: 0.891782
    [256]	valid_0's rmse: 0.891765
    [257]	valid_0's rmse: 0.891759
    [258]	valid_0's rmse: 0.891741
    [259]	valid_0's rmse: 0.891729
    [260]	valid_0's rmse: 0.891726
    [261]	valid_0's rmse: 0.891696
    [262]	valid_0's rmse: 0.891696
    [263]	valid_0's rmse: 0.891627
    [264]	valid_0's rmse: 0.891579
    [265]	valid_0's rmse: 0.891554
    [266]	valid_0's rmse: 0.891539
    [267]	valid_0's rmse: 0.891508
    [268]	valid_0's rmse: 0.891479
    [269]	valid_0's rmse: 0.891465
    [270]	valid_0's rmse: 0.891428
    [271]	valid_0's rmse: 0.891416
    [272]	valid_0's rmse: 0.891396
    [273]	valid_0's rmse: 0.891372
    [274]	valid_0's rmse: 0.891344
    [275]	valid_0's rmse: 0.891294
    [276]	valid_0's rmse: 0.891282
    [277]	valid_0's rmse: 0.891276
    [278]	valid_0's rmse: 0.891221
    [279]	valid_0's rmse: 0.891207
    [280]	valid_0's rmse: 0.891183
    [281]	valid_0's rmse: 0.891161
    [282]	valid_0's rmse: 0.891149
    [283]	valid_0's rmse: 0.891137
    [284]	valid_0's rmse: 0.891135
    [285]	valid_0's rmse: 0.891129
    [286]	valid_0's rmse: 0.8911
    [287]	valid_0's rmse: 0.891078
    [288]	valid_0's rmse: 0.89107
    [289]	valid_0's rmse: 0.891064
    [290]	valid_0's rmse: 0.891047
    [291]	valid_0's rmse: 0.891003
    [292]	valid_0's rmse: 0.890991
    [293]	valid_0's rmse: 0.890959
    [294]	valid_0's rmse: 0.890927
    [295]	valid_0's rmse: 0.890889
    [296]	valid_0's rmse: 0.890886
    [297]	valid_0's rmse: 0.890878
    [298]	valid_0's rmse: 0.890845
    [299]	valid_0's rmse: 0.890825
    [300]	valid_0's rmse: 0.890816
    [301]	valid_0's rmse: 0.890802
    [302]	valid_0's rmse: 0.890791
    [303]	valid_0's rmse: 0.890742
    [304]	valid_0's rmse: 0.890725
    [305]	valid_0's rmse: 0.890716
    [306]	valid_0's rmse: 0.890708
    




    0



- 머신러닝을 통해 나온 결과로 feature 별 중요도 그래프


```python
import lightgbm
lightgbm.plot_importance(lgbm, figsize = (20, 60))
gc.collect()
```




    205




![png](/images/kaggle/ubiquant-2/output_37_1.png)


# Step 6. 예측 및 제출

- 주최측에서 제공하는 API를 통해 제출


```python
import ubiquant
env = ubiquant.make_env()   # initialize the environment
iter_test = env.iter_test()    # an iterator which loops over the test set and sample submission

```


```python
for (test_df, sample_prediction_df) in iter_test:
    
    test_df['target']  = 0
    
    for lgbm in models:
        test_df['target'] += lgbm.predict(test_df[feature_cols])
    test_df['target'] /= len(models)
    env.predict(test_df[['row_id','target']])
```

    This version of the API is not optimized and should not be used to estimate the runtime of your code on the hidden test set.
    

# 현 과정
https://www.kaggle.com/code/blackjjw/ubiquant-2

# 이전 과정
https://www.kaggle.com/code/blackjjw/ubiquant-1

# References :    
https://www.kaggle.com/code/yoshikuwano/fast-read-data-and-memory-optimization/notebook   
https://www.kaggle.com/code/rohanrao/tutorial-on-reading-large-datasets/notebook#Format:-csv   
https://www.kaggle.com/code/sytuannguyen/ubiquant-data-preparation#3.-Save-the-reduced-dataframe   
https://www.kaggle.com/code/jnegrini/ubiquant-eda    
https://www.kaggle.com/code/gunesevitan/ubiquant-market-prediction-eda/notebook#3.-Target   
https://www.kaggle.com/code/ilialar/ubiquant-eda-and-baseline#Features    
https://www.kaggle.com/competitions/ubiquant-market-prediction/data    
https://www.kaggle.com/code/larochemf/ubiquant-low-memory-use-be-careful/notebook    
https://www.kaggle.com/code/robikscube/fast-data-loading-and-low-mem-with-parquet-files/notebook    
https://www.kaggle.com/code/lucamassaron/eda-target-analysis       
https://www.kaggle.com/code/edwardcrookenden/eda-and-lgbm-baseline-feature-imp    
https://www.kaggle.com/competitions/ubiquant-market-prediction/discussion/305031    
https://www.kaggle.com/code/junjitakeshima/ubiquant-simple-lgbm-removing-outliers-en-jp/notebook#(1)-Read-Trainiing-Data    
https://www.kaggle.com/code/valleyzw/ubiquant-lgbm-baseline/notebook#Stock-market-calendar-analysis:-discussion     
https://www.kaggle.com/code/sytuannguyen/ubiquant-with-lightgbm   
https://www.kaggle.com/code/jiprud/simple-light-gbm   
https://www.kaggle.com/code/ilialar/ubiquant-eda-and-baseline#Model-training
