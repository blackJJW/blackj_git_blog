---
title: "[Python][Library] Datatable"
description: ""
date: "2022-06-08T16:00:45+09:00"
thumbnail: ""
categories:
  - "Python"
tags:
  - "Python"
  - "Python Library"
  - "Kaggle"


---
<!--more-->


![Untitled](/images/lang_python/library/Datatable/Untitled.png)

- **tabular 데이터**를 다루는 데 쓰이는 `Python`라이브러리
- *메모리 이상의 데이터셋*, 멀티 쓰레드 데이터 처리, 유연한 API 지원

## Installation

---

- **전제 조건**
    - `Python` 3.6+
        
        ```bash
        $ python --version
        ```
        
        ![Untitled](/images/lang_python/library/Datatable/Untitled%201.png)
    
    - `pip` version 20.3+
        
        ```bash
        $ pip install pip --upgrade
        ```

        ![Untitled](Datatable%201cf3e7ec7e74449ca1631c244195de95/Untitled%202.png)
        

- **기본 설치**
    - `pip`를 통한 설치
        
        ```bash
        pip install datatable
        ```
        
        ![Untitled](/images/lang_python/library/Datatable/Untitled%203.png)
        
    
    - Windows
        - Windows 10 or later
    
- **version 확인**
    
    ```python
    import datatable as dt
    print(dt.__version__)
    ```
    
    ![Untitled](/images/lang_python/library/Datatable/Untitled%204.png)
    

## Loading Data

---

- 기본적으로 **데이터 프레임**을 분석하기 위한 용도
    - `pandas` DataFrame과 비슷
- `Frame` 생성
    - `python` list 또는 dictionary, `numpy` array, `pandas` DataFrame 를 통해 생성 가능
    
    ```python
    dt1 = dt.Frame(A=[1, 2, 3, 4, 5], B=['a', 'b', 'c', 'd', 'e'], stypes={"A": dt.int64})
    dt2 = dt.Frame(pandas_dataframe)
    dt3 = dt.Frame(numpy_array)
    ```
    
    ![Untitled](/images/lang_python/library/Datatable/Untitled%205.png)
    

- CSV/text/Excel file **load** 또는 사전에 저장된 binary `.jay` 파일 **open**
    
    ```python
    dt4 = dt.fread("/data-path/data_file.csv")
    dt5 = dt.open("data.jay")
    ```
    
    - **fread**() : powerful and extremely fast
        - **자동으로** 로드할 데이터 파일에서 **parse parameter** 들을 **탐지**

## References :

---

- [Documentation](https://datatable.readthedocs.io/en/latest/index.html#)
- [https://www.kaggle.com/code/imvision12/loading-dataset-using-datatable](https://www.kaggle.com/code/imvision12/loading-dataset-using-datatable)