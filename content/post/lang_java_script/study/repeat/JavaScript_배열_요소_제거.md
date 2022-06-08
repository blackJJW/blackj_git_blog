---
title: "[JavaScript] 배열 요소 제거"
description: ""
date: "2022-05-20T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 배열의 요소 제거 방법
    - **인덱스를 기반**으로 제거하는 경우
    - **값을 기반**으로 제거하는 경우

### 인덱스로 요소 제거

- 배열의 특정 인덱스에 있는 요소를 제거할 때는 **splice() 메소드**를 사용
    
    ```jsx
    배열.splice(인덱스, 제거할 요소의 개수)
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_배열_요소_제거/Untitled.png)
    

### 값으로 요소 제거

- 값을 기반으로 요소를 제거할 때는 배열 내부에서 특정 값의 위치를 찾는 **indexOf() 메소드**를 사용해 갑의 위치를 추출, **splice() 메소드를 사용해 제거**
    
    ```jsx
    const 인덱스 = 배열.indexOf(요소)
    배열.splice(인덱스, 1)
    ```
    
- **indexOf() 메소드 :** 배열 내부에 **요소가 있을 경우 인덱스를 리턴**
    - 배열 내부에 **요소가 없을 경우 -1을 리턴**
    
    ![Untitled](/images/lang_javascript/study/JavaScript_배열_요소_제거/Untitled%201.png)
    

- **문자열의 indexOf() 메소드**
    - 문자열 내부에서 특정 문자열의 위치를 찾을 수 있음
    
    ![Untitled](/images/lang_javascript/study/JavaScript_배열_요소_제거/Untitled%202.png)
    

- **배열 내부에서 특정 값을 가진 요소 모두 제거**
    - 반복문 사용
    - **filter() 메소드** 사용
    
    ![Untitled](/images/lang_javascript/study/JavaScript_배열_요소_제거/Untitled%203.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판