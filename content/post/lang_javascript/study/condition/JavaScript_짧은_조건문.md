---
title: "[JavaScript] 짧은 조건문"
description: ""
date: "2022-05-16T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 논리 연산자의 특성을 조건문으로 사용

### 논리합 연산자를 사용한 짧은 조건문

- 논리합 연산자를 사용한 표현식은 **뒤에 어떠한 값이 들어가도 항상 참**
    
    ```jsx
    true || ○○○
    ```
    
- 참(true)이 확실할 때 추가 연산을 진행하지 않음
    - 논리합 연산자의 **좌변이 참이면 우변을 실행하지 않음**
    
    ![Untitled](/images/lang_javascript/study/JavaScript_짧은_조건문/Untitled.png)
    
    ```jsx
    불 표현식 || 불 표현식이 거짓일 때 실행할 문장
    ```
    

### 논리곱 연산자를 사용한 짧은 조건문

- 논리곱 연산자는 **양변이 모두 참일 때만 참**
- 다음 표현식은 **항상 거짓**
    
    ```jsx
    false && ○○○
    ```
    
- **좌변이 거짓이면 우변을 실행하지 않음**
    
    ```jsx
    결과가 거짓인 불 표현식 && 불 표현식이 참일 때 실행할 문장
    ```
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판