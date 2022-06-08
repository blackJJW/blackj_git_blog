---
title: "[JavaScript] 조건부 연산자"
description: ""
date: "2022-05-15T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- **조건부 연산자** : 조건문과 비슷한 역할을 하는 연산자
    
    ```jsx
    불 표현식 ? 참일 경우 : 거짓일 경우
    ```
    
- 자바스크립트에서 **항을 3개를 갖는 연산자**는 조건부 연산자가 유일
    - **삼항 연산자**라고도 부름
- 조건부 연산자 사용
    
    ```jsx
    <script>
    	const input = prompt('숫자를 입력', '')
    	const number = Number(input)
    
    	const result = (number >= 0) ? '0 이상 숫자' : '0 보다 작은 숫자'
      alert(result)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_조건부_연산자/Untitled.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_조건부_연산자/Untitled%201.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_조건부_연산자/Untitled%202.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_조건부_연산자/Untitled%203.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_조건부_연산자/Untitled%204.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판