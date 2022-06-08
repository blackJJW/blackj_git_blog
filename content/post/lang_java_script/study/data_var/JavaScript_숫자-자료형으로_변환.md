---
title: "[JavaScript] 숫자 자료형으로 변환"
description: ""
date: "2022-05-09T16:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 다른 자료형을 숫자 자료형으로 변환할 때는 **Number() 함수** 사용
    
    ```jsx
    Number()
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_숫자_자료형으로_변환/Untitled.png)
    

- 다른 문자가 들어 있어 숫자로 변환할 수 없는 경우에는 **NaN(Not a Number)** 출력
    - **NaN** : 숫자로 나타낼 수 없는 숫자
    
    ![Untitled](/images/lang_javascript/study/JavaScript_숫자_자료형으로_변환/Untitled%201.png)
    

- 불을 숫자로 변환하면 true는 1로, false는 0으로 변환
    
    ![Untitled](/images/lang_javascript/study/JavaScript_숫자_자료형으로_변환/Untitled%202.png)
    

### <u>숫자 연산자를 사용해 자료형 변환</u>

- Number() 함수를 사용하지 않고도 다른 자료형을 숫자 자료형으로 변환 가능
- 숫자 연산자 **-**, *****,  **/** 를 사용
- 숫자가 아닌 다른 자료에서 **0을 빼거나**, **1을 곱하거나**, **1로 나누면** 숫자 자료형으로 변환
    
    ![Untitled](/images/lang_javascript/study/JavaScript_숫자_자료형으로_변환/Untitled%203.png)
    

- 불과 숫자를 **+ 연산자**로 연결하면 불이 숫자로 변환된 뒤에 더해짐
    
    ![Untitled](/images/lang_javascript/study/JavaScript_숫자_자료형으로_변환/Untitled%204.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판