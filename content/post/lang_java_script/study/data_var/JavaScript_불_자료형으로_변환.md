---
title: "[JavaScript] 불 자료형으로 변환"
description: ""
date: "2022-05-09T17:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 다른 자료형을 불 자료형으로 변환할 때는 **Boolean() 함수**를 사용
    
    ```jsx
    Boolean(자료)
    ```
    
- 대부분의 자료는 불로 변환했을 때 **true**로 변환
- **0, NaN, ‘…’, “…”(빈 문자열), null, undefined** 는 **false**로 변환
    
    ![Untitled](/images/lang_javascript/study/JavaScript_불_자료형으로_변환/Untitled.png)
    

### 논리 부정 연산자를 사용해 자료형 변환

- Boolean() 함수를 사용하지 않고 **논리 부정 연산자( ! )**를 사용
- 불이 아닌 다른 자료에 논리 부정 연산자를 2번 사용하면 불 자료형으로 변환
    
    ![Untitled](/images/lang_javascript/study/JavaScript_불_자료형으로_변환/Untitled%201.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판