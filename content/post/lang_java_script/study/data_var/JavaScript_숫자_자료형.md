---
title: "[JavaScript] 숫자 자료형"
description: ""
date: "2022-05-07T19:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 자바스크립트는 **소수점이 있는 숫자**와 **없는 숫자**를 **모두 같은 자료형**으로 인식

  ![Untitled](/images/lang_javascript/study/JavaScript_숫자_자료형/Untitled.png)

### <u>숫자 연산자</u>

- **숫자 연산자**로 기본적인 **사칙연산**이 가능

  |  연산자  |    설명    |  연산자  |    설명    |
  |:-----:|:--------:|:-----:|:--------:|
  |   +   | 더하기 연산자  |   *   | 곱하기 연산자  |
  |   -   |  빼기 연산자  |   /   | 나누기 연산자  |

- 숫자 자료형을 연산할 때 **연산자 우선순위**를 고려

  ```jsx
  5 + 3 * 2
  // 곱셈의 우선 순위가 더 높으므로 곱셈 먼저 계산
  ```

- 먼저 계산하고 싶은 연산이 있으면 **괄호**를 사용한다.

  ```jsx
  (5 + 3) * 2
  ```
  
  ![Untitled](/images/lang_javascript/study/JavaScript_숫자_자료형/Untitled%201.png)

- **나머지 연산자**(**%** 연산자)
    - 좌변을 우변으로 나눈 나머지를 출력

  |  연산자  |    설명    |
  |:-----:|:--------:|
  |   %   | 나머지 연산자  |
  
  ![Untitled](/images/lang_javascript/study/JavaScript_숫자_자료형/Untitled%202.png)

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판