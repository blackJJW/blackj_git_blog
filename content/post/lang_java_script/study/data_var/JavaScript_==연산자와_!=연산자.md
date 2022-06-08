---
title: "[JavaScript] ==연산자와 !=연산자"
description: ""
date: "2022-05-07T21:10:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 대부분의 프로그래밍 언어는 자료가 같은지 다른지를 비교할 때 **== 연산자**와 **!= 연산자**를 사용
    - ‘**값이 같은지**’를 비교하는 연산자
- **=== 연산자**와 **!== 연산자**
    - **‘값과 자료형이 같은지**’를 비교하는 연산자
    - 어떻게 해서라도 값을 같게 만들고 비교해주면서 일반적인 생각과 다른 결과를 낸다.

  ```jsx
  > 1 == "1"  -> 자료형이 달라도 어떻게든 변환을 하고 나면 값이 같아지므로 true
  
  > false == "0"  -> false가 0 이므로, "0"이 0으로 변환된 뒤에 비교
  
  > "" == []  -> 빈 문자열은 false, 비어있는 배열 []는 false로 변환된 뒤에 비교
  
  > 0 == []  -> 0은 false, 비어있는 배열 []는 false로 변환된 뒤에 비교
  ```

  ![Untitled](/images/lang_javascript/study/JavaScript_==연산자와_!=연산자/Untitled.png)

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판