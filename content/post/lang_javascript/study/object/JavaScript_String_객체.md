---
title: "[JavaScript] String 객체"
description: ""
date: "2022-06-06T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

### 문자열 양쪽 끝의 공백 없애기 : trim()

- `"      Hello World!!!     "`      → `"Hello World!!!"`
- 사용자의 실수 또는 악의적인 목적으로 문자열 앞뒤에 공백이 추가되는 경우가 많으므로 **공백을 제거하기 위해 사용**
- 문자열 **앞뒤 공백(띄어쓰기, 줄바꿈 등)을 제거 가능**
    
    ![Untitled](/images/lang_javascript/study/JavaScript_String_객체/Untitled.png)
    

### 문자열을 특정 기호로 자르기 : split()

- 웹 페이지에서 데이터를 긁으면 다음과 같이 **쉼표(또는 다른 것)로 구분된 문자열**을 읽어서 분해해야 하는 경우가 있음
    
    ```jsx
    일자,달러,엔,유로
    02,1141.8,1097.46,1262.37
    03,1148.7,1111.36,1274.65
    04,1140.6,1107.81,1266.58
    ...생략...
    ```
    
- **split() 메소드**는 문자열을 **매개변수(다른 문자열)로 잘라서 배열을 만들어 리턴**하는 메소드
    
    ![Untitled](/images/lang_javascript/study/JavaScript_String_객체/Untitled%201.png)
    
- 추가로 **String 객체 정보는 다음 링크**를 참조
    
    [모질라 String 객체의 속성과 메소드](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/String)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판
- https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/String