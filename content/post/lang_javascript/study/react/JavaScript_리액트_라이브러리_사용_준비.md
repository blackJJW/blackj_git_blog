---
title: "[JavaScript][Library][React] 리액트 라이브러리 사용 준비"
description: ""
date: "2022-06-24T14:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "JavaScript Library"
  - "HTML"
  - "React"


---
<!--more-->

- **리액트 라이브러리**(**React Library**)
    - 규모가 큰 자바스크립트 라이브러리
    - **사용자 인터페이스(UI)를 쉽게 구성할 수 있도록 도와줌**
    - **대규모 프론트엔드 웹 애플리케이션을 체계적으로 개발** 가능
    - 리액트 네이티브를 활용해서 스마트폰에서도 빠른 속도로 작동하는 애플리케이션을 만들 수 있다.
    - [리액트 라이브러리 링크](https://ko.reactjs.org/)
        
        ![Untitled](/images/lang_javascript/study_3/JavaScript_리액트_라이브러리_사용_준비/Untitled.png)
        

## 리액트 라이브러리 사용 준비

---

- 리액트 라이브러리를 사용하는 가장 기본적인 방법은 HTML 파일에서 다음과 같은 **3개의 자바스크립트를 읽어들이는 것**
    - 리액트를 **사용하기 위해 필요**
        - **react.development.js**
            - https://unpkg.com/react@17/umd/react.development.js
        - **react-dom.development.js**
            - https://unpkg.com/react-dom@17/umd/react-dom.development.js
    - 리액트 **코드를 쉽게 작성**할 수 있게 해주는 라이브러리
        - **babel.min.js**
            - https://unpkg.com/babel-standalone@6/babel.min.js
- 리액트 기본 사용 준비
    
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <!-- 리액트 사용 준비 -->
        <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
        <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
        <script src="https://unpkg.com/babel-standalone@6/babel.min.js"></script>
    </head>
    <body>
        <div id="root"></div> <!-- div#root 태그 생성-->
        
        <!-- 리액트를 사용하는 코드 입력 -->
        <script type="text/babel">  // type 속성에 "text/babel"을 지정
    
        </script>
        
    </body>
    </html>
    ```
    
- 리액트 라이브러리는 단순한 자바스크립트가 아니라 리액트를 위해서 개발된 자바스크립드 확장 문법을 사용
    - 이러한 문법을 사용하려면 **바벨**(**babel**)이라는 라이브러리를 **추가로 읽어들이고 바벨을 적용할 부분을 지정**
        
        ```html
        <script type="text/babel"></script>
        ```
        

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판