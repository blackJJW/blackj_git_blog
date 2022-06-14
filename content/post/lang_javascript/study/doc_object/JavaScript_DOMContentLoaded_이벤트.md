---
title: "[JavaScript] DOMContentLoaded 이벤트"
description: ""
date: "2022-06-09T13:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- 문서 객체를 조작할 때는 **DOMContentLoaded** 이벤트를 사용
- 코드를 입력할 때 DOMContentLoaded 문자열은 **오탈자를 입력해도 오류를 발생하지 않는다**.
    
    ```jsx
    document.addEventListener('DOMContentLoaded', () => {
    	// 문장
    }
    ```
    

- HTML 페이지는 코드를 위에서 아래로 차례대로 실행
    
    ```html
    <!DOCTYPE html>
    <html>
    <head>
        <title>Document</title>
    </head>
    <body>
        
    </body>
    </html>
    ```
    
    - 기본적인 실행 순서
        - `<!DOCTYPE html>` → HTML5 인식
        - `<html>` → `<head>` → `<title>` → `<body>`
    
- HTML 코드를 자바스크립트로 조작
    - document의 body 안에 있는 HTML 코드(innerHTML)를 자바스크립드로 조작할 수 있게 해주는 코드
    
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DOMContentLoaded</title>
        <script>
            // HTML 태그를 쉽게 만들 수 있는 콜백 함수를 선언
            const h1 = (text) => `<h1>${text}</h1>`
        </script>
        <script>
            // body 태그가 생성되기 이전에 script 태그로 body 태그를 조작
            document.body.innerHTML += h1('1번째 script 태그') 
                                        // 앞에서 선언한 h1 함수를 실행
        </script>
    </head>
    <body> 
        <!-- body 태그는 head 태그 다음에 생성 -->
        <script>
            document.body.innerHTML += h1('2번째 script 태그')
        </script>
        <h1>1번째 h1 태그</h1>
        <script>
            document.body.innerHTML += h1('3번째 script 태그')
        </script>
        <h1>2번째 h2 태그</h1>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_DOMContentLoaded_이벤트/Untitled.png)
    
    - body 태그가 생성되기 이전에 head 태그 안의 script 태그에서 body 태그를 조작하던 부분(<h1>${text}</h1>를 출력하는 부분)은 화면에 출력되지 않음
    - 기본적으로 head 태그 내부에 script 태그를 배치하면 **body 태그에 있는 문서 객체(요소)에 접근 불가**
    
    - head 태그 내부의 script 태그에서 body 태그에 있는 문서에 접근하려면 **화면에 문서 객체(요소)를 모두 읽어들일 때까지 기다려야함**

- **DOMContentLoaded 이벤트**는 웹 브라우저가 문서 객체를 모두 일고 나서 실행하는 이벤트
- DOMContentLoaded 상태가 되었을 때 콜백 함수를 호출
    
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>DOMContentLoaded</title>
        <script>
            // DOMContentLoaded 이벤트를 연결
            document.addEventListener('DOMContentLoaded', () => {
                const h1 = (text) => `<h1>${text}</h1>`
                document.body.innerHTML += h1('DOMContentLoaded 이벤트 발생')
            })
        </script>
    </head>
    <body>
        
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_DOMContentLoaded_이벤트/Untitled%201.png)
    
    - 코드를 실행하면 script 태그가 body 태그 이전에 위치해도 문제없이 코드가 실행
    
- **addEventListener() 메소드**
    - **document.addEventListener(”DOMContentLoaded”, () => {})** 대부분의 코드에서 사용
    - document라는 문서 객체의 DOMContentLoaded **이벤트가 발생했을 때, 매개변수로 지정한 콜백 함수를 실행**하라는 의미

- l**oad 이벤트**
    - DOMContentLoaded는 HTML5부터 추가된 이벤트
    - 구 버전의 웹 브라우저를 대상으로 만들어진 코드를 분석한다면 Load 이벤트를 볼 수 있다.
    
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <script>
            document.onload = function () {
                document.innerHTML += 'load 이벤트 발생'
            }
    
            document.addEventListener('load', function () {
                document.innerHTML += 'load 이벤트 발생'
            })
        </script>
    </head>
    <body>
        
    </body>
    </html>
    ```
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판