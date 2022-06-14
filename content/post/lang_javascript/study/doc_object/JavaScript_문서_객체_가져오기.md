---
title: "[JavaScript] 문서 객체 가져오기"
description: ""
date: "2022-06-10T13:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- **document.body** 코드를 사용하면 문서의 body 요소를 읽어들일 수 있음.
- HTML 문서에 있는 head 요소와 title 요소 등은 다음과 같은 방법으로 읽어들일 수 있음
    
    ```jsx
    document.head
    document.body
    document.title
    ```
    
    - 웹 브라우저의 자바스크립트가 **“당연히 있다”라는 전제**를 하고 만든 속성

- head 요소과 body 요소 **내부에 만든 다른 요소들**은 다음과 같은 **별도의 메소드를 사용**하여 접근
    
    ```jsx
    document.querySelector(선택자)
    document.querySelectorAll(선택자)
    ```
    
    - 선택자 부분에는 CSS 선택자를 입력
    - **CSS 선택자**
    
        | 이름 |    선택자 형태    |           설명            |
        |:------------:|:-----------------------:| :--- |
        | 태그 선택자 |      태그      |    특정 태그를 가진 요소를 추출     |
        | 아이디 선택자 |     #아이디     |   특정 id 속성을 가진 요소를 추출   |
        | 클래스 선택자 |     .클래스     | 특정 class 속성을 가진 요소를 추출  |
        | 속성 선택자 |    [속성=값]    |  특정 속성 값을 갖고 있는 요소를 추출  |
        | 후손 선택자 | 선택자_A 선택자_B  | 선택자_A 아래에 있는 선택자_B를 선택  |

- **querySelector() 메소드**는 요소를 하나만 추출
- **querySelectorAll() 메소드**는 문서 객체를 여러개 추출

- **querySelector() 메소드**, h1 태그를 추출하고 조작
    
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <script>
            document.addEventListener('DOMContentLoaded', () => {
                // 요소를 읽어들임
                const header = document.querySelector('h1');
    
                // 텍스트와 스타일을 변경
                header.textContent = 'HEADERS';
                header.style.color = 'white';
                header.style.backgroundColor = 'black';
                header.style.padding = '10px';
            });
        </script>
    </head>
    <body>
        <h1></h1>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_문서_객체_가져오기/Untitled.png)
    
    - 실제 h1 태그에는 내용을 입력하지 않았지만, **자바스크립트 쪽에서 문서 객체를 조작**했으므로 위와 같이 출력
    
- **querySelectorAll() 메소드**
    - 문서 객체 여러 개를 배열로 읽어들이는 함수
    - 내부의 요소에 **접근하고 활용하려면 반복을 돌려야함**
    - 일반적으로 **forEach() 메소드를 사용**해서 반복
    
     
    
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <script>
            document.addEventListener('DOMContentLoaded', () => {
                // 요소를 읽어 들임
                const headers = document.querySelectorAll('h1');
    
                // 텍스트와 스타일을 변경
                headers.forEach((header) => {
                    header.textContent = 'HEADERS';
                    header.style.color = 'white';
                    header.style.backgroundColor = 'black';
                    header.padding = '10px';
                });
            });
        </script>
    </head>
    <body>
        <h1></h1>
        <h1></h1>
        <h1></h1>
        <h1></h1>
        <h1></h1>
    </body>
    </html>
    ```

    ![Untitled](/images/lang_javascript/study_1/JavaScript_문서_객체_가져오기/Untitled%201.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판