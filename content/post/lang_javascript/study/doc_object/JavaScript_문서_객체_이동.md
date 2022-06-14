---
title: "[JavaScript] 문서 객체 이동"
description: ""
date: "2022-06-10T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- appendChild() 메소드는 문서 객체를 **이동할 때도 사용가능**
    - 문서 객체의 **부모(parent)는 반드시 하나**여야 함
    - 문서 객체를 **다른 문서 객체에 추가**하면 문서 객체의 이동이 이루어짐

- 1초마다 `<h1>`이동하는 h1 태그`</h1>`라는 요소
    - div#first(id 속성이 first인 div 태그)와 div#second로 움직이게 한다.
    
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
                // 문서 객체 읽어들이고 생성
                const divA = document.querySelector('#first'); // id속성 first
                const divB = document.querySelector('#second'); // id속성 second
                const h1 = document.createElement('h1'); // h1 태그 생성
                h1.textContent = '이동하는 h1 태그';
    
                // 서로 번갈아가면서 실행하는 함수 구현
                const toFirst = () => {
                    divA.appendChild(h1); // h1를 divA에 추가
                    setTimeout(toSecond, 1000); // 1초 뒤에 toSecond 실행
                };
                const toSecond = () => {
                    divB.appendChild(h1); // h1를 divB에 추가
                    setTimeout(toFirst, 1000); // 1초 뒤에 toFirst 실행
                };
                toFirst();
            });
        </script>
    </head>
    <body>
        <div id = "first">
            <h1>1) div 태그 내부</h1>
        </div>
        <hr>
        <div id = "second">
            <h1>2) div 태그 내부</h1>
        </div>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_문서_객체_이동/Untitled.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_문서_객체_이동/Untitled%201.png)
    
    - 코드 실행 시 h1 태그가 hr 태그를 기준으로 위와 아래로 이동하는 것을 확인
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판