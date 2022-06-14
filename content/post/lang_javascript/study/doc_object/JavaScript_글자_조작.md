---
title: "[JavaScript] 글자 조작"
description: ""
date: "2022-06-10T13:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- 

| 속성 이름             | 설명                    |
|:------------------|:----------------------|
| 문서_객체.textContent | 입력된 문자열을 그대로 넣음       |
| 문서_객체.innerHTML   | 입력된 문자열을 HTML 형식으로 넣음 |
- **글자 조작**
    
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
                const a = document.querySelector('#a');
                const b = document.querySelector('#b');
    
                a.textContent = '<h1>textContent 속성</h1>';
                b.innerHTML = '<h1>innerHTML 속성</h1>';
            });
        </script>
    </head>
    <body>
        <div id = 'a'></div>
        <div id = 'b'></div>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_글자_조작/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판