---
title: "[JavaScript] 속성 조작"
description: ""
date: "2022-06-10T14:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- **문서 객체의 속성을 조작**할 때는 다음과 같은 메소드를 사용
    
    
    | 메소드 이름 | 설명 |
    | --- | --- |
    | 문서_객체.setAttribute(속성_이름, 값) | 특성 속성에 값을 지정 |
    | 문서_객체.getAttribute(속성_이름) | 특성 속성을 추출 |

- img 태그의 **src 속성을 조작해서 이미지를 출력**하는 예
    
    ```html
    http://placekitten.com/너비/높이
    ```
    
- 속성 조작
    
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
                const rects = document.querySelectorAll('.rect');
    
                rects.forEach((rect, index) => {
                    const width = (index + 1) * 100 ;
                    const src = `http://placekitten.com/${width}/250`;
                    
                    rect.setAttribute('src', src);
    						 // rect.src = src; 도 가능
                });
            });
        </script>
    </head>
    <body>
        <img class="rect">
        <img class="rect">
        <img class="rect">
        <img class="rect">
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_속성_조작/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판