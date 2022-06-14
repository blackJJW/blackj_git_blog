---
title: "[JavaScript] 스타일 조작"
description: ""
date: "2022-06-10T14:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"
  - "CSS"


---
<!--more-->

- 문서 객체의 스타일을 조작할 때는 **style 속성을 사용**
- style 속성은 객체
    - 내부에는 속성으로 CSS를 사용해 지정할 수 있는 스타일이 존재
    - 이러한 속성에는 CSS로 입력할 때 사용하는 값과 같은 값을 입력
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_스타일_조작/Untitled.png)
    
    - 그림 속 속성들의 이름이 **CSS에서 사용할 때와 약간 다름**
    - 자바스크립트에서는 - 기호를 식별자에 사용할 수 없으므로, **두 단어 이상의 속성은 캐멀 케이스**로 나타냄
        
        
        | CSS 속성 이름 | 자바스크립트 style 속성 이름 |
        | --- | --- |
        | background-color | backgroundColor |
        | text-align | textAlign |
        | font-size | fontSize |
    - **style 객체**는 다음과 같은 방법으로도 스타일 조정 가능
        
        ```jsx
        h1.style.backgroundColor // 가장 많이 쓰는 형태
        h1.style['backgroundColor']
        h1.style['background-color']
        ```
        

- 25개의 div 태그를 조작하여 검은색에서 흰색으로 변화하는 그레이디언트를 생성
    
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
                const divs = document.querySelectorAll('body > div');
                                            // body 태그 아래에 있는 div 태그를 선택
                divs.forEach((div, index) => {
                    console.log(div, index);
                    const val = index * 10; // index는 0 부터 24까지 반복
                    div.style.height= `10px` // 크기 지정시 반드시 단위 붙어야함
                    div.style.backgroundColor = `rgba(${val}, ${val}, ${val})`
                });
            });
        </script>
    </head>
    <body>
        <!-- div 태그 25개-->>
        <div></div><div></div><div></div><div></div><div></div>
        <div></div><div></div><div></div><div></div><div></div>
        <div></div><div></div><div></div><div></div><div></div>
        <div></div><div></div><div></div><div></div><div></div>
        <div></div><div></div><div></div><div></div><div></div>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_스타일_조작/Untitled%201.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판