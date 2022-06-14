---
title: "[JavaScript] 문서 객체 생성"
description: ""
date: "2022-06-10T14:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- 문서 객체를 생성하고 싶을 때는 **document.createElement() 메소드 사용**
    
    ```jsx
    document.createElement(문서_객체_이름)
    ```
    

- 문서 객체를 생성했다고 문서 객체가 배치되는 것은 아니다.
    - 문서를 어떤 문서 아래에 추가할지를 지정해 주어야 함
    - 어떤 문서 객체가 있을때 위에 있는 것을 **부모**(**parent**), 아래에 있는 것을 **자식**(**child**)

- 문서 객체에는 appendChild() 메소드가 존재
    - 어떤 부모 객체 아래에 자식 객체를 추가 가능
    
    ```jsx
    부모_객체.appendChild(자식_객체)
    ```
    

- document.createElement() 메소드로 h1 태그를 생성, 이를 document.body 태그 아래에 추가
    
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
                // 문서 객체 생성
                // h1 태그 생성
                const header = document.createElement('h1');
    
                // 생성한 태그 조작
                header.textContent = '문서 객체 동적으로 생성';
                header.setAttribute('data-custom', '사용자 정의 속성');
                header.style.color = 'white';
                header.style.backgroundColor = 'black';
    
                // h1 태그를 body 태그 아래에 추가
                document.body.appendChild(header)
            });
        </script>
    </head>
    <body>
        
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_문서_객체_생성/Untitled.png)
    
    - 코드 실행 시 h1 태그 출력

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판