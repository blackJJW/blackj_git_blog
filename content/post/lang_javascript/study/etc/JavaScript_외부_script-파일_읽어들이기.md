---
title: "[JavaScript] 외부 script 파일 읽어들이기"
description: ""
date: "2022-06-07T13:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- HTML 페이지 내부에 script 태그를 만들고 자바스크립트 코드를 입력했을 경우, 프로그램의 규모가 커지면 파일 하나가 너무 방대해지기 때문에 **파일을 분리하는 것이 좋다.**
- main.html 파일과 test.js **파일을 따로 생성**한다.
    
    ![Untitled](/images/lang_javascript/study/JavaScript_외부_script_파일_읽어들이기/Untitled.png)
    

- 외부 자바스크립트 파일을 읽을 때에도 **script 태그를 사용**
    - **src 속성**에 읽어들일 **파일 경로를 입력**
    
    ```html
    <!DOCTYPE html>
    <!-- main.html -->
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <script src="test.js"></script>
        <script>
            console.log('# main.html의 script 태그')
            console.log(`sample : ${sample}`)
        </script>
    </head>
    <body>
        
    </body>
    </html>
    ```
    
    ```jsx
    // test.js
    console.log('# test.js 파일')
    const sample = 10
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_외부_script_파일_읽어들이기/Untitled%201.png)
    
    - HTML 파일은 **위에서 아래로 태그를 읽어들이면서 적절한 처리**를 한다.
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판