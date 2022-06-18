---
title: "[JavaScript] 기본 예외 처리"
description: ""
date: "2022-06-16T17:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- **기본 예외 처리** : **조건문을 사용**해서 예외가 발생하지 않게 만드는 것

- 다음 코드는 querySelector() 메소드로 문서 객체를 추출한 뒤 textContent 속성에 글자를 할당하는 프로그램
- 하지만 body 태그 내부에 h1 태그가 없음
    - 따라서 예외가 발생
    
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
                const h1 = document.querySelector('h1');
                h1.textContent = "Hello!!!";
            });
        </script>
    </head>
    <body>
        
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_기본_예외_처리/Untitled.png)
    

- 문서 객체를 선택했는 데 문서 객체가 없는 경우라면, **조건문**으로 h1이 존재하는 경우에만 textContent 속성을 변경하도록 **예외 처리**가 가능
- 기본 예외 처리
    
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
                const h1 = document.querySelector('h1');
                if (h1) { // h1이 존재하면 true, 없으면 false
                    h1.textContent = "Hello!!!";
                } else{
                    console.log("h1 태그 없음");
                }
            });
        </script>
    </head>
    <body>
        
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_기본_예외_처리/Untitled%201.png)
    

- 자바스크립트는 다른 프로그래밍 언어와 비교해서 **매우 유연**하기 때문에 **예외를 발생할 가능성이 적은 편**
- 예를 들면, 대부분의 프로그래밍 언어는 배열의 길이를 넘는 위치를 선택할 경우 오류를 발생
    - 자바스크립트는 undefined 를 출력
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_기본_예외_처리/Untitled%202.png)
    
    - 유연함 때문에 **예외를 발생하지 않는다고 좋은 것은 아니다.**
        - 프로그램에 문제가 발생했음에도 멈추지 않고 실행되면 계**속해서 문제가 발생할 가능성이 있음**
        - 따라서 **문제가 발생할 가능성이 부분은 조건문으로 처리**해주어야 함
        

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판