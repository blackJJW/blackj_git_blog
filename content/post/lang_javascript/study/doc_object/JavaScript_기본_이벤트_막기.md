---
title: "[JavaScript] 기본 이벤트 막기"
description: ""
date: "2022-06-15T13:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- 웹 브라우저는 이미지에서 마우스 오른쪽 버튼을 클릭하면 다음과 같은 **컨텍스트 메뉴**(**context menu**)를 출력
- 이처럼 어떤 이벤트가 발생했을 때 **웹 브라우저가 기본적으로 처리해주는 것**을 **기본 이벤트**라고 한다.
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_기본_이벤트_막기/Untitled.png)
    

- 링크를 클릭했을 때 이동하는 것, 제출 버튼을 눌렀을 때 이동하는 것 등이 모두 기본 이벤트의 예
- 이러한 **기본 이벤트를 제거**할 때는 **event 객체**의 **preventDefault() 메소드** 사용

- 모든 img 태그의 contextmenu 이벤트가 발생했을 때 preventDefault() 메소드를 호출해서 기본 이벤트를 막는 예
- 이미지 마우스 오른쪽 클릭 막기
    
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
                const imgs = document.querySelectorAll('img');
    
                imgs.forEach((img) => {
                    img.addEventListener('contextmenu', (event) => {
                        event.preventDefault(); 
    										// 컨텍스트 메뉴를 출력하는 기본 이벤트를 제거
                    });
                });
            });
        </script>
    </head>
    <body>
        <img src="http://placekitten.com/300/300" alt="">
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_기본_이벤트_막기/Untitled%201.png)
    
    - 인터넷에서 이미지 **불펌 방지등을 구현할 때 사용하는 코드**
    
- 체크 때만 링크 활성화
    
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
                let status = false;
    
                const checkbox = document.querySelector('input');
                checkbox.addEventListener('change', (event) => {
                    status = event.currentTarget.checked // checked 속성 사용
                });
    
                const link = document.querySelector('a');
                link.addEventListener('click', (event) => {
                    if (!status) {
                        event.preventDefault();
                    };
                });
            });
        </script>
    </head>
    <body>
        <input type="checkbox">
        <span>링크 활성화</span>
        <br>
        <a href="https://blackjjw.github.io/">블로그 링크</a>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_기본_이벤트_막기/Untitled%202.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_기본_이벤트_막기/Untitled%203.png)
    
    - **checked 속성**은 **불 자료형**
        - 체크 상태에 따라 **true** 또는 **false** 반환

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판