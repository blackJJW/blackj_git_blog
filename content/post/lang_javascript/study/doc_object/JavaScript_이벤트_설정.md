---
title: "[JavaScript] 이벤트 설정"
description: ""
date: "2022-06-11T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"
  - "CSS"


---
<!--more-->

- `document.addEventlister('DOMContentLoaded', () => {}`  라는 형태의 코드
    - ‘document라는 문석 객체의 DOMContentLoaded 이벤트가 발생했을 때, 매개변수로 지정한 콜백 함수를 실행하라’는 의미

- 모든 문서 객체는 동작이나 작동에 따라 **이벤트**(**event**)라는 것이 발생
- 이벤트가 밯생할 때 실행할 함수는 **addEventListener() 메소드**
    - **이벤트 리스너**(**event listener**) 또는 **이벤트 핸들러**(**event handler**)라고 부름
    
    ```jsx
    문서_객체.addEventListener(이벤트_이름, 콜백_함수)
    ```
    
- addEventListener() 메소드를 사용해서 h1 태그를 클릭할 때 이벤트 리스너(콜백 함수)를 호출하는 예
    - 이벤트 리스너 내부에서 변수counter를 증가시키고 출력
    
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
                let counter = 0;
                const h1 = document.querySelector('h1');
    
                h1.addEventListener('click', (event) => {
                    counter++
                    h1.textContent = `click count : ${counter}`;
                });
            });
        </script>
        <style>
            h1 {
                /* 클릭을 여러 번 했을 경우
                글자가 선택되는 것을 막기 위한 스타일*/
                user-select: none;
            };
        </style>
    </head>
    <body>
        <h1>click count : 0</h1>
    </body>
    </html>
    ```
    
    - CSS 사용
        - **user-select 속성**을 **none**으로 지정하면 해당 태그를 **마우스로 드래그하지 못함**
        - h1 태그를 여러 번 클릭할 때 글자가 선택되는 것을 막기 위함
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_이벤트_설정/Untitled.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_이벤트_설정/Untitled%201.png)
    
- 이벤트를 제거할 때는 **removeEventListener() 메소드**를 사용
    
    ```jsx
    문서_객체.removeEventListener(이벤트_이름, 이벤트_리스너)
    ```
    
    - 이벤트 리스너 부분에는 연결할 때 사용했던 이벤트 리스너를 넣음
    - 변수 또는 상수로 **이벤트 리스너를 미리 만들고**, 이를 **연결과 제거에 활용**

- 버튼으로 **이벤트 연결 상태를 제어**
    - 이벤트 리스너가 **여러 번 연결되지 않게 isConnect라는 변수를 활용**
    
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <script>
            documnet.addEventListener('DOMContentLoaded', () => {
                let counter = 0;
                let isConnect = false;
    
                const h1 = document.querySelector('h1');
                const p = document.querySelector('p');
                const connectButton = document.querySelector('#connect');
                const disconnectButton = document.querySelector('#disconnect');
                
                const listener = (event) => {
                    h1.textContent = `click count : ${counter++}`;
                };
    
                connectButton.addEventListener('click', () => {
                    if (isConnect === false){
                        h1.addEventListener('click', listener);
                        p.textContent = 'Event Connection Status : connected';
                        isConnect = true;
                    }
                });
    
                disconnectButton.addEventListener('click', () => {
                    if (isConnect === true){
                        h1.removeEventListener('click', listener);
                        p.textContent = 'Event Connection Status : disconnected';
                        isConnect = false;
                    }
                });
            });
        </script>
        <style>
            h1{
                user-select: none;
            };
        </style>
    </head>
    <body>
        <h1>click count : 0</h1>
        <button id="connect">Event connect</button>
        <button id="disconnect">Event disconnect</button>
        <p>Event Connection status : disconnect</p>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_이벤트_설정/Untitled%202.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_이벤트_설정/Untitled%203.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_이벤트_설정/Untitled%204.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_이벤트_설정/Untitled%205.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판