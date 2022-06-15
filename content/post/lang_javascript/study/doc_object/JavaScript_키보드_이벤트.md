---
title: "[JavaScript] 키보드 이벤트"
description: ""
date: "2022-06-12T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- **키보드 이벤트**는 3가지 이벤트가 존재
    - 일반적으로 **keyup 이벤트를 사용**
    
    | 이벤트 | 설명 |
    | --- | --- |
    | keydown | 키가 눌릴 때 실행. 키보드를 꾹 누르고 있을 때, 입력될 때 또한 실행 |
    | keypress | 키가 입력되었을 때 실행.  웹브라우저에 따라 아시아 문화권의 문자(한국어, 중국어, 일본어)를 제대로 처리하지 못하는 문제 존재 |
    | keyup | 키보드에서 키가 떨어질 때 실행 |

- **textarea**에 **keyup 이벤트를 적용**해서 입력한 글자 수를 세는 프로그램
    - textarea처럼 텍스트를 입력하는 **입력 양식의 값은 value 속성으로 읽어들임**
    
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
                const textarea = document.querySelector('textarea');
                const h1 = document.querySelector('h1');
    
                textarea.addEventListener('keyup', (event) => {
                    const length = textarea.value.length;
                    h1.textContent = `string count : ${length}`;
                });
            });
        </script>
    </head>
    <body>
        <h1></h1>
        <textarea></textarea>
    </body>
    </html>
    ```
    
    - 코드 실행 시 입력 양식에 **값을 입력하면 글자 수를 출력**
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_키보드_이벤트/Untitled.png)
    
    - **keydown 이벤트**의 경우
        - 글자 수를 잘 세지 못하는 문제 발생
        
        ```jsx
        textarea.addEventListener('keydown', (event) => {
                        const length = textarea.value.length;
                        h1.textContent = `string count : ${length}`;
                    });
        ```
        
        ![Untitled](/images/lang_javascript/study_1/JavaScript_키보드_이벤트/Untitled%201.png)
        
    - **keypress 이벤트**의 경우
        - 아시아권의 문자는 공백(띄어쓰기, 줄바꿈 등)이 들어가지 전까지 글자 수를 세지 않음
        
        ```jsx
        textarea.addEventListener('keypress', (event) => {
                        const length = textarea.value.length;
                        h1.textContent = `string count : ${length}`;
                    });
        ```
        
        ![Untitled](/images/lang_javascript/study_1/JavaScript_키보드_이벤트/Untitled%202.png)
        
    - **keyup 이벤트**의 경우
        - 키가 키보드에서 떨어질 때 발생하므로 **특정 키를 꾹 누르고 있으면 글자 수를 세지 않음**
        
        ```jsx
        textarea.addEventListener('keyup', (event) => {
                        const length = textarea.value.length;
                        h1.textContent = `string count : ${length}`;
                    });
        ```
        
        ![Untitled](/images/lang_javascript/study_1/JavaScript_키보드_이벤트/Untitled%203.png)
        

### 키보드 키 코드 사용

- 키보드 이벤트가 발생할 때는 **이벤트 객체로 어떤 키를 눌렀는지와 관련된 속성들이 따라옴**
    
    
    | 이벤트 속성 이름 | 설명 |
    | --- | --- |
    | code | 입력한 키 |
    | keyCode | 입력한 키를 나타내는 숫자 |
    | altKey | Alt 키를 눌렀는지 |
    | ctrlKey | Ctrl 키를 눌렸는지 |
    | shiftKey | Shift 키를 눌렸는지 |
- **code** 속성 : 입력한 기를 나타내는 문자열이 들어있음
- **altKey, ctrlKey, shiftKey** 속성 : 해당 키를 눌렀는지 불 자료형 값이 들어있음

- keydown 이벤트와 keyup 이벤트
    
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
                const print = (event) => {
                    let output = '';
                    output += `alt : ${event.altKey}<br>`;
                    output += `ctrl : ${event.ctrlKey}<br>`;
                    output += `shift : ${event.shiftKey}<br>`;
                    output += `code : ${typeof(event.code) !== 'undefined' ?
                     event.code : event.keyCode}<br>`;
                    h1.innerHTML = output;
                };
    
                document.addEventListener('keydown', print);
                document.addEventListener('keyup', print);
            });
        </script>
    </head>
    <body>
        <h1></h1>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_키보드_이벤트/Untitled%204.png)
    
    - **단축키**를 구현할 때 **키보드 이벤트 속성을 사용**한다.
    - ‘**event.code가 있는 경우**’를 확인하는 코드를 사용
        - 인터넷 익스플로러와 구버전 엣지 브라우저를 지원하기 때문
        - 인터넷 익스플로러와 구버전 엣지 브라우저는  **code 속성을 지원하지 않음**
        - 이런 웹 브라우저까지 지원하려면 **keyCode 속성을 활용해서 프로그램을 구현**해야 함
            - [code 속성 값 링크](https://developer.mozilla.org/en-US/docs/Web/API/UI_Events/Keyboard_event_code_values)
            - [keyCode 속성 값 링크](https://developer.mozilla.org/en-US/docs/Web/API/KeyboardEvent/keyCode)

- **keyCode 속성**은 입력한 키를 숫자로 나타낸다.
- **37, 38, 39, 40이 방향키 왼쪽, 위, 오른쪽, 아래**를 나타낸다.
- 키 별로 움직이기
    
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
                // 별의 초기 설정
                const star = document.querySelector('h1');
                star.style.position = 'absolute';
                                        // style 속성을 조작하여 position 값을 설정
                // 별의 이동을 출력
                let [x, y] = [0, 0];
                const block = 20;
                const print = () => {
                    star.style.left = `${x * block}px`;
                    star.style.top = `${y * block}px`;
                };
                print();
    
                // 별을 이동
                const [left, up, right, down] = [37, 38, 39, 40];
                document.body.addEventListener('keydown', (event) => {
                                                // 키보드가 눌릴 때 실행
                    switch (event.keyCode){
                        case left:
                            x -= 1;
                            break;
                        case up:
                            y -= 1;
                            break;
                        case right:
                            x += 1;
                            break;
                        case down:
                            y += 1;
                            break;
                    };
                    print();
                });
            });
        </script>
    </head>
    <body>
        <h1>★</h1>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_키보드_이벤트/Untitled%205.png)
    
    - 방향키를 사용하는 게임 등을 할 때는 방향키를 꾹 누르고 있을 가능성이 많으므로 **keydown 이벤트를 활용**

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판