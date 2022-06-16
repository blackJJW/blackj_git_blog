---
title: "[JavaScript] 글자 입력 양식 이벤트"
description: ""
date: "2022-06-13T16:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- **입력 양식**(**form**) : 사용자로부터 어떠한 입력을 받을 때 사용하는 요소
    - HTML에서는 **input 태그, textarea 태그, button 태그, select 태그** 등이 모두 입력 양식
    
- 입력 양식을 기반으로 inch를 cm 단위로 변환하는 프로그램
    
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
                const input = document.querySelector('input');
                const button = document.querySelector('button');
                const p = document.querySelector('p');
    
                button.addEventListener('click', () => {
                    // 입력을 숫자로 변환
                    const inch = Number(input.value);
                    // 숫자가 아니라면 바로 리턴
                    if (isNaN(inch)){
                        p.textContent = '숫자를 입력';
                        return;
                    };
    
                    // 변환해서 출력
                    const cm = inch * 2.54;
                    p.textContent = `${cm} cm`;
                });
            });
        </script>
    </head>
    <body>
        <input type="text"> inch<br>
        <button>계산</button>
        <p></p>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_글자_입력_양식_이벤트/Untitled.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_글자_입력_양식_이벤트/Untitled%201.png)
    
    - isNaN() 함수의 **결과가 true로 나오는 숫자가 아닌 경우 바로 return 키워드로 리턴해서 이후의 코드를 실행하지 않음**.
    - 다음과 같이 else 키워드를 사용할 수 있지만, 위 처럼 사용하면 들여쓰기 단체를 하나 줄일 수 있으므로 코드가 깔끔해짐.
        - 위 처럼 사용하는 패턴은 **조기 리턴(early return)** 이라는 **자주 사용되는 패턴**
        
        ```jsx
        if (isNaN(inch)){
        	p.textContent = '숫자를 입력';
        } else {
        	// 변환해서 출력
        	const cm = inch * 2.54;
          p.textContent = `${cm} cm`;
        }
        ```
        

- 이메일 형식 확인
    
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
                const input = document.querySelector('input');
                const p = document.querySelector('p');
                const isEmail = (value) => {
                    // @를 갖고 있고 && @ 뒤에 점이 있다면
                    return (value.indexOf('@') > 1) 
                        && (value.split('@')[1].indexOf('.') > 1)
                };
    
                input.addEventListener('keyup', (event) => {
                    const value = event.currentTarget.value;
                    if (isEmail(value)) {
                        p.style.color = 'green';
                        p.textContent = `이메일 형식 O : ${value}`;
                    } else {
                        p.style.color = 'red';
                        p.textContent = `이메일 형식 X : ${value}`;
                    }
                });
            });
        </script>
    </head>
    <body>
        <input type="text">
        <p></p>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_글자_입력_양식_이벤트/Untitled%202.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_글자_입력_양식_이벤트/Untitled%203.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판