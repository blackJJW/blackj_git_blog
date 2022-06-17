---
title: "[JavaScript] 체크 박스 활용"
description: ""
date: "2022-06-14T16:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- 체크 박스처럼 체크 상태를 확인 할 때는 **checked 속성**을 사용

- 체크 상태일 때만 타이머를 증가시키는 프로그램
    - **change 이벤트**가 발생했을 때 체크 박스의 체크 상태를 확인하고 **setInterval**() 함수 또는 **clearInterval**() 함수를 실행
        
        ```html
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Document</title>
        </head>
        <script>
            document.addEventListener('DOMContentLoaded', () => {
                let [timer, timerId] = [0, 0];
                const h1 = document.querySelector('h1');
                const checkbox = document.querySelector('input');
        
                checkbox.addEventListener('change', (evnet) => {
                    // checked 속성을 사용
                    if(event.currentTarget.checked){
                        //체크 상태
                        timerId = setInterval(() => {
                            timer += 1;
                            h1.textContent = `${timer} 초`;
                        }, 1000);
                    } else {
                        // 체크 해제 상태
                        clearInterval(timerId);
                    };
                });
            });
        </script>
        <body>
            <input type="checkbox">
            <span>타이머 활성화</span>
            <h1></h1>
        </body>
        </html>
        ```
        
        ![Untitled](/images/lang_javascript/study_1/JavaScript_체크_박스_활용/Untitled.png)
        
        ![Untitled](/images/lang_javascript/study_1/JavaScript_체크_박스_활용/Untitled%201.png)
        

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판