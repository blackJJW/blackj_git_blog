---
title: "[JavaScript] 타이머로 구현한 남은 글자 수 세기"
description: ""
date: "2022-06-16T14:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- 아시아권의 문자는 **키보드 이벤트**(**keydown, keypress, keyup 이벤트**)로 원하는 것을 제대로 구현할 수 없는 경우가 많음

- 트위터는 타이머를 사용해서 50밀리초마다 입력 양식 내부의 글자를 확인해서 글자 수를 센다.
    - **focus 이벤트**와 **blur 이벤트**를 사용
    - 입력 양식에 초점을 맞춘 경우(**활성화 상태**)와 초점을 해제한 경우(**비활성화 상태**)에 발생하는 이벤트
    - 입력 양식에 글자를 **입력하려고 선택한 순간부터** **타이머를 돌리고**, 다른 일을 하기 위해서 입력 양식에서 **초점을 해제**하면 **타이머를 정지**하게 만든다.
    
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
                let timerId
    
                textarea.addEventListener('focus', (event) => { // 입력 양식 활성화
                    timerId = setInterval(() => {
                        const length = textarea.value.length;
                        h1.textContent = `글자 수 : ${length}`;
                    }, 50);
                });
                textarea.addEventListener('blur', (event) => { // 입력 양식 비활성화
                    clearInterval(timerId);
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
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_타이머로_구현한_남은_글자_수_세기/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판