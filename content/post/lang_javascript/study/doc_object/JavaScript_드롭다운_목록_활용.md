---
title: "[JavaScript] 드롭다운 목록 활용"
description: ""
date: "2022-06-13T16:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- 드롭다운 목록은 기본적으로 **select 태그로 구현**
- 기본 select 태그
    
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
                const select = document.querySelector('select');
                const p = document.querySelector('p');
    
                select.addEventListener('change', (event) => {
                    const options = event.currentTarget.options;
                    const index = event.currentTarget.selectedIndex;
    
                    p.textContent = `selection : ${options[index].textContent}`;
                });
            });
        </script>
    </head>
    <body>
        <select>
            <option>Python</option>
            <option>JAVA</option>
            <option>C++</option>
            <option>JavaScript</option>
            <option>HTML</option>
        </select>
        <p>선택 : Python</p>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_드롭다운_목록_활용/Untitled.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_드롭다운_목록_활용/Untitled%201.png)
    

- select 태그에 multiple 속성을 부여하면 `Ctrl` 키 또는 `Shift` 키를 누르고 여러 항목을 선택할 수 있는 선택 상자 출력
- **multiple select 태그**
    
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
                const select = document.querySelector('select');
                const p = document.querySelector('p');
    
                select.addEventListener('change', (event) => {
                    const options = event.currentTarget.options;
                    const list = [];
                    for (const option of options){
                        // option 속성에는 forEach() 메소드가 없음
                        // 따라서 반복문으로 돌림
                        if(option.selected){ // selected 속성을 확인
                            list.push(option.textContent)
                        };
                    };
                    p.textContent = `선택 : ${list.join(',')}`;
                }); 
            });
        </script>
    </head>
    <body>
        <select multiple>
            <option>Python</option>
            <option>JAVA</option>
            <option>C++</option>
            <option>JavaScript</option>
            <option>HTML</option>
        </select>
        <p></p>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_드롭다운_목록_활용/Untitled%202.png)
    
    - options 속성으로 모든 속성을 선택하고 반복문을 돌린 뒤, selected 속성으로 선택된 요소를 확인
    - 코드 실행 뒤, `Ctrl` 키를 누르고 여러 개의 요소를 선택하면, 여러 항목들의 결과가 출력됨
    
- cm 단위를 여러 단위로 변환
    
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
                let current;
                let transVar = 10;
    
                const select = document.querySelector('select');
                const input = document.querySelector('input');
                const span = document.querySelector('span');
    
                const calculate = () => {
                    span.textContent = (current * transVar).toFixed(2);
                                                    // 소수점 2번째자리까지 출력
                };
    
                select.addEventListener('change', (event) => {
                    const options = event.currentTarget.options;
                    const index = event.currentTarget.options.selectedIndex;
                    transVar = Number(options[index].value);
                    calculate()
                });
    
                input.addEventListener('keyup', (event) => {
                    // 값을 입력하면 현재 값을 추출
                    current = Number(event.currentTarget.value)
                    calculate();
                });
            });
        </script>
    </head>
    <body>
        <input type="text"> cm =
        <span></span>
        <select>
            <option value="10">mm</option>
            <option value="0.01">m</option>
            <option value="0.393701">inch</option>
        </select>
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_드롭다운_목록_활용/Untitled%203.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_드롭다운_목록_활용/Untitled%204.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판