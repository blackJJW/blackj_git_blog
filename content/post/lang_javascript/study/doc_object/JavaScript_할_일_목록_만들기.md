---
title: "[JavaScript] 할 일 목록 만들기"
description: ""
date: "2022-06-16T13:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- 할 일 목록 만들기
    
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
    </head>
    <body>
        <h1>To Do List</h1>
        <input id="todo">
        <button id="add-button">Add</button>
        <div id="todo-list"></div>
    </body>
    <script>
        document.addEventListener('DOMContentLoaded', () => {
            // 문서 객체 가져옴
            const input = document.querySelector('#todo');
            const todoList = document.querySelector('#todo-list');
            const addButton = document.querySelector('#add-button');
    
            // 변수 선언
            // 이후에 removeTodo() 함수에서 문서 객체를 쉽게 제거하기 위한 용도
            let keyCount = 0;
    
            // 함수 선언
            const addTodo = () => {
                // 입력 양식에 내용이 없으면 추가하지 않음
                if (input.value.trim() === ''){
                    alert('할 일을 입력하세요.');
                    return;
                };
    
                // 문서 객체를 설정
                const item = document.createElement('div');
                const checkbox = document.createElement('input');
                const text = document.createElement('span');
                const button = document.createElement('button');
    
                // 문서 객체를 식별할 키를 생성
                // 이후에 removeTodo() 문서 객체를 쉽게 제거하기 위한 용도
                const key = keyCount;
                keyCount += 1;
    
                // item 객체를 조작하고 추가
                /* 구성 형태
                <div data-key="숫자">
                    <input>
                    <span></span>
                    <button></button>
                </div>
                */
                item.setAttribute('data-key', key);
                item.appendChild(checkbox);
                item.appendChild(text);
                item.appendChild(button);
                todoList.appendChild(item);
    
                // checkbox 객체 조작
                // <input type="checkbox"> 형태를 구성
                checkbox.type = 'checkbox';
                checkbox.addEventListener('change', (event) => {
                    item.style.textDecoration = event.target.checked ? 'line-through' : '';
                });
    
                // text 객체를 조작
                // <span>글자</span> 형태를 구성 
                text.textContent = input.value;
    
                // button 객체를 조작
                // <button>delete</button> 형태를 구성
                button.textContent = 'delete';
                button.addEventListener('click', () => {
                    removeTodo(key);
                });
    
                // 입력 양식의 내용을 비움
                input.value = '';
            };
    
            const removeTodo = (key) => {
                // 식별 키로 문서 객체를 제거
                // 위에서 지정한 <div data-key="숫자">를 기반으로 요소를 지정
                const item = document.querySelector(`[data-key="${key}"]`);
                todoList.removeChild(item);
            };
    
            // 이벤트 연결
            addButton.addEventListener('click', addTodo);
            input.addEventListener('keyup', (event) => {
                // 입력 양식에서 Enter 키를 누르면 바로 addTodo() 함수를 호출
                const ENTER = 13;
                if(event.keyCode === ENTER){
                    addTodo();
                };
            });
        });
    </script>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_할_일_목록_만들기/Untitled.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_할_일_목록_만들기/Untitled%201.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판