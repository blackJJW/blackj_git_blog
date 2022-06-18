---
title: "[JavaScript] localStorage 객체"
description: ""
date: "2022-06-16T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- **localStorage 객체** : 웹 브라우저에 **데이터를 저장**
- 웹 브라우저가 기본적으로 제공하는 객체
    - **localStorage.getItem**(**키**)
        - **저장된 값을 추출**
        - 없으면 undefined가 나옴
        - 객체의 속성을 추출하는 일반적인 형태로 **localStorage.키** 또는 **localStorage[키]** 형태로 사용 가능
    - **localStorage.setItem**(**키, 값**)
        - **값을 저장**
        - 객체에 속성을 지정하는 일반적인 형태를 사용 가능
    - **localStorage.removeItem**(**키**)
        - **특정 키의 값을 제거**
    - **localStorage.clear**()
        - **저장된 모든 값을 제거**

- 웹 브라우저에 데이터를 저장하는 localStorage 객체와 활용
    
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
              const p = document.querySelector('p')
              const input = document.querySelector('input')
              const button = document.querySelector('button')
      
              const savedValue = localStorage.getItem('input') // 값을 읽을 때 getItem() 메소드 사용
              // localStorage.input도 가능합니다.
              if (savedValue) {
                input.value = savedValue
                p.textContent = `이전 실행 때의 마지막 값: ${savedValue}`
              }
      
              input.addEventListener('keyup', (event) => {
                const value = event.currentTarget.value
                localStorage.setItem('input', value) // 값을 모두 저장할 때는 setItem() 메소드 사용
                // localStorage.input = value도 가능합니다.
              })
      
              button.addEventListener('click', (event) => {
                localStorage.clear() // 값을 모두제거할 때는 clear() 메소드 사용
                input.value = ''
              })
            })
          </script>
    </head>
    <body>
        <p></p>
        <button>delete</button>
        <input type="text">
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_localStorage_객체/Untitled.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_localStorage_객체/Untitled%201.png)
    
    - 값을 입력하고 새로고침을 하면 값이 저장된 것을 확인 할 수 있다.

- localStorage처럼 웹 브라우저가 제공해주는 기능을 **웹 API**라고 부른다.
    
    [모질라 웹 API 문서 목록 링크](https://developer.mozilla.org/ko/docs/Web/API)
    

 

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판
- [https://developer.mozilla.org/ko/docs/Web/API](https://developer.mozilla.org/ko/docs/Web/API)