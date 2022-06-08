---
title: "[JavaScript] 화살표 함수"
description: ""
date: "2022-06-01T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 단순한 형태의 **콜백 함수를 쉽게 입력**하고자 **화살표(arrow) 함수라는 함수 생성 방법**
- function 키워드 대신 **화살표**(**=>**)를 사용
- 다음과 같은 형태로 생성
    
    ```jsx
    (매개변수) => {
    
    }불 표현식 || 불 표현식이 false일 때 실행할 문장 
    ```


- 화살표 함수는 다음과 같이 간편하게 사용
    
    ```jsx
    (매개변수) => 리턴값
    ```
    
    ```jsx
    const array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    array.map((value) => value * value)
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_화살표_함수/Untitled.png)
    


- 배열의 메소드와 화살표 함수
    
    ```jsx
    <script>
    	let numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    	numbers
    		.filter((value) => value % 2 === 0)
    		.map((value) => value * value)
    		.forEach((value) => {
    			console.log(value)
    		}) // 메소드 체이닝
    </script>
    ```
    
    - **메소드 체이닝(method chaining)** : 어떤 메소드가 리턴하는 값을 기반으로 해서 **함수를 줄줄이 사용하는 것**
    
    ![Untitled](/images/lang_javascript/study/JavaScript_화살표_함수/Untitled%201.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판