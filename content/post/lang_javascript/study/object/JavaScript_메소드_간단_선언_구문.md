---
title: "[JavaScript] 메소드 간단 선언 구문"
description: ""
date: "2022-06-03T16:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 메소드 선언시 사용하는 형태
    
    ```jsx
    function () {}
    ```
    
- 최신 버전의 자바스크립트에서는 메소드를 조금 더 쉽게 선언할 수 있는 전용구문이 있음
- 메소드 선언 구문
    
    ```jsx
    <script>
    	const person = {
    		name: 'a',
    		eat (food) {
    			alert(`${this.name}'s food : ${food}`)
    		}
    	}
    
    	person.eat('soup')
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_메소드_간단_선언_구문/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판