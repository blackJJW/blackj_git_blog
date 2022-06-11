---
title: "[JavaScript] 객체의 속성과 메소드"
description: ""
date: "2022-06-03T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **배열** 내부에 있는 값을 **요소(element)**
- **객체** 내부에 있는 값을 **속성(property)**
- 객체의 속성도 **모든 형태의 자료형**을 가질 수 있음
    
    ```jsx
    <script>
    	const object = {
    		number: 123,
    		string: 'abc',
    		boolean: true,
    		array: [4, 5, 6],
    
    		method: function (){}
    	}
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_객체의_속성과_메소드/Untitled.png)
    

### 속성과 메소드 구분

- 객체의 속성 중 **함수 자료형**인 속성을 **메소드**(**method**)라고 부름
    
    ```jsx
    <script>
    	const person = {
    		name: 'a',
    		eat: function (food){}
    	}
    
    	// 메소드를 호출
    	person.eat()
    </script>
    ```
    
    - 기본적으로 **화살표 함수는 메소드로 사용하지 않음**

### 메소드 내부에서 this 키워드 사용

- 메소드 내에서 **자기 자신이 가진 속성을 출력**하고 싶을 때는 **자신이 가진 속성임을 분명하게 표시**해야함
- 자기 자신이 가진 속성이라는 것을 표시할 때는 **this 키워드**를 사용
- 메소드 내부에서의 this 키워드
    
    ```jsx
    <script>
    	const person = {
    		name: 'a',
    		eat: function (food) {
    			alert(`${this.name}'s food : ${food}`)
    		}
    	}
    	person.eat(steak)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_객체의_속성과_메소드/Untitled%201.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판