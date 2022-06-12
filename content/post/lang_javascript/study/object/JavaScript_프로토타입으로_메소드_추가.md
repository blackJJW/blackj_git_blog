---
title: "[JavaScript] 프로토타입으로 메소드 추가"
description: ""
date: "2022-06-06T14:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 어떤 객체의 **prototype 객체**에 속성과 메소드를 추가하면 모든 객체와 기본 자료형에서 해당 속성과 메소드를 사용 가능
    
    ```jsx
    객체_자료형_이름.prototype.메소드_이름 = function () {
    
    }
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_프로토타입으로_메소드_추가/Untitled.png)
    

- 프로토타입에 메소드를 추가하면 다양하게 활용 가능
- 프로토타입으로 숫자 메소드 추가
    
    ```jsx
    <script>
    	// power() 메소드를 추가
    	Number.prototype.power = function (n = 2) {
    		return this.valueOf() ** n
    	}
    
    	// Number 객체의 power() 메소드를 사용
    	const a = 12
    	console.log(`a.power() : ${a.power()}`)
    	console.log(`a.power(3) : ${a.power(3)}`)
    	console.log(`a.power(4) : ${a.power(4)}`)
    </script>
    ```
    
    - this.**valueOf**() 로 값을 추출
        - 객체 내부에서 값을 꺼낸 쓰는 것임을 명확하게 하기 위해 **valueOf**() 메소드를 사용하는 것이 일반적
    
    ![Untitled](/images/lang_javascript/study/JavaScript_프로토타입으로_메소드_추가/Untitled%201.png)
    
- **문자열** 내부에 어떤 문자열이 있는지, 배열 내부에 어떤 자료가 있는 지 확인할 때는 **indexOf**() 메소드를 사용
    
    ![Untitled](/images/lang_javascript/study/JavaScript_프로토타입으로_메소드_추가/Untitled%202.png)
    

- **배열의 indexOf() 메소드**
    
    ![Untitled](/images/lang_javascript/study/JavaScript_프로토타입으로_메소드_추가/Untitled%203.png)
    

- “**문자열.indexOf(문자열) >= 0**”등의 코드를 사용하면 문자열 내부에 어떤 문자열이 포함되어 있는지 true 또는 false로 얻을 수 있다.
- 하지만 이를 “**문자열.contain(문자열)**” 했을 때 true 또는 false로 출력하는 **형태로 변형**하면 좀 더 쉽게 사용할 수 있다.
- 프로토타입으로 문자열 메소드 추가
    
    ```jsx
    <script>
    	// contain() 메소드 추가
    	String.prototype.contain = function (data) {
    		return this.indexOf(data) >= 0
    	}
    
    	Array.prototype.contain = function (data) {
    		return this.indexOf(data) >= 0
    	}
    
    	// String 객체의 contain() 메소드
    	const a = 'Hello World!!!'
    	console.log(`Hello in ${a} : ${a.contain('Hello')}`)
    	console.log(`Bye in ${a} : ${a.contain('Bye')}`)
    
    	// Array 객체의 contain() 메소드
    	const b = [1, 2, 3, 4, 5]
    	console.log(`3 in ${b} : ${b.contain(1)}`)
    	console.log(`6 in ${b} : ${a.contain(6)}`)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_프로토타입으로_메소드_추가/Untitled%204.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판