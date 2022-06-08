---
title: "[JavaScript] 콜백 함수"
description: ""
date: "2022-06-01T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 자바스크립트는 **함수도 하나의 자료형**이므로 **매개변수로 전달 가능**
- **콜백(callback) 함수** : 매개변수로 전달하는 함수
    - 매개변수를 통해 함수를 받고, 그 함수를 통해 결과값을 도출
- 콜백 함수(1) : **선언적 함수** 사용
    
    ```jsx
    <script>
    	// 함수를 선언
    	function callThreeTimes (callback) {
    		for (let i = 0; i < 3; i++) {
    			callback(i) // callback이라는 매개변수는 함수이므로 호출 가능	
    		}	
    	}
    	
    	function print (i) {
    		console.log(`${i}번 호출`)
    	}
    
    	// 함수 호출
    	callThreeTimes(print)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_콜백_함수/Untitled.png)
    
    - callThreeTimes() 함수는 함수를 매개변수로 받아 해당 함수를 3번 호출
    - callThreeTimes() 함수의 callback 매개변수에 print() 함수를 전달
    - callThreeTimes() 함수 내부에서는 callback(i) 형태로 함수를 호출
- 콜백 함수(2) : **익명 함수** 사용
    
    ```jsx
    <script>
    	// 함수를 선언
    	function callThreeTimes (callback) {
    		for (let i = 0; i < 3; i++) {
    			callback(i) 
    		}	
    	}
    	
    	callThreeTimes(function (i) {
    		console.log(`${i} 함수 호출`)
    	})
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_콜백_함수/Untitled%201.png)
    

### 콜백 함수를 활용하는 함수 : forEach()

- 콜백 함수를 활용하는 가장 기본적인 함수는 **forEach() 메소드**
- **forEach()** : 배열이 갖고 있는 함수로써 단순하게 배열 내부의 요소를 사용해서 콜백 함수를 호출
- 배열이 갖고 있는 메소드 중에서 콜백 함수를 활용하는 메소드는 다음과 같은 형태의 콜백 함수를 사용
    
    ```jsx
    function (value, index, array) { }
    ```
    
- 배열의 forEach() 메소드
    
    ```jsx
    <script>
    	const numbers = [273, 52, 103, 32, 57]
    
    	numbers.forEach(function (value, index, array) {
    		console.log(`${index} data : ${value}`)
    	})
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_콜백_함수/Untitled%202.png)
    

### 콜백 함수를 활용하는 함수 : map()

- **map()** : 배열이 갖고 있는 함수
    - 콜백 함수에서 **리턴한 값들을 기반으로 새로운 배열을 만드는 함수**
- 배열의 map() 메소드
    
    ```jsx
    <script>
    	let numbers = [273, 52, 103, 32, 57]
    	
    	// 배열의 모든 값을 제곱
    	// 매개변수로 value, index, array를 갖는 콜백 함수를 사용
    	numbers = numbers.map(function (value, index, array) {
    		return value * value
    	})
    
    	// 매개변수로 console.log 메소드 자체를 넘김
    	numbers.forEach(console.log)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_콜백_함수/Untitled%203.png)
    
- **원하는 매개변수만 받기**
    - 일반적으로 value만 또는 value와 index만 사용하는 경우가 많음
    - 콜백 함수의 매개변수는 모두 입력할 필요는 없고, 사용하고자 하는 위치의 것만 순서에 맞춰 입력하면 된다.
    
    ```jsx
    <script>
    	let numbers = [273, 52, 103, 32, 57]
    	
    	// 함수 내부에서 value만 사용하므로 value만 매개변수로 넣음
    	numbers = numbers.map(function (value) {
    		return value * value
    	})
    
    	// 매개변수로 console.log 메소드 자체를 넘김
    	numbers.forEach(console.log)
    </script>
    ```
    

### 콜백 함수를 활용하는 함수 : filter()

- **filter()** : 배열이 갖고 있는 함수
    - 콜백 함수에서 **리턴하는 값이 true인 것들만 모아서 새로운 배열을 만드는 함수**
- 배열의 filter() 메소드
    
    ```jsx
    <script>
    	const numbers = [0, 1, 2, 3, 4, 5]
    	const evenNumbers = numbers.filter(function (value) {
    		return value % 2 === 0
    	})
    
    	console.log(`원래 배열 : ${numbers}`)
    	console.log(`짝수 배열 : ${evenNumbers}`)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_콜백_함수/Untitled%204.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판