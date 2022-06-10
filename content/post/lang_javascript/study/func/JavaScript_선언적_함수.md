---
title: "[JavaScript] 선언적 함수"
description: ""
date: "2022-05-27T16:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 일반적으로 **이름이 있는 함수**를 많이 사용
    
    ```jsx
    function 함수() {
    
    }
    ```
    
    ```jsx
    let 함수 = function () { };
    ```
    
- 선언적 함수 선언
    
    ```jsx
    <script>
    	// 함수를 생성
    	function func () {
    		console.log('함수 내부 코드 - 1')
    		console.log('함수 내부 코드 - 2')
    		console.log('함수 내부 코드 - 3')
    		console.log(' ')
    	}
    
    	func()
    	func()
    
    	console.log(typeof func)
    	console.log(func)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_선언적_함수/Untitled.png)
    
    - 함수에 **이름이 있는 것을 확인**

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판