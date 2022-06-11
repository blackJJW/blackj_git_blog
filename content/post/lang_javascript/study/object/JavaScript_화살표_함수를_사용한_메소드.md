---
title: "[JavaScript] 화살표 함수를 사용한 메소드"
description: ""
date: "2022-06-03T17:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- function ( ) { } 형태로 선언하는 **익명 함수**
- ( ) => { } 형태로 선언하는 **화살표 함수**
- 각각 객체의 메소드로 사용될 때 **this 키워드를 다루는 방식이 다름**

- this 키워드의 차이
    
    ```jsx
    <script>
    	const test = {
    		// 익명 함수로 선언
    		a: function () {
    			console.log(this)
    		},
    		// 화살표 함수로 선언
    		b: () => {
    			console.log(this)
    		}
    	}
    	
    	test.a()
    	test.b()
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_화살표_함수를_사용한_메소드/Untitled.png)
    
    - 위 코드를 실행한 결과 test 객체와 **window 객체**를 출력
    
    - 드롭 다운 버튼을 클릭해보면 this 키워드가 나타내는 것이 조금 다르다는 것을 확인 가능
    
    ![Untitled](/images/lang_javascript/study/JavaScript_화살표_함수를_사용한_메소드/Untitled%201.png)
    

- **window 객체**는 웹 브라우저 자체를 나타내는 ‘**웹 브라우저에서 실행하는 자바스크립트의 핵심 객체**’
- 메소드 내부에서 **this 키워드를 사용할 때 의미가 달라지므로 화살표 함수를 메소드로 사용하지 않는 편**
    - 기본적으로 this 키워드가 window 객체를 나타내지만, **상황에 따라 다른 객체를 나타낼 수도 있다.**

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판