---
title: "[JavaScript] JSON 객체"
description: ""
date: "2022-06-06T16:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 인더넷에서 문자열로 데이터를 주고 받을 때는 **CSV, XML, CSON 등의 다향한 자료 표현 방식을 사용** 가능
- 현재 가장 많이 사용되는 자료 표현 방식은 **JSON**
- **JSON**(**JavaScript Object Notation**) : 자바스크립트의 **객체처럼 자료를 표현**하는 방식
    - 하나의 자료
        
        ```jsx
        {
        	"name": "John",
        	"age": 31,
        	"program_language": "Python"
        }
        ```
        
    - 여러 개의 자료
        
        ```jsx
        [{
        	"name": "John",
        	"age": 31,
        	"program_language": "Python"
        }, {
        	"name": "Ann",
        	"age": 28,
        	"program_language": "JAVA"
        }]
        ```
        
- **JSON 형식은 약간의 추가 규칙이 존재**
    - 값을 표현할 때는 **문자열, 숫자, 불 자료형**만 사용 가능(**함수 등은 사용 불가**)
    - **문자열**은 반드시 **큰따옴표**로 만들어야함
    - **키**(**key**)에도 반드시 **큰따옴표**를 붇어야함
- 대부분의 프로그래밍 언어는 JSON 형식의 문자열을 읽어들이는 기능이 있음
- **네트워크를 통해**서 각각의 프로그래밍 언어로 만든 애플리케이션들이 **데이터를 교환할 때 활용**

- 자바스크립트 객체를 JSON 문자열로 변환할 때는 **JSON.stringify**() 메소드를 사용
- **JSON.stringify**() 메소드
    
    ```jsx
    <script>
    	const data = [{
    	"name": "John",
    	"age": 31,
    	"program_language": "Python"
    }, {
    	"name": "Ann",
    	"age": 28,
    	"program_language": "JAVA"
    }]
    
    	console.log(JSON.stringify(data))
    	console.log(JSON.stringify(data, null, 2))
    </script>
    ```
    
    - stringify()의 **두 번째 매개변수**
        - 객체에서 **어떤 속성만 선택해서 추출**하고 싶을 때 사용하나 **거의 사용하지 않음**
        - **일반적으로 null**을 넣음
    - stringify()의 **세 번째 매개변수**
        - **들여쓰기 설정**
        - 2 : 들여쓰기 2칸으로 설정
    
    ![Untitled](/images/lang_javascript/study/JavaScript_JSON_객체/Untitled.png)
    

- **JSON 문자열**을 자바스크립트 객체로 전개할 때는 **JSON.parse**() 메소드를 사용
    
    ```jsx
    <script>
    	const data = [{
    	"name": "John",
    	"age": 31,
    	"program_language": "Python"
    }, {
    	"name": "Ann",
    	"age": 28,
    	"program_language": "JAVA"
    }]
    
    	// 자료를 JSON으로 변환
    	const json = JSON.stringify(data)
    	console.log(json)
    
    	// JSON 문자열을 다시 자바스크립트 객체로 변환
    	console.log(JSON.parse(json))
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_JSON_객체/Untitled%201.png)
    

- 추가로 **JSON 객체 정보는 다음 링크**를 참조
    
    [모질라 JSON 객체의 속성과 메소드](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/JSON)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판
- https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/JSON