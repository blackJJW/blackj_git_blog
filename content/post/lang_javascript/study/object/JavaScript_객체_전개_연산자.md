---
title: "[JavaScript] 객체 전개 연산자"
description: ""
date: "2022-06-08T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 객체도 깊은 복사를 할 때 **전개 연산자를 사용 가능**
- 전개 연산자를 사용한 객체 복사
    
    ```jsx
    {...객체}
    ```
    

- 얕은 복사로 객체 복사
    
    ```jsx
    <script>
    	const object = {
    		"name" : "John",
    		"age" : 31,
    		"language" : "Python"
    	}
    	const person = object
    	person.name = 'Jack'
    	person.age = 25
    
    	console.log(JSON.stringify(object))
    	console.log(JSON.stringify(person))
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_객체_전개_연산자/Untitled.png)
    
- 전개 연산자를 사용해 깊은 복사
    
    ```jsx
    <script>
    	const object = {
    		"name" : "John",
    		"age" : 31,
    		"language" : "Python"
    	}
    	const person = {...object}
    	person.name = 'Jack'
    	person.age = 25
    
    	console.log(JSON.stringify(object))
    	console.log(JSON.stringify(person))
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_객체_전개_연산자/Untitled%201.png)
    
- 전개 연산자를 사용한 **객체 요소 추가**
    
    ```jsx
    {...객체, 자료, 자료, 자료}
    ```
    
    - 변경하고 싶은 속성만 추가
        
        ```jsx
        <script>
        	const object = {
        		"name" : "John",
        		"age" : 31,
        		"language" : "Python"
        	}
        	const person = {
        		...object,
        		name : 'Jack',
        		age : 25,
        		part : 'Back-end'
        	}
        
        	console.log(JSON.stringify(object))
        	console.log(JSON.stringify(person))
        </script>
        ```
        
        ![Untitled](/images/lang_javascript/study_1/JavaScript_객체_전개_연산자/Untitled%202.png)
        
    - 전개 부분 **뒤로 이동**
        
        ```jsx
        <script>
        	const object = {
        		"name" : "John",
        		"age" : 31,
        		"language" : "Python"
        	}
        	const person = {
        		name : 'Jack',
        		age : 25,
        		part : 'Back-end',
        		...object
        	}
        
        	console.log(JSON.stringify(object))
        	console.log(JSON.stringify(person))
        </script>
        ```
        
        ![Untitled](/images/lang_javascript/study_1/JavaScript_객체_전개_연산자/Untitled%203.png)
        
        - 신규 내용에 기존 내용이 덮어 써졌다.
        
    
    ## References
    
    - 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판