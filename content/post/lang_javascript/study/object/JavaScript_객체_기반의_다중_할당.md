---
title: "[JavaScript] 객체 기반의 다중 할당"
description: ""
date: "2022-06-08T14:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **객체 내부에 있는 속성을 꺼내서 변수로 할당**할 때 다음과 같은 코드를 사용 가능
- **객체 속성 꺼내서 다중 할당**
    
    ```jsx
    { 속성 이름, 속성 이름 } = 객체
    { 식별자 = 속성 이름, 식별자 = 속성 이름} = 객체
    ```
    
    ```jsx
    <script>
    	// 객체를 생성
    	const object = {
    		"name" : "John",
    		"age" : 31,
    		"language" : "Python"
    	}
    
    	// 객체에서 변수를 추출
    	const { name, language } = object
    	console.log('# 속성 이름 그대로 꺼내서 출력')
    	console.log(name, language)
    	console.log('')
    
    	const {a = name, b = language} = object
    	console.log('# 다른 이름으로 속성 꺼내서 출력')
    	console.log(a, b)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_객체_기반의_다중_할당/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판