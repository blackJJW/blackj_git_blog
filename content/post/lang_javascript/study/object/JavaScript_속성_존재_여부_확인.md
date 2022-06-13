---
title: "[JavaScript] 속성 존재 여부 확인"
description: ""
date: "2022-06-08T13:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 객체 내부에 어떤 속성이 있는 지 확인해보는 코드
- 객체에 없는 속성에 접근을 하면 **undefined 자료형이 출력**된다.
    - **조건문**으로 **undefined인지 아닌지 확인**하면 속성 존재 여부를 확인 가능
- 속성 존재 여부 확인
    
    ```jsx
    <script>
    	// 객체를 생성
    	const object = {
    		"name" : "John",
    		"age" : 31,
    		"language" : "Python"
    	}
    
    	// 객체 내부에 속성이 있는 지 확인
    	if (object.name !== undefined) {
    		console.log('naem 속성 있음')
    	} else {
    		console.log('name 속성 없음'
    	}
    
    	if (object.part !== undefined) {
    		console.log('part 속성 있음')
    	} else {
    		console.log('part 속성 없음'
    	}
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_속성_존재_여부_확인/Untitled.png)
    

- 객체의 특성 속성이 **false로 변환될 수 있는 값(0, false, 빈 문자열 등)이 아닐 때** 사용하는 코드
    
    ```jsx
    // 객체 내부에 속성이 있는지 확인
    if (object.name) {
    	console.log('name 속성 있음')
    } else {
    	console.log('name 속성 없음')
    }
    
    if (object.part) {
    	console.log('part 속성 있음')
    } else {
    	console.log('part 속성 없음')
    }
    
    ```
    
    - 속성이 **빈 문자열이면 데이터가 없는 것**이나 마찬가지
        - 따라서 **절대 false로 변환될 수 있는 값이 아닐 것**
        - 이와 같은 **전제가 있어야 안전하게 사용**할 수 있는 코드
    
- 더 짧은 조건문 사용
    
    ```jsx
    // 객체 내부에 속성이 있는 지 확인
    object.name || console.log('name 속성 있음')
    object.part || console.log('part 속성 있음')
    ```
    
    - 객체의 속성이 있는지 확인하고 있다면 **해당 속성**을, 없다면 **별도의 문자열**을 지정하는 코드

- 기본 속성 지정
    
    ```jsx
    <script>
    	// 객체를 생성
    	const object = {
    		"name" : "John",
    		"age" : 31,
    		"language" : "Python"
    	}
    
    	// 객체의 기본 속성을 지정
    	object.name = object.name !== undefined ? object.name : 'name 속성 없음'
    	object.part = object.part !== undefined ? object.part : 'part 속성 없음'
    
    	// 객체를 출력
    	console.log(JSON.stringify(object, null, 2))
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_속성_존재_여부_확인/Untitled%201.png)
    

- 속성이 **false로 변환될 수 있는 값이 들어오지 않을 것이라는 전제**가 있으면 다음고 같은 짧은 조건문으로도 구현 가능
    
    ```jsx
    // 객체의 기본 속성을 지정
    object.name = object.name || 'name 속성 없음'
    object.part = object.part || 'part 속성 없음'
    
    ```
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판