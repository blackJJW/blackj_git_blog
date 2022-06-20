---
title: "[JavaScript] 생성자"
description: ""
date: "2022-06-19T17:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **생성자**(**constructor**) : **객체가 생성될 때 호출**된다.
    
    ```jsx
    class 클래스_이름 {
    	constructor () {
    		/* 생성자 코드 */
    	}
    }
    ```
    
    - 메소드의 이름을 constructor로 지정했지만 constructor라는 이름으로 사용하는 것이 아니라 **클래스 이름으로 호출하여 사용**한다.
- 생성자는 클래스를 기반으로 **인스턴스를 생성할 때 처음 호출되는 메소드**
    - 생성자에서는 속성을 추가하는 등 **객체의 초기화 처리를 수행**
- 생성자 함수와 속성 추가
    
    ```jsx
    <script>
    	class Student{
    		constructor (이름, 국어, 영어, 수학, 과학){
    			this.이름 = 이름
    			this.국어 = 국어
    			this.영어 = 영어
    			this.수학 = 수학
    			this.과학 = 과학
    		}
    	}
    
    	// 객체를 선언
    	const students = []
    	students.push(new Student('John', 80, 85, 90, 95))
    	students.push(new Student('Amy', 76, 90, 96, 91))
    	students.push(new Student('Jack', 98, 60, 93, 77))
    	students.push(new Student('Ann', 70, 87, 87, 65))
    	students.push(new Student('Tom', 87, 94, 76, 82))
    </script>
    ```
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판