---
title: "[JavaScript] 클래스 선언"
description: ""
date: "2022-06-19T16:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **객체 지향 프로그래밍**(**Object Oriented Programming**) : **객체들을 정의**하고 **객체들을 활용**해서 프로그램을 만드는 것
- **클래스**(**class**)와 **프로토타입**(**prototype**)이라는 2가지 문법으로 **객체를 효율적으로 생성**
    - **클래스** : 객체를 만들 때 수많은 지원을 하는 대신 많은 제한을 거는 문법
    - **프로토타입** : 제한을 많이 하지 않지만, 대신 지원도 별로 하지 않는 문법

- **클래스**
    
    ```jsx
    class 클래스_이름 {
    
    }
    ```
    

- **인스턴스**(**instance**) : 클래스를 기반으로 만든 **객체**(**object**), 그냥 객체라고 부르는 경우도 많음
    
    ```jsx
    new 클래스_이름()
    ```
    

- 클래스 선언하고 인스턴스 생성
    
    ```jsx
    <script>
    	// 클래스 선언
    	class Student {
    
    	}
    	// 학생을 선언
    	const student = new Student()
    	
    	// 학생 리스트를 선언
    	const students = [
    		new Students(),
    		new Students(),
    		new Students(),
    		new Students(),
    		new Students(),
    	]
    </script>
    ```
    

- 클래스 이름은 **첫 글자를 대문자로 지정**하는 것이 **개발자들의 약속**

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판