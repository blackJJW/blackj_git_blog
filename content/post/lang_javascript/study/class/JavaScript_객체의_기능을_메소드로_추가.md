---
title: "[JavaScript] 객체의 기능을 메소드로 추가"
description: ""
date: "2022-06-19T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 함수를 메소드로써 **객체 내부에 넣어서 활용**하는 방법
- 객체를 처리하는 함수
    
    ```jsx
    <script>
    	// 객체를 선언
    	const students = []
    	students.push({ 이름: 'John', 국어: 80, 영어: 85, 수학: 90, 과학: 95 })
    	students.push({ 이름: 'Amy', 국어: 76, 영어: 90, 수학: 96, 과학: 91 })
    	students.push({ 이름: 'Jack', 국어: 98, 영어: 60, 수학: 93, 과학: 77 })
    	students.push({ 이름: 'Ann', 국어: 70, 영어: 87, 수학: 87, 과학: 65 })
    	students.push({ 이름: 'Tom', 국어: 87, 영어: 94, 수학: 76, 과학: 82 })
    
    	// students 배열 내부의 객체 모두에 메소드를 추가
    	for (const student of students){
        student.getSum = function (){
    			return this.국어 + this.영어 + this.수학 + this.과학
    		}
    
    		student.getAverage = function() {
    			return this.getSum() / 4
    		}
    	}
    
    	// 출력
    	let output = '이름\t총점\t평균\n'
    	for (const s of students) {
    	    output += `${s.이름} \t${s.getSum()}점 \t${s.getAverage()}점\n`
    	}
    	console.log(output)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_객체의_기능을_메소드로_추가/Untitled.png)
    
    - 함수 **이름 충돌이 발생하지 않고**, 함수를 **잘못 사용하는 경우도 줄일 수 있다**.
    
- 객체를 생성하는 함수
    
    ```jsx
    <script>
    	function createStudent(이름, 국어, 영어, 수학, 과학) {
    		return {
    			// 속성을 선언
    			이름: 이름,
    			국어: 국어,
    			영어: 영어,
    			수학: 수학,
    			과학: 과학,
    			
    			// 메소드를 선언
    			getSum(){
    				return this.국어 + this.영어 + this.수학 + this.과학
    			},
    			getAverage (){
    				return this.getSum() / 4
    			},
    			toString (){
    				return `${this.이름} \t${this.getSum()}점 \t${this.getAverage()}점\n`
    			}
    		}
    	}
    	
    	// 객체를 선언
    	const students = []
    	students.push(createStudent('John', 80, 85, 90, 95))
    	students.push(createStudent('Amy', 76, 90, 96, 91))
    	students.push(createStudent('Jack', 98, 60, 93, 77))
    	students.push(createStudent('Ann', 70, 87, 87, 65))
    	students.push(createStudent('Tom', 87, 94, 76, 82))
    
    	// 출력
    	let output = '이름\t총점\t평균\n'
    	for (const s of students) {
    	    output += s.toString()
    	}
    	console.log(output)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_객체의_기능을_메소드로_추가/Untitled%201.png)
    
    - createStudent() 함수를 만들고, 여기에 객체를 만들어 리턴

- 이렇게 함수를 만들면 **여러 가지 이득이 생김**
    - 객체를 하나하나 만들 때와 비교하여
        - **오탈자의 위험**이 줄어듬
        - **코드를 입력하는 양이 크게 줄어들어듬**
        - 속성과 메소드를 **한 함수 내부에서 관리 가능**하므로 객체를 **더 손쉽게 유지 보수 가능**
        

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판