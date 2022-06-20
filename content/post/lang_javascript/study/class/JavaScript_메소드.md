---
title: "[JavaScript] 메소드"
description: ""
date: "2022-06-19T18:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **메소드**(**method**)는 다음과 같은 형태로 추가
    - 이렇게 메소드를 만들면 **내부적으로 메소드가 중복되지 않고 하나만 생성되어 활용**
- 메소드 추가
    
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
    		// 메소드를 선언
    		// 처음 코드를 입력할 때, 메소드 사이에 쉼표(,)를 넣으면 안된다. 
    		getSum(){
    			return this.국어 + this.영어 + this.수학 + this.과학
    		}
    		getAverage (){
    			return this.getSum() / 4
    		}
    		toString (){
    			return `${this.이름} \t${this.getSum()}점 \t${this.getAverage()}점\n`
    		}
    	}
    	
    	// 객체를 선언
    	const students = []
    	students.push(new Student('John', 80, 85, 90, 95))
    	students.push(new Student('Amy', 76, 90, 96, 91))
    	students.push(new Student('Jack', 98, 60, 93, 77))
    	students.push(new Student('Ann', 70, 87, 87, 65))
    	students.push(new Student('Tom', 87, 94, 76, 82))
    
    	// 출력
    	let output = '이름\t총점\t평균\n'
    	for (const s of students) {
    	    output += s.toString()
    	}
    	console.log(output)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_메소드/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판