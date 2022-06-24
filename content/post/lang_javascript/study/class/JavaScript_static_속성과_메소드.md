---
title: "[JavaScript] static 속성과 메소드"
description: ""
date: "2022-06-23T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **디자인 패턴**(**design pattern**)
    - 프레임워크 개발자들은 **더 효율적으로 프레임워크를 개발할 수 있게 다양한 패턴을 고안**
- 원래 자바스크립트에는 클래스라는 기능이 없었음.
    - 하지만 여러 디자인 패턴을 사용하기 위해 **클래스 문법들이 계속해서 추가됨**
- 추가된 문법으로 **static 속성**과 **static 메소드**가 있음
    - **정적 속성**, **정적 메소드**라고도 부름
    
    ```jsx
    class 클래스_이름 {
    	static 속성 = 값
    	static 메소드 () {
    
    	}
    }
    ```
    
- **static 속성과 메소드**는 **인스턴스를 만들지 않고 사용할 수 있는** 속성과 메소드
    - 일반적인 **변수와 함수처럼 사용 가능**
    
    ```jsx
    클래스_이름.속성
    클래스_이름.메소드()
    ```
    
- static 키워드 사용
    
    ```jsx
    <script>
    	class Square {
    		#length
    		static #counter = 0 // private 특성과 static 특성은
    												// 한꺼번에 적용 가능
    		static get counter(){
    			return Square.#counter
    		}
    		constructor (length){
    			this.length = length
    			Square.#counter += 1
    		}
    		static perimeterOf (length){
    			return length * 4
    		}
    		static areaOf (length){
    			return length * length
    		}
    
    		get length () { return this.#length }
    		get perimeter () { return this.#length * 4 }
    		get area () { return this.#length * this.#length }
    
    		set length (length) {
    			if (length <= 0) {
    				throw '길이가 0보다 커야함'
    			}
    			this.#length = length
    		}
    	}
    	
    	// static 속성 사용
    	const squareA = new Square(10)
    	const squareB = new Square(20)
    	const squareC = new Square(30)
    	console.log(`지금까지 생성된 Square 인스턴스는 ${Square.counter} 개`)
    	
    	// static 메소드 사용
    	console.log(`한 변의 길이가 20인 정사각형의 둘레는 ${Square.perimeterOf(20)}`)
    	console.log(`한 변의 길이가 30인 정사각형의 넓이는 ${Square.areaOf(30)}`)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_static_속성과_메소드/Untitled.png)
    

---

- 변수와 함수를 **클래스 내부에 작성하면 생기는 장점**이 있음
    - 어떤 속성과 함수가 **클래스 내부에 귀속되어 있다는 것을 명시적으로 나타낼 수 있음**
    - **private 특성**과 **getter**, **setter**를 부여해서 **조금 더 안전한 변수와 함수를 사용 가능**

---

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판