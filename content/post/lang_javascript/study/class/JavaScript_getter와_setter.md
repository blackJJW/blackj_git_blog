---
title: "[JavaScript] getter와 setter"
description: ""
date: "2022-06-23T13:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- <mark>**private 속성**</mark>을 사용하면 <mark>**외부에서는 속성에 아예 접근할 수 없는 문제**</mark>가 발생
- 프레임워크 개발자들은 <mark>**상황에 따라서 속성을 읽고 쓸 수 있는 메소드를 만들어 제공**</mark>

---

- **getter와 setter 메소드**
    
    ```jsx
    <script>
    	// 정사각형 메소드
    	class Square {
    		#length
    
    		constructor (length){
    			this.setLength(length)
    		}
    
    		setLength (value){
    			if (value <= 0) {
    				throw `길이는 0보다 커야 합니다.`
    			}
    			this.#length = value
    		}
    		
    		getLength (value) {
    			return this.#length
    		}
    
    		getPerimeter () { return 4 * this.#length }
    		getArea () { return this.#length * this.#length }
    	}
    	// 클래스 사용
    	const square = new Square(10)
    	console.log(`한 변의 길이는 ${square.getLength()}`)
    
    	// 예외 발생
    	square.setLength(-10)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_getter와_setter/Untitled.png)
    
- 코드에 **getLength() 메소드**와 **setLength() 메소드**가 존재
- <mark>**getter**</mark> : get○○( ) 메소드처럼 <mark>**속성 값을 확인할 때 사용**</mark>하는 메소드
- <mark>**setter**</mark> : set○○( ) 메소드처럼 <mark>**속성에 값을 지정할 때 사용**</mark>하는 메소드

---

- <mark>**getter**와 **setter**는 **필요한 경우에만 사용**</mark>
    - 만약 사용자가 **값을 읽는 것을 거부하겠다면 getter를 만들지 않아도 된다.**
    - 또한 사용자가 **값을 지정하는 것을 거부하겠다면 setter를 만들지 않아도 된다.**
    - 아예 속성에 접근하지 못하게 **둘 다 막는 것도 가능**

---

- 코드를 더 쉽게 작성하고 사용할 수 있도록 **get 키워드**와 **set 키워드** 문법 제공
    
    ```jsx
    class 클래스_이름{
    	get 이름 () { return 값 }
    	set 이름 (value) { }
    }
    ```
    
- **get 키워드와 set 키워드 조합**
    
    ```jsx
    <script>
    	// 정사각형 메소드
    	class Square {
    		#length
    
    		constructor (length){
    			this.length = length // this.length 값을 지정하면,
    													 // set length (length) 메소드 부분이 호출 
    		}
    
    		get length () {
    			return this.#length
    		}
    		
    		get perimeter () {
    			return this.#length * 4
    		}
    
    		get area () {
    			return this.#length * this.#length
    		}
    
    		set length (length) {
    			if (length <= 0) {
    				throw '길이는 0보다 커야 한다.'
    			}
    			this.#length = length
    		}
    	}
    
    	// 클래스 사용
    	const squareA = new Square(10)
    	console.log(`한 변의 길이 : ${squareA.length}`) // 속성을 사용하는 형태로 사용하면,
    	console.log(`둘레 : ${squareA.perimeter}`)     // 자동으로 getter와 setter가 호출
    	console.log(`넓이 : ${squareA.area}`)
    
    	// 예외 발생
    	const squareB = new Square(-10)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_getter와_setter/Untitled%201.png)
    

---

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판