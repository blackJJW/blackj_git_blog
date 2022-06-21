---
title: "[JavaScript] 상속"
description: ""
date: "2022-06-20T17:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- Rectangle이라는 사각형을 나타내는 클래스를 선언하고 사용하는 예
    - getPerimeter( ) : 사각형의 둘레를 구하는 메소드
    - getArea( ) : 사각형의 넓이를 구하는 메소드
    
    ```jsx
    <script>
    	class Rectangle{
    		constructor (width, height){
    			this.width = width
    			this.height = height
    		}
    
    		// 사각형의 둘레
    		getPerimeter(){
    			return 2 * (this.width + this.height)
    		}
    
    		// 사각형의 넓이
    		getArea(){
    			return this.width * this.height
    		}
    	}
    
    	const rectangle = new Rectangle(10, 20)
    	console.log(`사각형의 둘레 : ${rectangle.getPerimeter()}`)
    	console.log(`사각형의 넓이 : ${rectangle.getArea()}`)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_상속/Untitled.png)
    
- 도형을 더 추가하고자 Square라는 이름의 정사각형을 나타내는 클래스를 추가
- Square 클래스 추가
    
    ```jsx
    <script>
    	// 사각형
    	class Rectangle{
    		constructor (width, height){
    			this.width = width
    			this.height = height
    		}
    
    		// 사각형의 둘레
    		getPerimeter(){
    			return 2 * (this.width + this.height)
    		}
    
    		// 사각형의 넓이
    		getArea(){
    			return this.width * this.height
    		}
    	}
    	
    	// 정사각형 
    	class Square{
    		constructor (length){
    			this.length = length
    		}
    
    		// 정사각형의 둘레
    		getPerimeter(){
    			return 4 * this.length
    		}
    
    		// 정사각형의 넓이
    		getArea(){
    			return this.length * this.length
    		}
    	}
    	const square= new Square(10)
    	console.log(`정사각형의 둘레 : ${square.getPerimeter()}`)
    	console.log(`정사각형의 넓이 : ${square.getArea()}`)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_상속/Untitled%201.png)
    

- Rectangle 클래스와 Square 클래스에는 **큰 차이가 없음**
- 클래스를 분리하는 것이 클래스를 활용하는 쪽에서 편리하겠지만, **분리하면 클래스 선언 부분이 복잡해지는 문제가 발생**

- 해결 방법은 **상속**(**inheritance**)
- 클래스의 선언 코드를 **중복해서 작성하지 않도록 함**으로써 코드의 **생산 효율을 올리는 문법**
    
    ```jsx
    class 클래스_이름 extends 부모클래스_이름 {
    
    }
    ```
    
- 상속은 어떤 클래스가 가지고 있는 **속성과 메소드를 다른 클래스에게 물려주는 형태**로 사용
    - 속성과 메소드를 **주는 클래스**는 **부모 클래스**(**parent class**)
    - 속성과 메소드를 **받는 클래스**는 **자식 클래스**(**child class**)

- 사각형 클래스와 정사각형 클래스
    
    ```jsx
    <script>
    	// 사각형 클래스
    	class Rectangle {
    		constructor (width, height){
    			this.width = width
    			this.height = height
    		}
    	
    		// 사각형의 둘레
    		getPerimeter(){
    			return 2 * (this.width + this.height)
    		}
    
    		// 사각형의 넓이
    		getArea(){
    			return this.width * this.height
    		}
    	}
    
    	// 정사각형 클래스
    	// Square 클래스가 자식 클래스
    	class Square extends Rectangle{
    		constructor(length){
    			super(length, length)
    			// 부모의 생성자 함수를 호출
    		}
    		// getPerimeter() 메소드와 getArea() 메소드를 제거
    	}
    
    	// 클래스 사용
    	const square = new Square(10, 20)
    	// 메소드를 상속받았으므로 메소드 사용 가능
    	console.log(`정사각형의 둘레 : ${square.getPerimeter()}`)
    	console.log(`정사각형의 넓이 : ${square.getArea()}`)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_상속/Untitled%202.png)
    
    - **super( ) 함수**는 **부모의 생성자**를 나타낸다.
        - super(length, length)를 호출하면 Rectangle 클래스의 constructor(width, height)가 호출되어 width 속성과 height 속성이 들어간다.
    
## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판