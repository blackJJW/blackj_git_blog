---
title: "[JavaScript] 오버라이드"
description: ""
date: "2022-06-24T13:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **오버라이드**(**override**)
    - **부모가 갖고 있는 함수를 자식에서 다시 선언해서 덮어쓰는 것**
    - 프레임워크를 다룰 때 **반드시 사용**하는 개념

---

- 메소드에서 순서대로 메소드 호출
    
    ```jsx
    <script>
    	// 클래스를 선언
    	class LifeCycle {
    		call () {
    			this.a()
    			this.b()
    			this.c()
    		}
    		a () { console.log('a() 메소드 호출') }
    		b () { console.log('b() 메소드 호출') }
    		c () { console.log('c() 메소드 호출') }
    	}
    	// 인스턴스 생성
    	new LifeCycle().call()
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_오버라이드/Untitled.png)
    

- **오버라이드**
    
    ```jsx
    <script>
    	// 클래스를 선언
    	class LifeCycle {
    		call () {
    			this.a()
    			this.b()
    			this.c()
    		}
    		a () { console.log('a() 메소드 호출') }
    		b () { console.log('b() 메소드 호출') }
    		c () { console.log('c() 메소드 호출') }
    	}
    
    	class Child extends LifeCycle{
    		a () {                        // 오버라이드
    			console.log('자식의 a() 메소드') 
    		}
    	}
    	// 인스턴스 생성
    	new Child().call()
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_오버라이드/Untitled%201.png)
    
    - 코드 실행시 **원래 a( ) 메소드에 있던 출력이 바뀌는 것을 확인**
    - call( ) 메소드에서 a( ) 메소드를 실행하는 데, **a( ) 메소드가 덮여쓰여져 새로운 a( ) 메소드의 내용을 출력**하는 것이 전부

---

- 부모에 있던 메소드의 내용도 사용하고 싶을 시 `super.메소드()` 형태의 코드를 사용
- `super.a()`는 부모의 a() 메소드를 실행하는 코드
- 부모에 있던 내용 가져오기
    
    ```jsx
    <script>
    	// 클래스를 선언
    	class LifeCycle { 
    		call () {
    			this.a()
    			this.b()
    			this.c()
    		}
    		a () { console.log('a() 메소드 호출') }
    		b () { console.log('b() 메소드 호출') }
    		c () { console.log('c() 메소드 호출') }
      }
    	class Child extends LifeCycle {
    		a () {
    			super.a()
    			console.log('자식의 a() 메소드')
    		}
    	}
    	// 인스턴스를 생성
    	new Child().call()
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_오버라이드/Untitled%202.png)
    

---

### 오버라이드 예

- 어떤 **객체를 문자열로 만드는 메소드**는 **toString**( )
- 자바스크립트의 **모든 객체는 toString()이라는 메소드를 갖음**
    - 숫자, 문자열, 불, 배열, 함수, 클래스, 클래스의 인스턴스 모두 toString() 메소드를 갖고 있음
- 자바스크립트가 **Object라는 최상위 클래스를 가지며**, **어떤 클래스를 생성해도 자동으로 Object 클래스를 상속받게 되어 발생**하는 현상
    - toString()이라는 이름으로 메소드를 만들면 **Object 클래스에 있던 toString() 메소드를 오버라이드하는 것**이다.
- toString() 메소드 오버라이드
    
    ```jsx
    <script>
    	class Pet {
    		constructor (name, age) {
    			this.name = name
    			this.age = age
    		}
    		toString () {    // 오버라이드
    			return `이름 : ${this.name}\n나이 : ${this.age}` 
    		}
    	}
    	const pet = new pet('a', 5)
    	alert(pet)
    	console.log(pet + '')
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_오버라이드/Untitled%203.png)
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_오버라이드/Untitled%204.png)
    
    - 자바스크립트의 **alert() 함수**는 **매개변수로 받은 자료를 문자열로 바꾼 뒤에 출력**
        - toString() 메소드를 오버라이드했으므로 바꾼 형태로 출력되는 것을 확인
    - 문자열과 다른 자료형을 결합할 때도 **내부적으로 다른 자료형을 문자열로 변환한 뒤 결합**
        - 문자열 결합 연산자를 호출할 때도 오버라이드한 toString() 메소드의 리턴값이 나오는 것을 확인

---

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판