---
title: "[Java] 다른 생성자 호출(this())"
description: ""
date: "2022-07-22T14:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

- 생성자 오버로딩이 많아질 경우 생성자 간의 중복된 코드가 발생 가능
    - 매개 변수의 수만 달리하고 필드 초기화 내용이 비슷한 생성자에서 많이 볼 수 있음
    - 이 경우에는 필드 초기화 내용은 한 생성자에만 집중적으로 작성
        - 나머지 생성자는 초기화 내용을 가지고 있는 생성자를 호출하는 방법으로 개선 가능
    - 생성자에서 다른 생성자를 호출할 때 `this()` 코드 사용
        
        ```java
        클래스( [매개변수선언, ... ] ) {
        	this( 매개변수, ..., 값, ...); // 클래스의 다른 생성자 호출
        	실행문;
        }
        ```
        
- `this()`는 자신의 다른 생성자를 호출하는 코드로 반드시 생성자의 첫줄에만 허용
- `this()`의 매개값은 호출되는 생성자의 매개 변수 타입에 맞게 제공
- `this()` 다음에는 추가적인 실행문들이 올 수 있다.
    - 호출되는 생성자의 실행이 끝나면 원래 생성자로 돌아와서 다음 실행문을 진행한다는 의미
- ex) 생성자 오버로딩에서 생기는 중복 코드를 제거
    
    ```java
    Car(String model){
    	// 중복 코드
    	this.model = model;
    	this.color = "은색";
    	this.maxSpeed = 250;
    }
    
    Car(String model, String color){
    	// 중복 코드
    	this.model = model;
    	this.color = color;
    	this.maxSpeed = 250;
    }
    
    Car(String model, String color, int maxSpeed){
    	// 중복 코드
    	this.model = model;
    	this.color = color;
    	this.maxSpeed = maxSpeed;
    }
    ```
    
    - 세 개의 생성자의 내용이 비슷하므로 앞의 두 개의 생성자에서 `this()`를 사용해서 마지막 생성자인 `Car(String model, String color, int maxSpeed)`를 호출하도록 수정
        - 중복 코드 최소화 가능
    - 다른 생성자를 호출해서 중복 코드 줄이기
        
        ```java
        public class Car3 {
        	// 필드
        	String company = "A자동차";
        	String model;
        	String color;
        	int maxSpeed;
        	
        	// 생성자
        	Car3(){
        	}
        	
        	Car3(String model){
        		this(model, "은색", 250);  // 호출
        	}
        	
        	Car3(String model, String color){
        		this(model, color, 250);  // 호출
        	}
        	
        	Car3(String model, String color, int maxSpeed){
        		// 공통 실행 코드
        		this.model = model;
        		this.color = color;
        		this.maxSpeed = maxSpeed;
        	}
        }
        ```
        
    - 객체 생성 시 생성자 선택
        
        ```java
        public class Car3Ex {
        
        	public static void main(String[] args) {
        		Car3 car1 = new Car3();
        		System.out.println("car1.company : " + car1.company);
        		System.out.println();
        		
        		Car3 car2 = new Car3("자가용");
        		System.out.println("car2.company : " + car2.company);
        		System.out.println("car2.model : " + car2.model);
        		System.out.println();
        		
        		Car3 car3 = new Car3("자가용", "흰색");
        		System.out.println("car3.company : " + car3.company);
        		System.out.println("car3.model : " + car3.model);
        		System.out.println("car3.color : " + car3.color);
        		System.out.println();
        		
        		Car3 car4 = new Car3("자가용", "검정", 250);
        		System.out.println("car4.company : " + car4.company);
        		System.out.println("car4.model : " + car4.model);
        		System.out.println("car4.color : " + car4.color);
        		System.out.println("car4.maxSpeed : " + car4.maxSpeed);
        		System.out.println();
        
        	}
        
        }
        ```
        
        ![Untitled](/images/lang_java/class/다른_생성자_호출(this())/Untitled.png)
        

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판