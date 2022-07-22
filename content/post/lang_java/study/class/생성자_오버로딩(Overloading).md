---
title: "[Java] 생성자 오버로딩(Overloading)"
description: ""
date: "2022-07-22T13:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

- 외부에서 제공되는 다양한 데이터들을 이용해서 객체를 초기화하려면 생성자도 다양화될 필요가 있다.
- ex)
    - Car 객체를 생성할 때 외부에서 제공되는 데이터가 없다면 기본 생성자로 Car 객체를 생성해야 함
    - 외부에서 model 데이터가 제공되거나 model과 color가 제공될 경우에도 Car 객체를 생성 가능 해야 함
    - 생성자가 하나 뿐이라면 이러한 요구 조건을 수용할 수 없다.
- 자바는 다양한 방법으로 객체를 생성할 수 잇도록 생성자 오버로딩(Overloading)을 제공
- 생성자 오버로딩 : 매개 변수를 달리하는 생성자를 여러 개 선언하는 것
    
    ![Untitled](/images/lang_java/class/생성자_오버로딩(Overloading)/Untitled.png)
    
- ex) Car 클래스에서 생성자를 오버로딩
    
    ```java
    public class Car {
    	Car() {...}
    	Car(String model) {...}
    	Car(String model, String color) {...}
    	Car(String model, String color, int maxSpeed) {...}
    }
    ```
    
    ### 생성자 오버로딩 시 주의할 점
    
    - 매개 변수의 타입과 개수 그리고 선언된 순서가 똑같은 경우 매개 변수 이름만 바꾸는 것은 생성자 오버로딩이라고 볼 수 없다.
        
        ```java
        Car(String model, String color) {...}
        Car(String color, String model) {...} // 오버로딩 아님
        ```
        
    - 생성자가 오버로딩되어 있을 경우
        - new 연산자로 생성자를 호출할 때 제공되는 매개값의 타입과 수에 의해 호출될 생성자가 결정
        
        ```java
        Car car1 = new Car();
        Car car2 = new Car("A-model");
        Car car3 = new Car("A-model", "검은색");
        Car car4 = new Car("A-model", "검은색", 300);
        ```
        
        - `new Car()`는 기본 생성자로 객체를 생성
        - `new Car(”A-model”)`는 `Car(String model)` 생성자로 객체를 생성
        - `new Car(”A-model”, “검은색”)`는 `Car(String model, String color)` 생성자로 객체를 생성
        - `new Car(”A-model”, “검은색”, 300)`는 `Car(String model, String color, int maxSpeed)` 생성자로 객체를 생성
- ex) Car2 생성자를 오버로딩해서 Car2Ex 클래스에서 다양한 방법으로 Car 객체를 생성
    - 생성자의 오버로딩
        
        ```java
        public class Car2 {
        	// 필드
        	String company = "A자동차";
        	String model;
        	String color;
        	int maxSpeed;
        	
        	// 생성자
        	Car2(){
        	}
        	
        	Car2(String model){
        		this.model = model;
        	}
        	
        	Car2(String model, String color){
        		this.model = model;
        		this.color = color;
        	}
        	
        	Car2(String model, String color, int maxSpeed){
        		this.model = model;
        		this.color = color;
        		this.maxSpeed = maxSpeed;
        	}
        }
        ```
        
    - 객체 생성 시 생성자 선택
        
        ```java
        public class Car2Ex {
        
        	public static void main(String[] args) {
        		Car2 car1 = new Car2();
        		System.out.println("car1.company : " + car1.company);
        		System.out.println();
        		
        		Car2 car2 = new Car2("자가용");
        		System.out.println("car2.company : " + car2.company);
        		System.out.println("car2.model : " + car2.model);
        		System.out.println();
        		
        		Car2 car3 = new Car2("자가용", "검정색");
        		System.out.println("car3.company : " + car3.company);
        		System.out.println("car3.model : " + car3.model);
        		System.out.println("car3.color : " + car3.color);
        		System.out.println();
        		
        		Car2 car4 = new Car2("자가용", "검정색", 300);
        		System.out.println("car4.company : " + car4.company);
        		System.out.println("car4.model : " + car4.model);
        		System.out.println("car4.color : " + car4.color);
        		System.out.println("car4.maxSpeed : " + car4.maxSpeed);
        		System.out.println();
        
        	}
        
        }
        ```
        
        ![Untitled](/images/lang_java/class/생성자_오버로딩(Overloading)/Untitled%201.png)
        

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판