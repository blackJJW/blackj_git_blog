---
title: "[Java] 인스턴스 멤버와 this"
description: ""
date: "2022-07-24T14:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

- 인스턴스(instance) 멤버란 객체(인스턴스)를 생성한 후 사용할 수 있는 필드와 메소드
    - 각각 인스턴스 필드, 인스턴스 메소드라 부른다.
    - 인스턴스 필드와 메소드는 객체에 소속된 멤버이기 때문에 객체 없이는 사용 불가
- ex) Car 클래스에 gas 필드와 `setSpeed()` 메소드가 선언
    
    ```java
    public class Car {
    	// 필드
    	int gas;
    
    	// 페소드
    	void setSpeed(int speed) { ... }
    }
    ```
    
    - gas 필드와 `setSpeed()` 메소드는 인스턴스 멤버이기 때문에 외부 클래스에서 사용하기 위해서는 우선 Car 객체(인스턴스)를 생성하고 참조 변수 myCar 또는 yourCar로 접근
        
        ```java
        Car myCar = new Car();
        myCar.gas = 10;
        myCar.setSpeed(60);
        
        Car yourCar = new Car();
        yourCar.gas = 20;
        yourCar.setSpeed(80);
        ```
        
    - 인스턴스 필드 gas는 객체마다 따로 존재
    - 인스턴스 메소드 `setSpeed()`는 객체마다 존재하지않고 메소드 영역에 저장되고 공유
    
    ![Untitled](/images/lang_java/class/인스턴스_멤버와_this/Untitled.png)
    
- 객체 외부에서 인스턴스 멤버에 접근하기 위해 참조 변수를 사용하는 것과 마찬가지로 객체 내부에서도 인스턴스 맴버에 접근하기 위해 `this`를 사용 가능
- 객체는 자신을 “this”라고 한다.
    - `this.model`은 자신이 가지고 있는 model 필드라는 의미
    - this는 주로 생성자와 메소드의 매개 변수 이름이 필드와 동일한 경우
        - 인스턴스 맴버인 필드임을 명시하고자 할 때 사용
    - ex) 매개 변수 model의 값을 필드 model에 저장
        
        ```java
        Car(String model) {
        	this.model = model;
        }
        
        void setModel(String model) {
        	this.model = model;
        }
        ```
        
- ex)
    - 인스턴스 맴버와 this
    
    ```java
    public class Car6 {
    	// 필드
    	String model;
    	int speed;
    	
    	// 생성자
    	Car6(String model){
    		this.model = model;
    	}
    	
    	// 메소드
    	void setSpeed(int speed) {
    		this.speed = speed;
    	}
    	
    	void run() {
    		for(int i = 0; i <= 50; i += 10) {
    			this.setSpeed(i);
    			System.out.println(this.model + "가 달립니다.(시속 : " + this.speed + " km/h)");
    		}
    	}
    }
    ```
    
    - 인스턴스 맴버와 this Ex
    
    ```java
    public class Car6Ex {
    
    	public static void main(String[] args) {
    		Car6 myCar = new Car6("AAA");
    		Car6 yourCar = new Car6("BBB");
    		
    		myCar.run();
    		yourCar.run();
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/class/인스턴스_멤버와_this/Untitled%201.png)
    

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판