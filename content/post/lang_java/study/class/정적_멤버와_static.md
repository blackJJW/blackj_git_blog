---
title: "[Java] 정적 멤버와 static"
description: ""
date: "2022-07-24T15:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

- 정적 맴버는 클래스에 고정된 맴버로서 객체를 생성하지 않고 사용할 수 있는 필드와 메소드를 말한다.
    - 각각 정적 필드, 정적 메소르라 부른다.
- 정적 맴버는 객체(인스턴스)에 소속된 맴버가 아니라 클래스에 소속된 맴버이기 때문에 클래스 맴버라고도 한다.

# 정적 멤버 선언

---

- 정적 필드와 정적 메소드를 선언하는 방법
    - 필드와 메소드 선언 시 static 키워드를 추가적으로 붙이면 된다.
    
    ```java
    public class 클래스 {
    	// 정적 필드
    	static 타입 필드 [= 초기값];
    
    	// 정적 메소드
    	static 리턴 타입 메소드( 매개변수선언, ... ) { ... }
    }
    ```
    
- 정적 필드와 정적 메소드는 클래스에 고정된 맴버
    - 클래스 로더가 클래스(바이트 코드)를 로딩해서 메소드 메모리 영역에 적재할 때 클래스 별로 관리
    - 클래스의 로딩이 끝나면 바로 사용 가능
    
    ![Untitled](/images/lang_java/class/정적_멤버와_static/Untitled.png)
    

### 인스턴스 필드 vs 정적 필드 선언 시 판단 기준

- 객체마다 가지고 있어야 할 데이터라면 인스턴스 필드로 선언
- 객체마다 가지고 있을 필요성이 없는 공용적인 데이터인 경우 정적 필드로 선언
- ex) Calculator 클래스에서
    - 원의 넓이나 둘레를 구할 때 필요한 파이($\pi$)는 객체마다 가지고 있을 필요가 없는 변하지 않는 공용적인 데이터
        - 정적 필드로 선언
    - 객체 별로 색상이 다를 경우
        - 인스턴스 필드로 선언
    
    ```java
    public class Calulator {
    	String color; // 계산기 별로 색상이 다를 수 있다.
    	static double pi = 3.14159; // 계산기에서 사용하는 파이 값은 동일
    }
    ```
    

### 인스턴스 메소드 vs 정적 메소드 선언 시 판단 기준

- 인스턴스 필드를 이용해서 실행해야 한다면 인스턴스 메소드로 선언
- 인스턴스 필드를 이용하지 않을 경우 정적 메소드로 선언
- ex) Calculator 클래스에서
    - 덧셈, 뺄셈 기능은 인스턴스 필드를 이용하기 보다는 외부에서 주어진 매개값들을 가지고 수행
        - 정적 메소드로 선언
    - 인스턴스 필드인 색상을 변경하는 메소드
        - 인스턴스 메소드로 선언
    
    ```java
    public class Calculator {
    	Stirng color; // 인스턴스 필드
    	void setColor(String color) {this.color = color;} // 인스턴스 메소드
    	static int plus(int x, int y) {return x + y;} // 정적 메소드
    	static int minus(int x, int y) {return x - y;} // 정적 메소드
    }
    ```
    

# 정적 멤버 사용

---

- 클래스가 메모리로 로딩되면 정적 멤버를 바로 사용 가능
- 클래스 이름과 함께 도트( . ) 연산자로 접근
    
    ```java
    클래스.필드;
    클래스.메소드( 매개값, ... );
    ```
    
- ex) Calculator 클래스
    
    ```java
    public class Calculator {
    	static double pi = 3.14159;
    	static int plus(int x, int y) { ... }
    	static int minus(int x, int y) { ... }
    }
    ```
    
    - 정적 필드 pi와 적 메소드 `plus()`, `minus()`는 다음과 같이 사용 가능
        
        ```java
        double result1 = 10 * 10 * Calculator.pi;
        int result2 = Calculator.plus(10, 5);
        int result3 = Calculator.minus(10, 5);
        ```
        
    - 정적 필드와 정적 메소드는 원칙적으로 클래스 이름으로 접근해야 하지만 객체 참조 변수로도 접근이 가능
        
        ```java
        Calculator myCalc = new Calculator();
        double result1 = 10 * 10 * myCalc.pi;
        int result2 = myCalc.plus(10, 5);
        int result3 = myCalc.munus(10, 5);
        ```
        
- 정적 요소는 클래스 이름으로 접근하는 것이 좋다.
    - 이클립스에서는 정적 멤버를 클래스 이름으로 접근하지 않고 객체 참조 변수로 접근했을 경우, 경고 표시가 나타남.
    
    ![Untitled](/images/lang_java/class/정적_멤버와_static/Untitled%201.png)
    
- ex) 정적 멤버 사용
    
    ```java
    public class Calculator4 {
    	static double pi = 3.14159;
    	
    	static int plus(int x, int y) {
    		return x + y;
    	}
    	
    	static int minus(int x, int y) {
    		return x - y;
    	}
    }
    ```
    
    ```java
    public class Calculator4Ex {
    
    	public static void main(String[] args) {
    		double result1 = 10 * 10 * Calculator4.pi;
    		int result2 = Calculator4.plus(10, 5);
    		int result3 = Calculator4.minus(10, 5);
    		
    		System.out.println("result1 : " + result1);
    		System.out.println("result2 : " + result2);
    		System.out.println("result3 : " + result3);
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/class/정적_멤버와_static/Untitled%202.png)
    

# 정적 초기화 블록

---

- 정적 필드는 필드 선언돠 동시에 초기값을 주는 것이 보통
    
    ```java
    static double pi = 3.14159;
    ```
    
- 계산이 필요한 초기화 작업이 필요한 경우가 있을 수 있음.
- 정적 필드는 객체 생성 없이도 사용해야 하므로 생성자 초기화 작업을 할 수 없다.
    - 생성자는 객체 생성 시에만 실행되기 때문
- 자바는 정적 필드의 복잡한 초기화 작업을 위해 정적 블록(static block)을 제공
    
    ```java
    static {
    	...
    }
    ```
    
    - 정적 블록은 클래스가 메모리로 로딩될 때 자동적으로 실행
    - 정적 블록은 클래스 내부에 여러 개가 선언되어도 상관없다.
        - 클래스가 메모리에 선언된 순서대로 실행
- ex) 정적 초기화 블록
    - 3 개의 정적 필드
    - company와 model은 선언 시 초기값을 설정
    - info는 초기값을 설정하지 않음
    - info 필드는 정적 블록에서 company와 model 필드값을 서로 연결해서 초기값으로 설정
    
    ```java
    public class Television {
    	static String company = "AAA";
    	static String model = "LCD";
    	static String info;
    	
    	static {
    		info = company + "-" + model;
    	}
    }
    ```
    
    ```java
    public class TelevisionEx {
    
    	public static void main(String[] args) {
    		System.out.println(Television.info);
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/class/정적_멤버와_static/Untitled%203.png)
    

# 정적 메소드와 블록 선언 시 주의할 점

---

- 객체가 없어도 실행된다는 특징 때문에, 이들 내부에 인스턴스 필드나 인스턴스 메소드를 사용 불가
- 객체 자신의 참조인 `this` 키워드 사용 불가
- 다음 코드는 컴파일 오류가 발생
    
    ```java
    public class ClassName {
    	// 인스턴스 필드와 메소드
    	int field;
    	void method1() { ... }
    
    	// 정적 필드와 메소드
    	static int field2;
    	static void method2() { ... }
    
    	// 정적 블록
    	static {
    		field1 = 10; // (x) 컴파일 에러
    		method1();   // (x) 컴파일 에러
    		field2 = 10; // (o)
    		method2();   // (o)
    	}
    
    	// 정적 메소드
    	static void Method3 {
    		this.field1 = 10; // (x) 컴파일 에러
    		this.method1();   // (x) 컴파일 에러
        field2 = 10;      // (o)
    		method2();        // (o)
    	}
    }
    ```
    
- 정적 메소드와 정적 블록에서 인스턴스 멤버를 사용하고 싶은 경우
    - 객체를 먼저 생성하고 참조 변수로 접근
    
    ```java
    static void Method3() {
    	ClassName obj = new ClassName();
    	obj.field1 = 10;
    	obj.method1();
    }
    ```
    
- `main()` 메소드도 정적(static) 메소드이므로 객체 생성 없이 인스턴스 필드와 인스턴스 메소드를 `main()` 메소드에서 바로 사용 불가
    
    ```java
    public class Car {
    	int speed;
    
    	void run() { ... }
    
    	public static void main(String[] args) {
    		speed = 60; // (x) 컴파일 에러
    		run(); // (x) 컴파일 에러
    	}
    }
    ```
    

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판