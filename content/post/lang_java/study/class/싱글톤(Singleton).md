---
title: "[Java] 싱글톤(Singleton)"
description: ""
date: "2022-07-24T16:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

- 전체 프로그램에서 단 하나의 객체만 만들도록 보장해야하는 경우가 존재
    - 단 하나만 생성된다고 해서 이 객체를 싱글톤(Singleton)
- 싱글톤을 만들려면 클래스 외부에서 new 연산자로 생성자를 호출할 수 없도록 막아야 한다.
    - 생성자를 호출한 만큼 생성되기 때문
- 생성자를 외부에서 호출할 수 없도록 하는 방법
    - 생성자 앞에 private 접근 제한자를 붙여준다.
    - 자신의 타입인 정적 필드를 하난 선언하고 자신의 객체를 생성해 초기화 한다.
        - 클래스 내부에서는 new 연산자로 생성자 호출이 가능
    - 정적 필드도 private 접근 제한자를 붙여 외부에서 필드값을 변경하지 못하도록 막는다.
        - 외부에서 호출할 수 있는 정적 메소드인 `getInstance()`를 선언하고 정적 필드에서 참조하고 있는 자신의 객체를 리턴
- ex) 싱글톤 생성
    
    ```java
    public class 클래스 {
    	// 정적 필드
    	private static 클래스 singleton = new 클래스();
    	
    	// 생성자
    	private 클래스() {}
    
    	// 정적 메소드
    	static 클래스 getInstance() {
    		return singleton;
    	} 
    }
    ```
    
- 외부에서 객체를 얻는 유일한 방법
    - `getInstance()` 메소드를 호출
    - `getInstance()` 메소드는 단 하나의 객체만 리턴
        - 아래 코드에서 변수1과 변수2는 동일한 객체를 리턴
        
        ```java
        클래스 변수1 = 클래스.getInstance();
        클래스 변수2 = 클래스.getInstance();
        ```
        
        ![Untitled](/images/lang_java/class/싱글톤(Singleton)/Untitled.png)
        
- ex) 싱글톤
    - 싱글톤
    
    ```java
    public class Singleton {
    	private static Singleton singleton = new Singleton();
    	
    	private Singleton() {}
    	
    	static Singleton getInstance() {
    		return singleton;
    	}
    }
    ```
    
    - 싱글톤 객체
    
    ```java
    public class SingletonEx {
    
    	public static void main(String[] args) {
    		/*
    		 * Singleton obj1 = new Singleton(); // 컴파일 에러
    		 * Singleton obj2 = new Singleton(); // 컴파일 에러
    		 */
    		
    		Singleton obj1 = Singleton.getInstance();
    		Singleton obj2 = Singleton.getInstance();
    		
    		if(obj1 == obj2) {
    			System.out.println("같은 Singleton 객체");
    		} else {
    			System.out.println("다른 Singleton 객체");
    		}
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/class/싱글톤(Singleton)/Untitled%201.png)
    

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판