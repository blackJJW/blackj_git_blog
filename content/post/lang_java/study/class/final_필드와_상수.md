---
title: "[Java] final 필드와 상수"
description: ""
date: "2022-07-24T17:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

# final 필드

---

- final 필드 : 최종적인 필드
    - 초기값이 저장되면 이것이 최종적인 값이 되어 프로그램 실행 도중 수정 불가
    
    ```java
    final 타입 필드 [= 초기값];
    ```
    

### final 필드에 초기값을 주는 방법

- 필드 선언 시에 주는 방법
    - 단순 값인 경우 필드 선언 시에 주는 것이 제일 간단
- 생성자에게 주는 방법
    - 복잡한 초기화 코드가 필요하거나 객체 생성 시에 외부 데이터로 초기화하는 경우, 생성자에서 초기값을 지정해야 한다.
    - 생성자는 final 필드의 최종 초기화를 마쳐야 한다.
        - 만약 초기화되지 않은 final 필드를 그대로 남겨두면 컴파일 에러가 발생
- ex) 주민등록번호 필드를 한 번 값이 저장되면 변경할 수 없도록 final 필드로 선언
    - 생성자 매개값으로 주민등록번호를 받아서 초기값으로 지정
    - nation은 항상 고정된 값을 갖기 때문에 필드 선언 시 초기값으로 “Korea” 지정
    - final 필드 선언과 초기화
    
    ```java
    public class Person {
    	final String nation = "Korea";
    	final String ssn;
    	String name;
    	
    	public Person(String ssn, String name) {
    		this.ssn = ssn;
    		this.name = name;
    	}
    }
    ```
    
    - final 필드 테스트
    
    ```java
    public class PersonEx {
    
    	public static void main(String[] args) {
    		Person p1 = new Person("123456-1234567", "ABC");
    		
    		System.out.println(p1.nation);
    		System.out.println(p1.ssn);
    		System.out.println(p1.name);
    		
    		//p1.nation = "USA";
    		//p1.ssn = "654321-7654321";
    		p1.name = "DEF";
    		System.out.println(p1.name);
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/class/final_필드와_상수/Untitled.png)
    
    - final 필드는 값 수정 불가
    
    ![Untitled](/images/lang_java/class/final_필드와_상수/Untitled%201.png)
    

# 상수(static final)

---

- 일반적으로 불변의 값을 상수
    - ex) 원주율 파이($\pi$), 지구의 무게 및 둘레 등이 해당
    - 이런 불변의 값을 저장하는 필드를 자바에서는 상수(constant)
- final 필드는 한 번 초기화되면 수정할 수 없는 필드
    - final 필드를 상수라 부르지 않는다.
    - 불변의 값은 객체마다 저장할 필요가 없는 공용성을 띠고 있으며, 여러 가지 값으로 초기화될 수 없기 때문
    - final 필드는 객체마다 저장되고, 생성자의 매개값을 통해 여러 가지 값를 가질 수 있기 때문에 상수가 될 수 없다.
    - 상수는 static이면서 final이어야 한다.
        - static final 필드는 객체마다 저장되지 않고, 클래스에만 포함
        - 한 번 초기값이 저장되면 변경 불가
    
    ```java
    static final 타입 상수 [= 초기값];
    ```
    
- 초기값이 단순 값이라면 선언 시에 주는 것이 일반적이지만, 복잡한 초기화일 경우 정적 블록에서도 가능
    
    ```java
    static final 타입 상수;
    static {
    	상수 = 초기값;
    }
    ```
    
- 상수 이름은 모두 대문자로 작성하는 것이 관례
    - 만약 서로 다른 단어가 혼합된 이름이라면 언더바( _ )로 단어들을 연결
    
    ```java
    static final double PI = 3.14159;
    static final double EARTH_SURFACE_AREA;
    ```
    
- ex)
    - 상수 선언
    
    ```java
    public class Earth {
    	static final double EARTH_RADIUS = 6400;
    	static final double EARTH_SURFACE_AREA;
    	
    	static {
    		EARTH_SURFACE_AREA = 4 * Math.PI * EARTH_RADIUS * EARTH_RADIUS;
    	}
    }
    ```
    
    - 상수 사용
    
    ```java
    public class EarthEx {
    
    	public static void main(String[] args) {
    		System.out.println("지구의 반지름 : " + Earth.EARTH_RADIUS + " km");
    		System.out.println("지구의 표면적 : " + Earth.EARTH_SURFACE_AREA + " km^2");
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/class/final_필드와_상수/Untitled%202.png)
    

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판