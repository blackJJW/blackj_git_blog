---
title: "[Java] import문"
description: ""
date: "2022-07-26T17:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

- 같은 패키지에 속하는 클래스들은 아무런 조건 없이 다른 클래스를 사용 가능

## 다른 패키지에 속하는 클래스를 사용

---

### 1. 패키지와 클래스를 모두 기술

- ex) com.company 패키지에 소속된 Tire 클래스를 이용
    - 필드를 선언하고 객체를 생성
    
    ```java
    package com.company;
    
    public class Car {
    	com.company.Tire tire = new com.company.Tire();
    }
    ```
    
- 패키지 이름이 짧을 경우에는 불편함이 없지만, 패키지 이름이 길거나 사용해야 할 클래스 수가 많다면 패키지 이름을 붙인다는 것은 전체 코드를 난잡하게 만들 수 있음

### 2. 사용하고자 하는 패키지를 import로 선언

- 클래스를 사용할 때에는 패키지를 생략
    
    ```java
    package com.mycompany;
    
    import com.company.Tire;
    // 또는 import com.company.*;
    
    public class Car {
    	Tire tire = new Tire();
    }
    ```
    
    - import 문이 작성되는 위치는 패키지 선언과 클래스 선언 사이
    - 패키지에 포함된 다수의 클래스를 사용해야 한다면 클래스별로 import문을 작성할 필요 없이 클래스 이름을 생략하고 대신 * 를 사용해서 import 문을 한 번 작성
        - * 는 패키지에 속하는 모든 클래스를 의미
- import 문의 개수를 제한이 없고 필요한 패키지가 있다면 얼마든지 추가 가능

### 주의할 점

- import 문으로 지정된 패키지의 하위 패키지는 import 대상이 아니다.
    - ex) com.mycompany 패키지에 있는 클래스도 사용해야 하고, com.mycompany.project 패키지에 있는 클래스도 사용해야 하는 경우
        
        ```java
        import com.mycompany.*;
        import com.mycompany.project.*;
        ```
        
- 패키지 이름 전체를 기술하는 첫 번째 방법이 꼭 필요한 경우가 존재
    - 서로 다른 패키지에 동일한 클래스 이름이 존재하고 두 패키지가 import 되어 있을 경우에 전부 기술
    - 자바 컴파일러는 어떤 패키지에서 클래스를 로딩할지 결정할 수 없기 때문에 컴파일 에러 발생
- ex) 서로 다른 패키지에 소속된 Tire 클래스 사용 방법
    
    ![Untitled](/images/lang_java/class/import문/Untitled.png)
    
    - 세 패키지를 모두 사용하는 Car.java
        
        ```java
        package com.test.mycompany;
        
        import com.test.companya.*;
        import com.test.companyb.*;
        import com.test.companyc.Engine;
        
        public class Car {
        	// 필드
        	Engine engine = new Engine();
        	SnowTire tire1 = new SnowTire();
        	BigWidthTire tire2 = new BigWidthTire();
        	com.test.companya.Tire tire3 = new com.test.companya.Tire();
        	com.test.companyb.Tire tire4 = new com.test.companyb.Tire();
        }
        ```
        
    

## 필요한 import문 자동 추가 기능

---

- 이클립스에는 개발자가 import 문을 작성하지 않아도 사용된 클래스를 조사해서 필요한 import 문을 자동적으로 추가해주는 기능 존재
- 현재 작성중인 클래스 메뉴
    
    [ Source → Organize imports ]
    
    ![Untitled](/images/lang_java/class/import문/Untitled%201.png)
    
    - 또는 단축키 : `Ctrl` + `Shift` + `O`

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판