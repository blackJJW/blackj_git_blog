---
title: "[Java] String 타입"
description: ""
date: "2022-07-12T13:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

- 자바는 **문자열을 String 변수에 저장**하기 때문에 **String 변수를 우선 선언**해야 한다.
    
    ```java
    String 변수;
    ```
    
- String 변수에 문자열을 저장하려면 큰 따옴표로 감싼 문자열 리터럴을 입력한다.
    
    ```java
    변수 = "문자열";
    ```
    
- 변수 선언과 동시에 문자열을 저장 가능
    
    ```java
    String 변수 = "문자열";
    ```
    
- ex)
    
    ```java
    String name;
    name = "John";
    String lang = "Java";
    ```
    
    ![Untitled](/images/lang_java/referenceType/String_타입/Untitled.png)
    
    - 문자열은 String 객체로 생성되고 변수는 String 객체를 참조
        - 일반적으로 String 변수에 저장한다는 표현을 사용
- 자바는 문자열 리터럴이 동일하다면 String 객체를 공유하도록 되어 있다.
    
    ```java
    String name1 = "John";
    String name2 = "John";
    ```
    
    ![Untitled](/images/lang_java/referenceType/String_타입/Untitled%201.png)
    
- 일반적으로 변수에 문자열을 저장할 경우에는 문자열 리터럴을 사용하지만, **new 연산자를 사용해서 직접 String 객체를 생성시킬 수도 있다.**
- **new 연산자**는 힙 영역에 새로운 객체를 만들 때 사용하는 연산자로 **객체 생성 연산자**라고 한다.
    
    ```java
    String name1 = new String("John");
    String name2 = new String("John");
    ```
    
    - name1과 name2는 서로 다른 객체를 참조
    
    ![Untitled](/images/lang_java/referenceType/String_타입/Untitled%202.png)
    
- 문자열 리터럴로 생성하느냐 new 연산자로 생성하느냐에 따라 **비교 연산자의 결과가 달라질 수 있다.**
- 동일한 리터럴로 String 객체를 생성했을 경우 == 연산의 결과는 true가 나오지만,  new 연산자로 String 객체를 생성했을 경우 == 연산의 결과는 false가 나온다.
    
    ```java
    String name1 = "John";
    String name2 = "John";
    String name3 = new String("John");
    
    System.out.println(name1 == name2);
    System.out.println(name1 == name3);
    ```
    
    ![Untitled](/images/lang_java/referenceType/String_타입/Untitled%203.png)
    
- 동일한 String 객체이건 다른 String 객체이건 상관없이 문자열만을 비교할 때에는 String 객체의 `equals()` 메소드로 사용해야 한다.
    - equals() 메소드는 원본 문자열과 매개값으로 주어진 **비교 문자열이 동일한지 비교한 후 true 또는 false를 리턴**
    
    ```java
    boolean reuslt = str1.equals(str2);
    ```
    
- Stirng 변수는 참조 타입이므로 초기값으로 null을 대입 가능
    - null은 String 변수가 참조하는 String 객체가 없다는 뜻
        
        ```java
        String a = null;
        ```
        
    - a 변수가 String 객체를 참조하였으나, **null을 대입함으로써 더 이상 String 객체를 참조하지 않도록 할 수 있다.**
    - 참조를 읽은 String 객체는 JVM이 참조되지 않은 쓰레기 객체로 취급하고 Garbage Collector를 구동시켜 메모리에서 자동 제거
    
    ```java
    String lang = "Java";
    lang = null;
    ```
    
    ![Untitled](/images/lang_java/referenceType/String_타입/Untitled%204.png)
    

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판