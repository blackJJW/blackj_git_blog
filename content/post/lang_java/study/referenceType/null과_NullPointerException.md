---
title: "[Java] null과 NullPointerException"
description: ""
date: "2022-07-12T12:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

- 참조 타입 변수는 힙 영역의 객체를 참조하지 않는다는 뜻으로 null 값을 가질 수 있다.
- **null 값도 초기값으로 사용**할 수 있기 때문에 null로 초기화된 참조 변수는 **스택 영역에 생성**
    
    ![Untitled](/images/lang_java/referenceType/null과_NullPointerException/Untitled.png)
    
    ```java
    String refVar1 = "객체1";
    String refVar2 = null;
    ```
    
    - 참조 변수가 null 값을 가지는 지 확인하려면 **==, != 연산을 수행**하면 된다.
        
        ```java
        System.out.println(refVar1 == refVar2);
        System.out.println(refVar1 != refVar2);
        ```
        
        ![Untitled](/images/lang_java/referenceType/null과_NullPointerException/Untitled%201.png)
        
        ```java
        System.out.println(refVar2 == null);
        System.out.println(refVar2 != null);
        ```
        
        ![Untitled](/images/lang_java/referenceType/null과_NullPointerException/Untitled%202.png)
        

# NullPointerException

---

- 참조 변수를 사용하면서 가장 많이 발생하는 예외 중 하나
- 참조 타입 변수를 잘못 사용하면 발생
- 참조 타입 변수가 **null을 가지고 있을 경우, 참조 타입 변수를 사용할 수 없다.**
    - 참조 변수를 사용한다는 것은 **객체를 사용하는 것을 의미**하는 데, **참조할 객체가 없으므로 사용할 수가 없는 것**
    - null 값을 가지고 있는 참조 변수를 사용하면 NullPointException 발생
    
    ```java
    int[] intArray = null;
    intArray[0] = 10; // NullPointerException 
    ```
    
    ![Untitled](/images/lang_java/referenceType/null과_NullPointerException/Untitled%203.png)
    
    - 참조하는 배열 객체가 없기 때문
    
    ```java
    String str = null;
    System.out.println(str.length()); // NullPointerException
    ```
    
    ![Untitled](/images/lang_java/referenceType/null과_NullPointerException/Untitled%204.png)
    
    - 참조하는 String 객체가 없기 때문

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판