---
title: "[JavaScript] 기본 자료형을 객체로 선언"
description: ""
date: "2022-06-05T14:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 자바스크립트는 **기본 자료형을 객체로 선언하는 방법을 제공**
- 숫자, 문자열, 불 등으로 **자료형을 변환하는 함수**(**Number**( ), **String**( ), **Boolean**( )) 사용
    
    ```jsx
    const 객체 = new 객체_자료형_이름()
    ```
    
    ```jsx
    new Number(10)
    new String('Hello World!!!')
    new Boolean(true)
    ```
    

- 단순한 기본 자료형이 아니므로 **속성을 가짐**
- **속성과 메소드를 활용 가능**
    
    ![Untitled](/images/lang_javascript/study/JavaScript_기본_자료형을_객체로_선언/Untitled.png)
    
    - 속성을 설정한 뒤 콘솔에서 단순하게 출력을 하면 **객체의 형태로 출력**
    - **숫자와 똑같이 활용**할 수 있고 **valueOf() 메소드를 사용해 값을 추출**하는 것이 가능
    
- **new 키워드를 사용하지 않을 때 주의할 점**
    - new 키워드를 사용하지 않으면 **함수가 자료형 변환 기능으로 작동**
        
        ![Untitled](/images/lang_javascript/study/JavaScript_기본_자료형을_객체로_선언/Untitled%201.png)
        
        - typeof() 출력이 **객체가 아니다**.

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판