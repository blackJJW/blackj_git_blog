---
title: "[JavaScript] 오류의 종류"
description: ""
date: "2022-06-16T16:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- 프로그래밍 언어의 **오류**(**error**)에는 크게 **2가지 종류가 존재**
    - **구문 오류(syntax error)** : 프로그램 **실행 전**에 발생하는 오류
    - **예외(exception)** 또는 **런타임 오류(runtime error)** : 프로그램 **실행 후**에 발생하는 오류

### 구문 오류

- 괄호의 짝을 맞추기 않은 경우
- 문자열을 열었는데 닫지 않은 경우

- 이러한 구문 오류가 있으면 **웹 브라우저가 코드를 분석조차 하지 못하므로 실행 되지 않음**
- 구문 오류가 발생하는 코드
    
    ```jsx
    <script>
    	// 프로그램 시작 확인
    	console.log("# 프로그램 시작")
    
    	// 구문 오류 발생 부분
    	console.log("괄호를 닫지 않음"
    </script>
    ```
    
    - 실행 시 **Syntax Error**라는 오류 발생 : **구문 오류**라는 의미
        
        ![Untitled](/images/lang_javascript/study_2/JavaScript_오류의_종류/Untitled.png)
        
    - 오류 메시지의 내용을 확인하고 해당 코드를 수정하면 오류를 해결할 수 있다.

### 예외

- **예외(exception)** 또는 **런타임 오류**(**runtime error**)는 **실행 중 발생하는 오류**를 의미
- 예외가 발생하는 코드
    
    ```jsx
    <script>
    	// 프로그램 시작 확인
    	console.log("# 프로그램 시작")
    
    	// 구문 오류 발생 부분
    	// 식별자를 잘못 입력
    	console.rog("log를 rog로 작성")
    </script>
    ```
    
    - 프로그램 실행 시 오류 발생
        
        ![Untitled](/images/lang_javascript/study_2/JavaScript_오류의_종류/Untitled%201.png)
        
    - 이전 코드와의 차이점은 코드 중 첫 번째 코드가 실행됐다는 점이다.
        - 즉 일단 코드는 실행된다는 의미
        - **오탈자 수정**을 통해 수정 가능
- **실행 중에 발생하는 오류**가 **예외**
    - 자바스크립트에서 **SyntaxError**라고 출력되는 오류 **이외의 모든 오류(TypeError, ReferenceError, RangeError)가 예외로 분류**
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판