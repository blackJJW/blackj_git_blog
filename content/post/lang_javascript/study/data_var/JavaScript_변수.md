---
title: "[JavaScript] 변수"
description: ""
date: "2022-05-07T21:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 변수
    - ‘**변할 수 있는 수**’로 값을 **수정**할 수 있다.
    - 변수를 생성할 때는 **let 키워드**를 사용
        - 익스플로러 1.1 미만의 버전에서는 const, let을 지원하지 않음 → 오류 발생
    
    ```jsx
    let 이름 = 변수
    ```
    
    - 기본적인 사용방법은 상수와 같음
    
    ```jsx
    let pi = 3.141592  -> pi라는 이름의 변수를 선언하고, 3.141592라는 값을 지정
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_변수/Untitled.png)
    
    - 변수의 값을 변경할 때는 변수 이름 뒤에 **= 기호를 입력하고 값을 넣어주면 된다.**
    
    ```jsx
    변수 = 값
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_변수/Untitled%201.png)
    

### <u>구문 오류 : Identifier has already been declared</u>

- 특정한 이름의 변수는 **한 파일에서 한 번만 선언**할 수 있다.
- 만약 같은 이름으로 변수를 한 번 더 선언하면 오류가 발생
    - 콘솔에서 입력시에는 사용자 편의를 위해 오류가 발생하지 않도록 설정되어 있음

  ```jsx
  <script>
      let a = "12345"
      let a = "67890"
  </script>
  
  Uncaught SyntaxError: Identifier 'a' has already been declared
  ```

- 해결 방법
    - **다른 이름의 식별자를 사용**해서 변수를 선언하면 해결 가능
    
    ```jsx
    <script>
    	let a = "12345"
    	let b = "67890"
    </script>
    ```
    

- 변수는 상수와 비교했을 때 **유연함**

- var 키워드
    - 변수를 생성할 수 있는 키워드
    - 변수를 중복해서 선언할 수 있다는 위험성, 변수의 범위가 애매하다는 이유로 let 키워드가 등장하면서 대체되었다.

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판