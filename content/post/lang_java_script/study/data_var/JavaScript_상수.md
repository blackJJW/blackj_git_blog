---
title: "[JavaScript] 상수"
description: ""
date: "2022-05-07T21:20:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- **상수(constant)**
    - ‘**항상 같은 수**’라는 의미로 값에 **이름을 한 번 붙이면 값을 수정할 수 없다**.
    - **선언** : 상수를 만드는 과정
    - **const 키워드**로 선언
    
    ```jsx
    const 이름 = 값
    ```
    
    - 3.141592를 pi라는 이름으로 선언
    
    ```jsx
    const pi = 3.141592   -> pi라는 이름의 상수를 선언하고, 3.141592라는 값을 할당
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_상수/Untitled.png)
    
    - 상수를 만든 뒤에 상수 이름을 상용해서 자료를 사용 가능
        - 숫자 상수는 숫자 연산
        - 문자열 상수는 문자열 연산

### <u>구문오류 : Identifier has already declared</u>

- 특정한 이름의 상수는 한 파일에서 한 번만 선언 가능
- 만약 같은 이름으로 상수를 한 번 더 선언하면 오류 발생

  ```jsx
  >const name = "11111"
  undefined
  >const name = "22222"
  Uncaught SyntaxError: Identifier 'name' has already declared
  ```

- 해결 방법
    1. 콘솔에서 코드를 입력하여 발생한 경우, 새로고침(F5)를 눌러 자바스크립트 상태를 초기화한 후 다시 코드를 입력한다.
    2. 다른 이름의 식별자를 사용하여 상수를 선언한다.

### <u>구문오류 : Missing initializer in const declaration</u>

- 상수는 한 번만 선언 가능
    - 선언 시 반드시 값을 지정해줘야 한다.
    
    ![Untitled](/images/lang_javascript/study/JavaScript_상수/Untitled%201.png)
    

### <u>구문오류 : Assignment to constant variable</u>

- 한 번 선언된 상수의 자료는 변경 불가
    
    ![Untitled](/images/lang_javascript/study/JavaScript_상수/Untitled%202.png)
    

- 해결 방법
    - 상수가 아니라 변수를 사용해야 한다.

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판