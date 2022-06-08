---
title: "[JavaScript] 문자열 입력"
description: ""
date: "2022-05-08T19:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

### <u>문자열 입력</u>

- 문자열 자료형 입력시 사용하는 함수는 **prompt()**

   ```jsx
   prompt(메시지 문자열, 기본 입력 문자열)
   ```

- prompt() 함수 **매개변수**의 역할

   ```jsx
   <script>
       const input = prompt('message', '_default')
       alert(input)
   </script>
   ```

1. 코드 실행 시, 사용자에게 입력을 요구하는 창이 나타남
    1. prompt() 함수의 매개변수들이 어디에 출력되는 지 확인
    
    ![Untitled](/images/lang_javascript/study/JavaScript_문자열_입력/Untitled.png)
    
2. 입력 창에 문자열을 입력하고, [확인] 버튼을 클릭
    
    ![Untitled](/images/lang_javascript/study/JavaScript_문자열_입력/Untitled%201.png)
    
3. 입력한 문자열일 경고창에 출력
    
    ![Untitled](/images/lang_javascript/study/JavaScript_문자열_입력/Untitled%202.png)
    
- 사용자가 입력창에 값을 입력하고 [확인] 버튼을 클릭하면 prompt() 함수는 **입력한 문자열을 input에 저장**
- 함수를 실행한 후 값을 남기는 것을 **리턴**(**return**)

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판