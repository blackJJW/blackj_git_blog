---
title: "[JavaScript] 불 입력"
description: ""
date: "2022-05-08T20:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 문자열 외에 불 자료형도 입력 가능
- **confirm() 함수** 사용

  ```jsx
  confirm(메시지 문자열)
  ```

- confirm() 함수의 사용

  ```jsx
  <script>
      const input = confirm('확인 or 취소')
      alert(input)
  </script>
  ```
  
  ![Untitled](/images/lang_javascript/study/JavaScript_불_입력/Untitled.png)

- [확인] 클릭 시, **true 리턴**
    
    ![Untitled](/images/lang_javascript/study/JavaScript_불_입력/Untitled%201.png)
    

- [취소] 클릭 시, **false 리턴**
    
    ![Untitled](/images/lang_javascript/study/JavaScript_불_입력/Untitled%202.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판