---
title: "[JavaScript] if 조건문"
description: ""
date: "2022-05-10T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 가장 일반적인 조건문은 **if 조건문**
- 불 표현식의 값이 **true면 중괄호 안의 문장을 실행**, **false면 문장을 무시**
    
    ```jsx
    if(불 값이 나오는 표현식){
    	불 값이 참일 때 실행
    }
    ```
    
- if 조건문 사용하기
    
    ```jsx
    <script>
    	if (10 < 15) { // if 조건문
    		alert("10 < 15 -> True")
    	}
    
    	alert("종료")
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_if_조건문/Untitled.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_if_조건문/Untitled%201.png)
    
    ```jsx
    <script>
    	if (10 > 15) { // if 조건문
    		alert("10 > 15 -> True")
    	}
    
    	alert("종료")
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_if_조건문/Untitled%201.png)
    
- 오전과 오후 구분
    
    ```jsx
    <script>
    	const date = new Date() // 현재 날짜와 시간을 갖는 객체 생성
      const hour = date.getHours() // 현재시간을 0 ~ 23 사이의 값으로 출력하는 메소드
    
      if (hour < 12) {
          alert('오전')
      }
    
      if (hour >= 12) {
          alert('오후')
      }
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_if_조건문/Untitled%202.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판