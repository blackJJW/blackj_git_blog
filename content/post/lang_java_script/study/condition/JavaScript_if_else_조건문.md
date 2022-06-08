---
title: "[JavaScript] if else 조건문"
description: ""
date: "2022-05-11T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- **else 구문** : 좀 더 편리하게 조건문을 사용할 수 있도록 서로 반대되는 상황을 표현하는 구문
- else 구문은 if 조건문 바로 뒤에 붙여서 사용
    
    ```jsx
    if (불 값이 나오는 표현식){
    	불 값이 참일 때
    } else {
    	불 값이 거짓일 때
    }
    ```
    
- if else 조건물을 사용해 현재 시간 구하기
    
    ```jsx
    <script>
    	const date = new Date() // 현재 날짜와 시간을 갖는 객체 생성
      const hour = date.getHours() // 현재시간을 0 ~ 23 사이의 값으로 출력하는 메소드
    
      if (hour < 12) {
    	  alert('오전')
      } else {
    	  alert('오후')
      }
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_if_else_조건문/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판