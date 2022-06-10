---
title: "[JavaScript] if else if 조건문"
description: ""
date: "2022-05-13T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- if 조건문은 **조건이 한 문장이라면 중괄호 생략 가능**하다는 점을 이용
- 겹치지 않는 **3가지 이상의 조건**으로 나눌 때 사용
    
    ```jsx
    if (불 표현식) {
    	문장
    } else if (불 표현식) {
    	문장
    } else if (불 표현식) {
    	문장
    } else {
    	문장
    }
    ```
    
- if else if 조건문으로 시간 파악
    
    ```jsx
    <script>
    	const date = new Date() // 현재 날짜와 시간을 갖는 객체 생성
      const hour = date.getHours() // 현재시간을 0 ~ 23 사이의 값으로 출력하는 메소드
    
      if (hour < 11) {
    	  alert('아침 식사 시간')
      } else if (hour < 15){
    	  alert('점심 식사 시간')
    	} else {
    	  alert('저녁 식사 시간')
      }
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_if_else_if_조건문/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판