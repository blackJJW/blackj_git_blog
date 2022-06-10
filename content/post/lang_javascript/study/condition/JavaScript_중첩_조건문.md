---
title: "[JavaScript] 중첩 조건문"
description: ""
date: "2022-05-12T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 조건문 안에 **조건문을 중첩**해 사용하는 것
- **여러 번 중첩하는 것도 가능**
    
    ```jsx
    if (불 값이 나오는 표현식 1) {
    	if(불 값이 나오는 표현식 2) { // 표현식 1이 참일 때
    		표현식 2가 참일 때
    	} else{
    		표현식 2가 거짓일 때
    	}
    } else{
    	if (불 값이 나오는 표현식 3) { // 표현식 1이 거짓일 때
    		표현식 3이 참일 때
    	} else{
    		표현식 3이 거깃일 때
    	}
    }
    ```
    
- 중첩 조건문으로 시간 파악
    
    ```jsx
    <script>
    	const date = new Date() // 현재 날짜와 시간을 갖는 객체 생성
      const hour = date.getHours() // 현재시간을 0 ~ 23 사이의 값으로 출력하는 메소드
    
      if (hour < 11) {
    	  alert('아침 식사 시간')
      } else {
    	  if (hour < 15){
    	  alert('점심 식사 시간')
    	  } else {
    	  alert('저녁 식사 시간')
        }
      }
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_중첩_조건문/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판