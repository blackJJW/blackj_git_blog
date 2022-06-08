---
title: "[JavaScript] switch 조건문"
description: ""
date: "2022-05-14T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- **switch 조건문**의 기본 형태
    - **default 키워드는 생략 가능**
    
    ```jsx
    switch (자료) {
    	case 조건1:
    		break
    	case 조건2:
    		break
    	default:
    		break
    }
    ```
    
- switch 조건문 사용
    
    ```jsx
    <script>
    	const input = Number(prompt('숫자를 입력.', '숫자'))
            
    	switch (input % 2) {
    			case 0:
    				alert('짝수')
    				break
    			case 1:
    				alert('홀수')
    				break
    			default:
    				alert('숫자 아님')
    				break
      }
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_switch_조건문/Untitled.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_switch_조건문/Untitled%201.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_switch_조건문/Untitled%202.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_switch_조건문/Untitled%203.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_switch_조건문/Untitled%204.png)
    

- break 키워드 : switch 조건문이나 반복문을 빠져나가시 위해 사요하는 키워드
    - 코드를 읽다가 break 키워드를 만나면 감싼 switch 조건문이나 반복문을 완전히 빠져나감

### switch 조건문을 if 조건문으로 변환

- 모든 switch 조건문은 if 조건문으로 바꾸는 것이 가능
    
    ```jsx
    <script>
      const date = new Date() // 현재 날짜와 시간을 갖는 객체 생성
      const hour = date.getHours() // 현재시간을 0 ~ 23 사이의 값으로 출력하는 메소드
    	
      switch (true) {
        case hour < 11: 
          alert('아침 식사 시간')
    	  break
        case hour < 15:
          alert('점심 식사 시간')
          break 
        default:
          alert('저녁 식사 시간')
    	  break 
    	}
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_switch_조건문/Untitled%205.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판