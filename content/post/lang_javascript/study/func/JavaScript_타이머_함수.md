---
title: "[JavaScript] 타이머 함수"
description: ""
date: "2022-06-01T16:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **타이머(Timer) 함수** : **특정 시간**마다 또는 **특정 시간 이후**에 콜백 함수를 호출할 수 있는 함수
    - 시간과 관련된 처리 가능
    
    | 함수 이름 |          설명          |
    |:--------------------:| :--- |
    | setTimeout(함수, 시간) | 특정 시간 후에 함수를 한 번 호출  |
    | setInterval(함수, 시간) |    특정 시간마다 함수를 호출    |
- 타이머 걸기
    
    ```jsx
    <script>
    	setTimerout(() => {
    		console.log('1초 후에 실행')
    	}, 1 * 1000)
    
    	let count = 0
    	setInterval(() => {
    		console.log(`1초마다 실행 ${count}번`)
    		count++
    	}, 1 * 1000)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_타이머_함수/Untitled.png)
    
    - **setTimeout() 함수**와 **setInterval() 함수**를 사용해서 **특정 시간 후에 코드를 호출**

- 타이머를 종료하고 싶을 때는 **clearTimeout() 함수**와 **clearInterval() 함수**를 사용
    
    
    | 함수 이름 | 설명 |
    | :---: | --- |
    | clearTimeout(타이머_ID) | setTimeout() 함수로 설정한 타이머를 제거 |
    | clearInterval(타이머_ID) | setInterval() 함수로 설정한 타이머를 제거 |
    - **타이머 ID** : setTimeout() 함수와 setInterval() 함수를 호출할 때 리턴값으로 나오는 숫자

- 타이머 취소
    
    ```jsx
    <script>
    	let id
    	let count = 0
    	id = setInterval(() => {
    		console.log(`1초마다 실행(${count}번)`)
    		count++
    	}, 1 * 1000)
    
    	setTimeout(() => {
    		console.log(`타이머를 종료`)
    		clearInterval(id)
    	}, 5 * 1000)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_타이머_함수/Untitled%201.png)
    
    - setInteval() 함수를 사용해 1초마다 메세지를 출력
    - setTimeout() 함수를 사용해 5초 후 타이머 종료

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판