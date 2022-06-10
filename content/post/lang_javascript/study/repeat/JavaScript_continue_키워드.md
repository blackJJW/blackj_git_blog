---
title: "[JavaScript] continue 키워드"
description: ""
date: "2022-05-26T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **반복문 안의 반복 작업을 멈추고 반복문의 처음으로 돌아가 다음 반복 작업을 진행**
- continue 키워드 활용(1)
    
    ```jsx
    <script>
    	for (let i  = 0; i < 5; i++) {
    		continue
    		alert(i)
    	}
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_continue_키워드/Untitled.png)
    
    - 코드를 실행하면 경고창이 출력되지 않음


- continue 키워드 활용(2)
    
    ```jsx
    <script>
    	let output = 0
    
    	for (let i = 1; i <= 10; i++) {
    		if (i % 2 ===1) { // 홀수면 현재 반복을 중지하고 다음 반복을 수행
    			continue
    		}
    		output += i
    	}
    	alert(output)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_continue_키워드/Untitled%201.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판