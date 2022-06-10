---
title: "[JavaScript] break 키워드"
description: ""
date: "2022-05-25T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 조건문이나 반복문을 **벗어날 때 사용하는 키워드**
    
    ```jsx
    while (true) {
    
    } break
    ```
    
- break 키워드 활용
    
    ```jsx
    <script>
    	// 반복문
    	for (let i = 0; true; i++) {
    		alert(`${i} : 반복`)
    	
    		//진행여부
    		const isContinue = confirm('계속?')
    		if (!isContinue) {
    			break
    		}
    	}
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_break_키워드/Untitled.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_break_키워드/Untitled%201.png)
    
    - [확인] 버튼을 클릭하면 true가 되어 반복 계속
    - [취소] 버튼능 클릭하면 false가 되어 실행 종료
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판