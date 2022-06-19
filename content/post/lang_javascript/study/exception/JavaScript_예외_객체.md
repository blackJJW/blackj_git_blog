---
title: "[JavaScript] 예외 객체"
description: ""
date: "2022-06-17T14:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **예외 객체**(**exception object**) : **try catch 구문을 사용**할 때 **catch의 괄호 안에 입력**하는 **식별자**
- 아무 식별자나 입력해도 괜찮지만, 일반적으로 **e**나 **exception**이라는 식별자를 사용
    
    ```jsx
    try {
    
    } catch {exception} {
    				// 예외 객체
    }
    ```
    

- **예외 객체의 속성**
    
    
    | 속성 이름 | 설명 |
    | --- | --- |
    | name  | 예외 이름 |
    | message | 예외 메시지 |

- 예외 정보 출력
    
    ```jsx
    <script>
    	try {
    		const array = new Array(9999999999999999)
    	} catch (e){
    		console.log(e)
    		console.log()
    		console.log(`예외 이름 : ${e.name}`)
    		console.log(`예외 메시지 : ${e.message}`)	
    	}
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_예외_객체/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판