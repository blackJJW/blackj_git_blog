---
title: "[JavaScript] 고급 예외 처리"
description: ""
date: "2022-06-17T13:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **고급 예외 처리** : **try catch finally 구문**으로 예외를 처리하는 방법
- 기본적인 형태
    
    ```jsx
    try {
    	// 예외가 발생할 가능성이 있는 코드
    } catch (exception) {
    	// 예외가 발생했을 때 실행할 코드
    } finally { 
    	// 무조건 실행할 코드
    	// 필요한 경우에만 사용 
    }
    ```
    
    - **try 구문** 안에서 예외가 발생하면 이를 **catch 구문**에서 처리
    - **finally 구문**은 **필수 사항은 아니며** 예외 발생 여부와 상관 없이 수행해야 하는 작업이 있을 때 사용

- try catch 구문의 사용
    
    ```jsx
    <script>
    	try {
    		willExcept.byeBye();
    		console.log("try 구문 마지막");
    	} catch (exception) {
    		console.log("catch 구문 마지막")
    	}
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_고급_예외_처리/Untitled.png)
    
- finally 구문
    
    ```jsx
    <script>
    	try {
    		willExcept.byeBye();
    		console.log("try 구문 마지막");
    	} catch (exception) {
    		console.log("catch 구문 마지막")
    	} finally {
    		console.log("finally 구문 마지막")
    		// 예외의 발생 여부와 상관 없이 무조건 실행
    	}
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_고급_예외_처리/Untitled%201.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판