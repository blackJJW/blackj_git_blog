---
title: "[JavaScript] while 반복문"
description: ""
date: "2022-05-24T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- if 조건문과 형태가 비슷
- if 조건문과 다른 점은 문장을 한 번만 실행하고 끝나는 것이 아니라 **불 표현식이 true면 계속해서 문장을 실행**
    
    ```jsx
    while (불 표현식) {
    	문장
    }
    ```
    
- 조건이 변하지 않는다면 무한히 반복 실행하므로 조건을 거짓으로 만들 수 있는 내용이 문장에 포함되어 있어야 한다.
- **무한 루프**(**infinite loop**) : 반복문이 무한히 반복되는 것
- 무한 반복문
    
    ```jsx
    <script>
    	let i = 0
    	while (true) {
    		alert(`${i}`)
    		i = i + 1
    	}
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_while_반복문/Untitled.png)
    
    - 코드 실행 시 [확인] 버튼을 클릭할 때마다 경고 창이 계속 뜬다.
        - 웹 브라우저를 종료할 때까지 계속 반복됨
- while 반복문 기본
    
    ```jsx
    <script>
    	let i = 0
    	while (confirm('계속 진행?')) {
    		// 사용자가 [확인] 버튼을 클릭하면 true가 됨
    		alert(`${i} 반복`)
    		i = i + 1
    	}
    </script>
    ```
    
    - **confirm() 함수**를 입력하면 사용자에게 확인을 받는 대화상자가 실행
        - [확인] 버튼을 클릭하면 **true가 되어 반복문이 계속 실행**
        - [취소] 버튼을 클릭하면 **false가 되어 반복문 종료**
    
    ![Untitled](/images/lang_javascript/study/JavaScript_while_반복문/Untitled%201.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_while_반복문/Untitled%202.png)
    

### while 반복문과 함께 배열 사용

- 배열과 함께 사용
    
    ```jsx
    <script>
    	let i = 0
    	const array = [1, 2, 3, 4, 5]
    	
    	while (i < array.length) {
    		console.log(`${i} : ${array[i]}`)
    		i++
    	}
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_while_반복문/Untitled%203.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판