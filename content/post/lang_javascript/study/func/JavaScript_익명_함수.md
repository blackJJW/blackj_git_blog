---
title: "[JavaScript] 익명 함수"
description: ""
date: "2022-05-26T16:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **함수** : 코드의 집합를 나타내는 자료형
    
    ```jsx
    function () { }
    ```
    
    - 코드의 집합이라고 말하는 이유는 **중괄호** {…} 내부에 코드를 넣기 때문
    - **함수 호출** : 함수를 사용하는 것
    - **매개변수** : 함수를 호출할 때는 괄호 내부에 넣는 여러 가지 자료
    - **리턴값** : 함수를 호출해서 최종적으로 나오는 결과
- 함수 사용의 장점
    - 반복되는 코드를 한 번만 정의하고 필요할 때마다 호출하므로 **반복 작업을 피할 수 있음**
    - 긴 프로그램을 기능별로 나눠 여러 함수로 나누어 작성하면 **모듈화로 전체 코드의 가독성이 좋아진다.**
    - 기능별로 수정이 가능하므로 **유지보수가 쉬움**
- 익명 함수 선언
    
    ```jsx
    <script>
    	const func = function () { // 변수를 생성
    		console.log('함수 내부 코드 - 1')
    		console.log('함수 내부 코드 - 2')
    		console.log('함수 내부 코드 - 3')
    		console.log(' ')
    	}
    
    	func() // 함수 호출
    	func()
    
    	console.log(typeof func)
    	console.log(func)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_익명_함수//Untitled.png)
    
- 함수의 자료형은 **function**
    - 코드에서 출력하면 **f ( ) { }** 라고 출력
    - 함수를 출력했을 때 **이름이 붙어있지 않은 것**을 확인
        - **익명 함수**(**anonymous function**) : 이름이 붙어있지 않은 함수
    
    ![Untitled](/images/lang_javascript/study/JavaScript_익명_함수//Untitled%201.png)
    
    - 기존 함수들로 코드를 실행하면 함수의 이름이 출력되는 것을 확인

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판