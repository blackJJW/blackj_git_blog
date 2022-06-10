---
title: "[JavaScript] 익명 함수와 선언적 함수의 차이"
description: ""
date: "2022-06-02T18:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

### 익명 함수의 사용

- **익명 함수** : 순차적인 코드 실행에서 코드가 해당 줄을 읽을 때 생성
- 익명 함수 호출
    
    ```jsx
    <script>
    	// 변수를 선언
    	let a
    
    	// 익명 함수를 2번 생성
    	a = function () {
    		console.log('1번')
    	}
    	a = function () {
    		console.log('2번')
    	}
    
    	a()
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_익명_함수와_선언적_함수의_차이/Untitled.png)
    
    - 코드가 위에서 아래로 실행되면서 a 변수에 두번째 함수가 할당되었다.

### 선언적 함수의 사용

- **선언적 함수** : 순차적인 코드 실행이 일어나기 전에 생성
- 같은 블록이라면 **어디에서 함수를 호출해도 상관없음**
    - 선언적 함수를 생성하기 전에 함수를 호출해도 이미 생성된 상태이므로 문제 없이 실행됨
- 선언적 함수 호출
    
    ```jsx
    <script>
    	// 선언적 함수 호출
    	a()
    
    	// 선언적 함수를 2번 생성
    	function a () {
    		console.log('1번')
    	}
    	function a () {
    		console.log('2번')
    	}
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_익명_함수와_선언적_함수의_차이/Untitled%201.png)
    

### 선언적 함수와 익명 함수의 조합

- 2가지 상황이 조합된 경우
    - **선언적 함수는 먼저 생성**
    - 이후 순차적인 코드 진행을 하면서 **익명 함수 생성**
- 선언적 함수와 익명 함수의 조합
    
    ```jsx
    <script>
    	a = function() {
    		console.log('익명 함수')
    	}
    
    	function a () {
    		console.log('선언적 함수')
    	}
    
    	a()
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_익명_함수와_선언적_함수의_차이/Untitled%202.png)
    

- **익명 함수**는 코드를 읽을 때와 **같은 순서로 함수가 선언**되지만, **선언적 함수**는 코드를 읽는 순서와 **다른 순서로 함수가 선언**
- 함수를 같은 이름으로 덮어쓰는 것은 굉장히 **위험한 일**
- 안전하게 사용할 수 있는 **익명 함수를 더 선호**

### 블록이 다른 경우에 선언적 함수의 사용

- 선언적 함수는 어떤 코드 블록을 읽어들일 때 먼저 생성
- 블록이 다른 경우 선언적 함수의 사용
    
    ```jsx
    <script>
    	a()
    
    	function a (){
    		console.log('1번 선언적 함수')
    	}
    </script>
    <script>
    	function a () {
    		console.log('2번 선언적 함수')
    	}
    </script>
    <script>
    	a()
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_익명_함수와_선언적_함수의_차이/Untitled%203.png)
    

- 다른 프로그래밍 언어들은 일반적으로 선언적 함수 형태를 많이 사용
- 자바스크립트에서는 **블록이 예상하지 못하게 나뉘는 문제** 등이 발생할 수 있음
    - 안전을 위해 **익명 함수를 더 많이 사용**

- 과거 자바스크립트는 **var 키워드를 사용하여 번수를 선언**
    - var 키워드는 덮어쓰는 문제가 발생
- 현대의 자바스크립트는 **let 키워드**와 **const 키워드**를 사용하여 변수와 상수를 선언
    - **위험을 원천적으로 차단하기 위해 오류를 발생시킴**
- let 사용의 의미
    
    ```jsx
    <script>
    	// 익명 함수를 2번 생성
    	let a = function () {
    		console.log('익명 함수')
    	}
    	// 선언적 함수를 2번 생성하고 할당
    	function a() {
    		console.log('선언적 함수')
    	}
    	// 함수 호출
    	a()
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_익명_함수와_선언적_함수의_차이/Untitled%204.png)
    

- **한 가지로 통일해서 사용**하는 것이 오류의 위험을 줄일 수 있다.
- 통일한다면 **익명 함수로 통일해서 사용하는 것**이 안전을 위한 더 편한 선택

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판