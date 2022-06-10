---
title: "[JavaScript] 나머지 매개변수"
description: ""
date: "2022-05-28T16:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **가변 매개변수 함수** : 호출할 때 배개변수의 개수가 고정적이지 않은 함수
- **나머지 매개변수**(**rest parameter**)
    - 자바스크립트에서 **가변 매개변수 함수를 구현하기 위해 사용**
    
    ```jsx
    function 함수이름(...나머지 매개변수) { }
    ```
    
    - 함수의 매개변수 앞에 **마침표 3개(…)를 입력**하면 매개변수들이 **배열**로 들어온다.

### 나머지 매개변수를 사용한 배열 만들기

```jsx
<script>
	function sample(...items) {
		console.log(items)
	}
	
	sample(1, 2)
	sample(1, 2, 3)
	sample(1, 2, 3, 4)
</script>
```

![Untitled](/images/lang_javascript/study/JavaScript_나머지_매개변수/Untitled.png)

- 코드를 실행시, 매개변수가 배열의 형태로들어온 것을 확인

![Untitled](/images/lang_javascript/study/JavaScript_나머지_매개변수/Untitled%201.png)

- 나머지 매개변수를 사용한 min() 함수
    
    ```jsx
    <script>
    	//나머지 매개변수를 사용한 함수 생성
    	function min(...items) {
    		//매개변수items는 배열로 사용 가능
    		let output = items[0]
    		for (const item of itmes) {
    			if (output > item) {
    				output = item
    			}
    		}
    		return output
    	}
    	console.log('min(52, 273, 32, 103, 275, 24, 57)')
    	console.log(`=${min(52, 273, 32, 103, 275, 24, 57)}`)	
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_나머지_매개변수/Untitled%202.png)
    

### 나머지 매개변수와 일반 매개변수 조합

- 나머지 매개변수는 이름 그래도 **나머지**
- 다음과 같이 **일반적인 매개변수와 조합**해서 사용
    
    ```jsx
    function 함수이름(매개변수, 매개변수, ...나머지 매개변수) { }
    ```
    
- 나머지 매개변수와 일반 매개변수를 갖는 함수
    
    ```jsx
    <script>
    	function sample(a, b, ...c) {
    		console.log(a, b, c)
    	]
    	
    	sample(1, 2)
    	sample(1, 2, 3)
    	sample(1, 2, 3, 4)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_나머지_매개변수/Untitled%203.png)
    
    - 실행 후, 매개변수 a, b가 먼저 들어가고, 남은 것들은 모두 c에 배열 형태로 들어간다.

### 전개 연산자

- 다음과 같이 매개변수로 **배열을 입력할 수 없고 숫자를 입력해야 하는 함수**가 존재할 경우
    
    ```jsx
    min(52, 273, 32, 103)
    min(52, 273, 32, 103, 103, 275, 24, 57)
    ```
    
- 이때 **배열 자료형**으로 읽어들였다면 위와 같은 형태의 min() 함수를 어떻게 호출하는 가?
    
    ```jsx
    const array = [1, 2, 3, 4]
    ```
    
    - 기본적으로 배열 요소를 하나하나 전개해서 입력하는 방법이 존재
        
        ```jsx
        min(array[0], array[1], array[2], array[3])
        ```
        
    - 이런 상황에 자바스크립트는 **배열을 전개해서 함수의 매개변수로 전달**해주는 **전개 연산자**(**spread operator**) 제공
    - 배열 앞에 마침표 3개(…)를 붙이는 형태로 사용
        
        ```jsx
        함수이름(...배열)
        ```
        
- 전개 연산자 활용
    
    ```jsx
    <script>
    	// 단순하게 매개변수를 모두 출력
    	function sample(...items) {
    		console.log(items)
    	}
    
    	//전개 연산자 사용 여부 비교
    	const array = [1, 2, 3, 4]
    
    	console.log('# 전개 연산자를 사용하지 않은 경우')
    	sample(array)
    	console.log('# 전개 연산자를 사용한 경우')
    	sample(...array)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_나머지_매개변수/Untitled%204.png)
    
    - 전개 연산자를 사용하지 않은 경우에는 **배열이 매개변수**로 들어온다.
    - 전개 연산자를 사용한 경우에는 **숫자 하나하나 전개되어 매개변수**로 들어온다.

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판