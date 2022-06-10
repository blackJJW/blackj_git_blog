---
title: "[JavaScript] 매개변수와 리턴값"
description: ""
date: "2022-05-27T16:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **매개변수** : 함수를 호출할 때 괄호 안에 적는 것
- **리턴값** : 함수의 최종 결과

### 사용자 정의 함수의 매개변수와 리턴값

```jsx
function 함수(매개변수, 매개변수, 매개변수) {
	문장
	문장
	return 리턴값
}
```

- **기본 형태의 함수** 만들기
    
    ```jsx
    <script>
    	function f(x) {
    		return x * x
    	}
    	
    	console.log(f(5))
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_매개변수와_리턴값/Untitled.png)
    
    - 매개변수와 리턴값
        
        ![Untitled](/images/lang_javascript/study/JavaScript_매개변수와_리턴값/Untitled%201.png)
        
        - 코드 작성시 **보조 기능이 동작**하는 것을 확인

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판