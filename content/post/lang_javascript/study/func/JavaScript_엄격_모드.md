---
title: "[JavaScript] 엄격 모드"
description: ""
date: "2022-06-02T17:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 블록의 가장 위쪽에 ‘**use strict**’라는 문자열이 등장하는 경우가 있다.
- **엄격 모드**(**strict mode**)라고 부르는 기능으로 이러한 문자열을 읽어들이 순간부터 **코드를 엄격하게 검사**
    
    ```jsx
    <script>
    	'ues strict'
    	문장
    	문장
    </script>
    ```
    

- 선언 없이 변수 사용
    - data라는 변수를 let 키워드 등으로 선언하지 않고 곧바로 사용
    - 일반적인  자바스크립트 코드에서는 문제없이 실행
    
    ```jsx
    <script>
    	data = 10
    	console.log(data)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_엄격_모드/Untitled.png)
    
- 엄격 모드에서 선언 없이 변수 사용
    - let 키워드 등을 사용하지 않아 오류 발생
    
    ```jsx
    <script>
    	'use strict'
    	data = 10
    	console.log(data)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_엄격_모드/Untitled%201.png)
    
- 일반적으로 엄격 모드를 사용하는 것이 좋다
- 엄격 모드에서 발생하는 오류에 대해서는 **모질라 엄격 모드 문서**를 참조
    
    [모질라 엄격 모드 문서](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Strict_mode)
    
    - **엄격의 기준은 시간에 따라 변하므로 주의가 필요**

- 즉시 호출 함수를 만들고, 이 블록의 **가장 위쪽에서 엄격 모드를 적용**하는 경우가 많음
    - 해당 **블록 내부에만 엄격 모드**가 적용
    - 사용 패턴
        
        ```jsx
        <script>
        	(function () {
        		'use strict'
        		문장
        		문장
        	})()
        </script>
        ```
        

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판