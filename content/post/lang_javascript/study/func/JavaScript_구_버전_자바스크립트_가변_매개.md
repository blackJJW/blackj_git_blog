---
title: "[JavaScript] 구 버전 자바스크립트에서 가변 매개변수 함수 구현"
description: ""
date: "2022-05-30T16:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 구 버전의 자바스크립트에서 가변 매개변수 함수를 구현할 때는 **배열 내부에서 사용할 수 있는 특수한 변수인 arguments**를 활용
- arguments는 **매개변수와 관련된 여러 정보를 확인**할 수 있고 **배열과 비슷하게 사용** 가능
- arguments를 사용한 가변 매개변수 함수
    
    ```jsx
    <script>
    	function sample() {
    		console.log(arguments)
    		for (let i = 0; i < arguments.length; i++) {
    			console.log(`${i} : ${arguments[i]}`)
    		}
    	}
    
    	sample(1, 2)
    	sample(1, 2, 3)
    	sample(1, 2, 3, 4)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_구_버전_자바스크립트에서_가변_매개/Untitled.png)
    
    - 실행 결과
        - arguments 내부에 **callee: f**와 **Symbol(Symbol.iterator): f** 존재
            - 배열과 비슷하지만 **배열은 아님**
            - 일반적인 배열처럼 사용하면 **여러 위험을 내포**
            - 따라서 가변 매개변수 함수를 더 편리하게 만들 수 있는 **나머지 매개변수가 최신 버전의 자바스크립트에 포함**됨

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판