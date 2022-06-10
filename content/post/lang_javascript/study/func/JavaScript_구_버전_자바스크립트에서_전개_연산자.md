---
title: "[JavaScript] 구 버전 자바스크립트에서 전개 연산자 구현"
description: ""
date: "2022-05-30T17:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- **전개 연산자**는 최신 버전의 자바스크립트에 추가된 기능
- 구 버전에서는 **apply() 함수** 사용
- 전개 연산자가 없던 구 버전에서 apply() 함수 사용
    
    ```jsx
    <script>
    	// 단순히 매개변수를 모두 출력
    	function sample(...items) {
    		console.log(items)
    	}
    
    	// 전개 연산자 사용 여부 비교
    	const array = [1, 2, 3, 4]
    	console.log(sample.apply(null, array))
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_구_버전_자바스크립트에서_전개_연산자/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판