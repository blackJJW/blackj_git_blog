---
title: "[JavaScript] 이벤트 모델"
description: ""
date: "2022-06-11T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- **이벤트 모델(event model)** : **이벤트를 연결**하는 방법
- **표준 이벤트 모델** : **addEventListener() 메소드**를 사용하는 표준 방법
    
    ```jsx
    document.body.addEventListener('keyup', () => {
    
    })
    ```
    

- **고전 이벤트 모델** : **`on` 으로 시작하는 속성에 함수를 할당**해서 이벤트를 연결하는 방법
    
    ```jsx
    document.body.onkeyuo = (event) => {
    							// 속성
    }
    ```
    

- **인라인 이벤트 모델** : 고전 이벤트 모델처럼  **`on` 으로 시작하는 속성을 HTML 요소에 직접 넣어서 이벤트를 연결**하는 방법
    
    ```jsx
    <script>
    	const listener = (event) => {
    
    	}
    </script>
    <body onkeyup="listener(event)">
    
    </body>
    ```
    
- 모든 이벤트 모델의 이벤트 리스너는 **첫 번째 매개변수**로 **이벤트 객체**(**event object**)를 받음
    - **이벤트 객체** : 이벤트와 관련된 정보가 들어있음
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판