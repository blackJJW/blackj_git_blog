---
title: "[JavaScript] 즉시 호출 함수 문제 해결"
description: ""
date: "2022-06-02T16:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 블록을 사용하는 방법과 함수 블록을 사용해 변수 충돌을 막는 방법 모두 **최신 자바스크립트**를 지원하는 웹 브라우저에서는 사용가능
- **구 버전의 자바스크립트**에서 변수를 선언할 때 사용하던 **var 키워드**는 **함수 블록을 사용하는 경우에만 변수 충돌을 막을 수 있음**

- 많은 개발자들이 함수 블록을 사용해 문제를 해결
- 충돌 문제를 해결하기 위해 사용하는 것이므로 **함수를 만들자마자 즉시 호출할 수 있도록** 다음과 같이 작성
- 즉시 호출 함수를 사용한 문제 해결
    
    ```jsx
    <script>
            let pi = 3.14
            console.log(`파이 : ${pi}`)
    </script>
    <script>
            (function () {
                let pi = 3.1415
                console.log(`파이 : ${pi}`)
            })()
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_즉시_호출_함수_문제_해결/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판