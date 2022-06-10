---
title: "[JavaScript] 즉시 호출 함수"
description: ""
date: "2022-06-02T16:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 여러 웹사이트에서 다음과 같이 **익명 함수를 생성**하고 곹바로 **즉시 호출**하는 패턴을 사용
    - **함수 즉시 호출**
        
        ```jsx
        (function () {})()
        ```
        

- 일반적으로 자바스크립트는HTML 페이지 내부에서 사용할 때 **script 태그**를 **여러 개 사용**하고 코드를 입력한다.
    - 코드가 여러 곳에서 사용되면 **변수 이름이 충돌**할 가능성이 높음
        
        ```jsx
        <script>
                let pi = 3.14
                console.log(`파이 : ${pi}`)
        </script>
        <script>
                let pi = 3.141592
                console.log(`파이 : ${pi}`)
        </script>
        ```
        
        ![Untitled](/images/lang_javascript/study/JavaScript_즉시_호출_함수/Untitled.png)
        
    - 식별자가 이미 사용되고 있다는 오류를 발생
        - 아래의 script부분이 실행되지 않음
- **스코프**(**scope**) : 변수가 **존재하는 범위**
    - **같은 단계**이 있을 경우 무조건 충돌이 발생
    - 이러한 **스코프 단계를 변경**하는 방법
        - **중괄호를 사용**해서 **블록을 생성**하는 방법
        - **함수를 생성**해서 **블록을 만드는 방법**
        
        ```jsx
        <script>
                let pi = 3.14
                console.log(`파이 : ${pi}`)
        
                // 블록을 사용한 스코프 생성
                {
                    let pi = 3.1415
                    console.log(`파이 : ${pi}`)
                }
                console.log(`파이 : ${pi}`)
        
                // 함수 블록을 사용한 스코프 생성
                function sample() {
                    let pi = 3.121592
                    console.log(`파이 : ${pi}`)
                }
                sample()
                console.log(`파이 : ${pi}`)
        </script>
        ```
        
        ![Untitled](/images/lang_javascript/study/JavaScript_즉시_호출_함수/Untitled%201.png)
        
        - 블록 내부에서 같은 이름으로 변수를 선언하면 변수가 외부 변수와 충돌하지 않고 외부 변수를 가림
        - **섀도잉**(**shadowing**) : 블록이 다른 경우 내부 변수가 외부 변수를 가리는 현상
        

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판