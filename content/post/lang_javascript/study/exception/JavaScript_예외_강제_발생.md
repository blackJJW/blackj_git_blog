---
title: "[JavaScript] 예외 강제 발생"
description: ""
date: "2022-06-17T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 상황에 따라서 예외를 강제로 발생시켜야 하는 경우도 존재
- 강제로 발생시킬 때는 **throw 키워드** 사용
    
    ```jsx
    // 단순하게 예외를 발생
    throw 문자열
    
    // 조금 더 자세하게 예외를 발생
    throw new Error(문자열)
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_예외_강제_발생/Untitled.png)
    

- 예외 강제롤 발생시키고 잡기
    
    ```jsx
    <script>
        function divide(a, b) {
            if (b === 0) {
                throw '0으로 나눌 수 없다.'
            }
            return a / b
        }
    
        console.log(divide(10, 2))
        console.log(divide(10, 0))
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_예외_강제_발생/Untitled%201.png)

    
- 예외를 강제로 발생시키기
    
    ```jsx
    <script>
        function test(object){
            console.log(object.a + object.b)
        }
    
        test({})
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_예외_강제_발생/Untitled%202.png)
 
    - 일반적인 프로그래밍 언어의 경우
        - object 객체에 a 속성과 b 속성이 없으므로 예외를 발생
        - 존재하지 않는 것을 더하므로 object.a + object.b를 할 때도 예외를 발생
    - 자바스크립트의 경우
        - object.a 가 undefined로, object.b 도 undefined로 출력
        - **undefined + undefined 를 하면 NaN이 출력**
        - 아무 오류없이 코드가 정상적으로 실행된다.


- 자바스크립트는 **NaN**과 **undefined**라는 값이 있어서 다른 프로그래밍 언어에 비해서 예외를 많이 발생하지 않음
- 사용자에게 함수를 잘못 사용했다는 것을 **강제로라도 인지시켜줄 필요**가 있다.
- 예외를 강제로 발생시켜 사용 유도
    
    ```jsx
    <script>
        function test(object) {
            if (object.a !== undefined && object.b !== undefined) {
                console.log(object.a + object.b)
            } else {
                throw new Error("a 속성과 b 속성을 지정하지 않았음.")
            }
        }
    
        test({})
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_예외_강제_발생/Untitled%203.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판