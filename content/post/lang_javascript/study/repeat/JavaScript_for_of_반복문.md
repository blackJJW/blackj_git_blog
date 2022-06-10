---
title: "[JavaScript] for of 반복문"
description: ""
date: "2022-05-22T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 요소의 값을 **안정적으로 사용 가능**
- **for of 반복문 기본 형태**
    
    ```jsx
    for (const 반복 변수 of 배열 또는 객체) {
    	문장
    }
    ```
    
    - for in 반복문과는 다르게 **반복 변수에 요소의 값이 들어간다.**
- **for of 코드블록** 사용
    - 코드 작성 시 **for** 입력
    - **for 와 관련된 여러 블록**들이 나옴
    
      ![https://blackjjw.github.io/images/lang_javascript/study/JavaScript_for_in_%EB%B0%98%EB%B3%B5%EB%AC%B8/Untitled%201.png](https://blackjjw.github.io/images/lang_javascript/study/JavaScript_for_in_%EB%B0%98%EB%B3%B5%EB%AC%B8/Untitled%201.png)
    
    - **forof 코드 블록**으로 이동 후 `Enter` 또는 `Tab` 키를 누른다.
    
      ![Untitled](/images/lang_javascript/study/JavaScript_for_of_반복문/Untitled.png)
    
      ![Untitled](/images/lang_javascript/study/JavaScript_for_of_반복문/Untitled%201.png)
    
      ```jsx
      for (const iterator of object) {
                
      }
      ```
    
- for of 반복문
    
    ```jsx
    <script>
            const array = ['a', 'b', 'c', 'd', 'e']
    
            for (const i of array) {
                console.log(`data : ${i}`)
            }
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_for_of_반복문/Untitled%202.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판