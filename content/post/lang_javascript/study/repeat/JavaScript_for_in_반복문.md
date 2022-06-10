---
title: "[JavaScript] for in 반복문"
description: ""
date: "2022-05-22T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 배열을 함께 사용할 수 있는 **for in 반복문**
- 배열 요소를 하나하나 꺼내서 특정 문장을 실행
    
    ```jsx
    for (const 반복 변수 in 배열 또는 객체) {
    	문장
    }
    ```
    
- for in 반복문
    
    ```jsx
    <script>
            const array = ['a', 'b', 'c', 'd', 'e']
    
            for (const i in array) {
                console.log(`${i} : ${array[i]}`)
            }
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_for_in_반복문/Untitled.png)
    
- for 반복문의 **반복 변수**(**위의 코드의 i**)에는 요소의 인덱스들이 들어옴
    - 이를 활용하여 배열 요소에 접근
- **for in 코드 블록** 사용
    - 코드 작성 시 **for** 입력
    - **for 와 관련된 여러 블록**들이 나옴
        
        ![Untitled](/images/lang_javascript/study/JavaScript_for_in_반복문/Untitled%201.png)
        
    - **forin 코드 블록**으로 이동 후 `Enter` 또는 `Tab` 키를 누른다.
        
        ![Untitled](/images/lang_javascript/study/JavaScript_for_in_반복문/Untitled%202.png)
        
        ![Untitled](/images/lang_javascript/study/JavaScript_for_in_반복문/Untitled%203.png)
        
    - 위에서 살펴보았던 기본적인 코드 외의 코드가 추가되있는 것을 확인
        - 일반적으로 추가 코드는 제거하고 사용해도 문제 없음
        
        ```jsx
        for (const key in object) {
            if (Object.hasOwnProperty.call(object, key)) {
               const element = object[key];
                        
            }
        }
        ```
        

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판