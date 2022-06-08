---
title: "[JavaScript] for 반복문"
description: ""
date: "2022-05-23T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 일반적으로 **특정 횟수만큼 반복하고 싶을때 사용**하는 범용적인 반복문
    
    ```jsx
    for (let i = 0; i < 반복 횟수; i++) {
    	문장
    }
    ```
    
- **for** 코드 블록 사용
    - 코드 작성 시 **for** 입력
    - **for 와 관련된 여러 블록**들이 나옴
        
        ![Untitled](/images/lang_javascript/study/JavaScript_for_반복문/Untitled.png)
        
    - **for 코드 블록**으로 이동 후 `Enter` 또는 `Tab` 키를 누른다.
        
        ![Untitled](/images/lang_javascript/study/JavaScript_for_반복문/Untitled%201.png)
        
        ![Untitled](/images/lang_javascript/study/JavaScript_for_반복문/Untitled%202.png)
        
        ```jsx
        for (let index = 0; index < array.length; index++) {
            const element = array[index];
                    
        }
        ```
        
- for 반복문 기본
    - 배열과 함께 사용시 for문의 반복횟수는 **배열의 length 속성을 사용**한다.
    
    ```jsx
    <script>
            const array = ['a', 'b', 'c', 'd', 'e']
    
            for (let index = 0; index < array.length; index++) {
                const element = array[index];
                console.log(`data ${index} : ${element}`)
            }
    
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_for_반복문/Untitled%203.png)
    
- 1부터 N까지 더하기
    
    ```jsx
    <script>
            output = 0
    
            for (let index = 0; index <= 100; index++) {
                output += index
            }
            console.log(`1 ~ 100까지의 합 : ${output}`)
    
        </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_for_반복문/Untitled%204.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판