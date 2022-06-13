---
title: "[JavaScript] 배열 전개 연산자"
description: ""
date: "2022-06-08T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 배열과 객체는 할당할 때 **얕은 복사**가 이루어짐
- 얕은 복사 이해
    
    ```jsx
    <script>
    	const computer_1 = ['monitor', 'mouse']
    	const computer_2 = computer_1
    	
    	computer_2.push('keyboard')
    	computer_2.push('desktop')
    
    	console.log(computer_1)
    	console.log(computer_2)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_배열_전개_연산자/Untitled.png)
    
    - 특이하게도 같은 값이 출력
        - 같은 배열에 다른 이름을 붙을 뿐인 것을 **얕은 복사**(**참조 복사**)

- **깊은 복사**
    - 복사한 두 배열이 **완전히 독립적으로 작동**
    - 깊은 복사를 ‘클론(clone)을 만드는 것’이라고 표현하기도 함
    - **전개 연산자**를 사용
    - 전개 연산자를 사용한 **배열 복사**
        
        ```jsx
        [...배열]
        ```
        
        ```jsx
        <script>
        	const computer_1 = ['monitor', 'mouse']
        	const computer_2 = [...computer_1]
        	
        	computer_2.push('keyboard')
        	computer_2.push('desktop')
        
        	console.log(computer_1)
        	console.log(computer_2)
        </script>
        ```
        
        ![Untitled](/images/lang_javascript/study_1/JavaScript_배열_전개_연산자/Untitled%201.png)
        
    - 전개 연산자를 사용한 **배열 요소 추가**
        
        ```jsx
        <script>
        	const computer_1 = ['monitor', 'mouse']
        	const computer_2 = ['keyboard', ...computer_1, 'desktop']
        	
        	console.log(computer_1)
        	console.log(computer_2)
        </script>
        ```
        
        ![Untitled](/images/lang_javascript/study_1/JavaScript_배열_전개_연산자/Untitled%202.png)
        
    - 배열을 **여러 번 전개**하는 것도 가능
        
        ![Untitled](/images/lang_javascript/study_1/JavaScript_배열_전개_연산자/Untitled%203.png)
        

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판