---
title: "[JavaScript] 배열 뒷부분에 요소 추가"
description: ""
date: "2022-05-19T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

### push() 메소드를 사용해 배열 뒷부분에 요소 추가

- 배열 **뒷부분에 요소를 추가**할 때는 **push() 메소드**를 사용
    
    ```jsx
    배열.push(요소)
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_배열_뒷부분에_요소_추가/Untitled.png)
    

### 인덱스를 사용해 배열 뒷부분에 요소 추가

- 자바스크립드에서 배열의 길이는 고정이 아님
- 인덱스를 지정하여 인덱스 위치에 요소를 강제로 추가 가능
    - ex) 3개의 요소를 가진 배열에 10번째 인덱스에 요소를 추가
        - 4 ~ 9번째 인덱스에는 아무 것도 없는 **empty**가 된다.
    
    ![Untitled](/images/lang_javascript/study/JavaScript_배열_뒷부분에_요소_추가/Untitled%201.png)
    
- length 속성을 사용하여 배열의 마지막 위치에 요소를 추가 가능
    
    ![Untitled](/images/lang_javascript/study/JavaScript_배열_뒷부분에_요소_추가/Untitled%202.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판