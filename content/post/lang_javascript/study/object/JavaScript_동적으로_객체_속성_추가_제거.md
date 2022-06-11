---
title: "[JavaScript] 동적으로 객체 속성 추가/제거"
description: ""
date: "2022-06-03T16:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 객체를 **처음 생성한 후**에 **속성을 추가하거나 제거**하는 것을 ‘**동적으로 속성을 추가**한다’ 또는 ‘**동적으로 속성을 제거**한다’고 표현

### 동적으로 객체 속성 추가

- 객체를 생성한 후 **속성을 지정하고 값을 입력**
- 동적으로 객체 속성 추가
    
    ```jsx
    <script>
    	const programmer = {}
    	programmer.name = 'John'
    	programmer.language = 'JavaScript'
    	programmer.part = 'front-end'
    
    	console.log(JSON.stringify(programmer, null, 2))
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_동적으로_객체_속성_추가_제거/Untitled.png)
    

### 동적으로 객체 속성 제거

- 객체의 속성을 제거할 때는 delete 키워드를 사용
    
    ```jsx
    delete 객체.속성
    ```
    
- 동적으로 객체 속성 제거
    
    ```jsx
    <script>
    	const programmer = {}
    	programmer.name = 'John'
    	programmer.language = 'JavaScript'
    	programmer.part = 'front-end'
    	
    	delete programmer.part
    
    	console.log(JSON.stringify(programmer, null, 2))
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_동적으로_객체_속성_추가_제거/Untitled%201.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판