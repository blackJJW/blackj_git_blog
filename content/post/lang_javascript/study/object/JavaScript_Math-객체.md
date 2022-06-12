---
title: "[JavaScript] Math 객체"
description: ""
date: "2022-06-06T16:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 수학과 관련된 기본적인 연산을 할 때는 **Math 객체**를 사용
- 객체 속성 : **pi**, **e**
- 메소드
    - 삼각함수 : **Math.sin**(), **Math.cos**(), **Math.tan**()
    
    ![Untitled](/images/lang_javascript/study/JavaScript_Math_객체/Untitled.png)
    

- 추가로 **Math 객체 정보는 다음 링크**를 참조
    
    [모질라 Math 객체의 속성과 메소드](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Math)
    

- Math 객체의 메소드 중에서 많이 사용하는 ‘**랜덤한 숫자 생성**’을 하려면 **Math.random**()  메소드를 사용
    - **0 이상 1 미만의 랜덤한 숫자**를 생성
- Math.random() 메소드
    
    ```jsx
    <script>
    	const num = Math.random()
    	
    	console.log('# 랜덤 숫자')
    	console.log(`0 ~ 1 사이의 랜덤 숫자 : ${num}`)
    	console.log('')
    
    	console.log('# 랜덤 숫자 범위 확대')
    	console.log(`0 ~ 10 사이의 랜덤 숫자 : ${num * 10}`)
    	console.log(`0 ~ 50 사이의 랜덤 숫자 : ${num * 50}`)
    	console.log('')
    
    	console.log('# 랜덤 숫자 범위 이동')
    	console.log(`-5 ~ 5 사이의 랜덤 숫자 : ${num * 10 - 5}`)
    	console.log(`-25 ~ 25 사이의 랜덤 숫자 : ${num * 50 - 25}`)
    	console.log('')
    
    	console.log('# 랜덤 정수 숫자 ')
    	console.log(`-5 ~ 5 사이의 랜덤 숫자 : ${Math.floor(num * 10 - 5)}`)
    	console.log(`-25 ~ 25 사이의 랜덤 숫자 : ${Math.floor(num * 50 - 25)}`)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_Math_객체/Untitled%201.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_Math_객체/Untitled%202.png)
    
    - 랜덤하게 숫자가 출력되는 것을 확인

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판
- https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Math