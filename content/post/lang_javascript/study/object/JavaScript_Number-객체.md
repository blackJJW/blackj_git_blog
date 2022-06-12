---
title: "[JavaScript] Number 객체"
description: ""
date: "2022-06-06T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

### 숫자 N번째 자릿수까지 출력 : toFixed()

- 소수점 이하 몇 자리까지만 출력하고 싶을 시 사용
    
    ![Untitled](/images/lang_javascript/study/JavaScript_Number_객체/Untitled.png)
    

### NaN과 Infinity 확인 : isNaN(), isFinite()

- 어떤 숫자가 **NaN**(**Not a Number**)인지 또는 **Infinity**(무한)인지 확인할 때는 **Number.isNaN**() 메소드와 **Number.isFinite**() 메소드를 사용
- 이 메소드들은 **Number 뒤에 점을 찍고 사용**
- **isNaN**()
    
    ![Untitled](/images/lang_javascript/study/JavaScript_Number_객체/Untitled%201.png)
    
    - NaN과 비교해서는 NaN인지 확인할 수 없다.
- **isFinity**()
    
    ![Untitled](/images/lang_javascript/study/JavaScript_Number_객체/Untitled%202.png)
    
    - 무한대 숫자는 양의 무한대인 **Infinity** 와 음의 무한대인 -**Infinity** 로 나뉨
    - NaN과 다르게 무한대 값은 비교 연산자로 비교 가능
        
        ![Untitled](/images/lang_javascript/study/JavaScript_Number_객체/Untitled%203.png)
        

- 추가로 **Number 객체 정보는 다음 링크**를 참조
    
    [모질라 Number 객체의 속성과 메소드](https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Number)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판
- https://developer.mozilla.org/ko/docs/Web/JavaScript/Reference/Global_Objects/Number