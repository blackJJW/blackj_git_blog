---
title: "[JavaScript] 기본 자료형의 일시적 승급"
description: ""
date: "2022-06-06T14:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 문자열 자료형 등을 생성하고 뒤에 온점을 찍으면 **자동 완성 기능으로 실행 가능한 메소드들이 출력**
    
    ![Untitled](/images/lang_javascript/study/JavaScript_기본_자료형의_일시적_승급/Untitled.png)
    

- 원래 기본 자료형은 속성과 메소드를 가질 수 없다.
- 자바스크립트는 사용의 편리성을 위해서 기본 자료형의 속성과 메소드를 호출할 때 **일시적으로 기본 자료형을 객체로 승급**
- 이러한 승급은 **일시적**
    - 속성을 추가할 때 된 것처럼 보이지만 실제로는 추가 되지 않음
        
        ![Untitled](/images/lang_javascript/study/JavaScript_기본_자료형의_일시적_승급/Untitled%201.png)
        

- 기본 자료형의 경우 **속성과 메소드를 사용할 수는 있지만**, 속성과 메소드를 **추가로 가질 수는 없다.**

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판