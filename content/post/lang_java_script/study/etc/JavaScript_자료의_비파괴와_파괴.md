---
title: "[JavaScript] 자료의 비파괴와 파괴"
description: ""
date: "2022-05-21T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 자료 처리 연산자, 함수, 메소드는 크게 **비파괴적 처리**와 **파괴적 처리**로 구분
    - 처리 후 **원본의 상태 변화**에 따라 구분
    - **비파괴적 처리** : 처리 후에 원본 내용이 변경되지 않음
    - **파괴적 처리** : 처리 후에 원본 내용이 변경

### 비파괴적 처리

- 예시
    
    ![Untitled](/images/lang_javascript/study/JavaScript_자료의_비파괴와_파괴/Untitled.png)
    
    - 코드가 실행되어도 a, b, c, 의 값이 변경되지 않았다.

### 파괴적 처리

- 예시
    
    ![Untitled](/images/lang_javascript/study/JavaScript_자료의_비파괴와_파괴/Untitled%201.png)
    
    - 코드가 실행되어 원본의 내용이 변경되었다.

- 과거에는 메모리가 많이 부족하여 파괴적 처리가 주로 이루어졌지만, **현대의 프로그래밍에서는 자료의 보호를 위해 대부분 비파괴적 처리가 이루어진다.**

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판