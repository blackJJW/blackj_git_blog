---
title: "[JavaScript] 템플릿 문자열"
description: ""
date: "2022-05-07T21:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 자바스크립트에서 문자열 **내부에 표현식을 삽입**할 때는 **문자열 연결 연산자(+)**를 사용
    - 이 경우 표현식을 많이 결합할 경우 **코드가 복잡**해짐.

    ![Untitled](/images/lang_javascript/study/JavaScript_템플릿_문자열/Untitled.png)

- 자바스크립트에 **템플릿 문자열**이라는 기능이 추가되어 간단하게 코드 작성이 가능해짐
    - 탬플릿 문자열은 **백틱**( **`** ) 기호로 감싸서 생성
        - 백틱( ` ) : 작은 따옴표가 아님
            - 키보드의 ‘1’ 키의 왼쪽에 있는 키
    - 문자열 내부에 ‘**${…}**'기호를 사용하여 표현식이 문자열 내부에서 계산된다.
    
    ```jsx
    console.log(`식 100 + 200의 값은 ${100 + 200}입니다.`)
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_템플릿_문자열/Untitled%201.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판