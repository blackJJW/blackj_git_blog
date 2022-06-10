---
title: "[JavaScript] 구 버전 자바스크립트에서 기본 매개변수 구현"
description: ""
date: "2022-05-30T17:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 함수의 기본 매개변수는 최신 자바스크립트에서 추가된 기능
- 구 버전의 자바스크립트에서는 일반적으로 다음과 같이 기본 매개변수를 구현
    
    ```jsx
    function earnings (wage, hours) {
    	wage = typeof(wage) != undefined ? wage : 8590
    	hours = typeof(hours) != undefined ? hours : 52
    	
    	return wage * hours
    }
    ```
    
- 매개변수로 들어오는 값이 **false 또는 false로 변환되는 값(0, 빈 문자열 등)이 아니라는 게 확실하다면** 다음과 같이 **짧은 조건문을 사용해 구현**
    
    ```jsx
    function earnings (wage, hours) {
    	wage = wage || 8590
    	hours = hours || 52
    	
    	return wage * hours
    }
    ```
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판