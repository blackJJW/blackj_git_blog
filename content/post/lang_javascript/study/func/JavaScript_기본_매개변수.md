---
title: "[JavaScript] 기본 매개변수"
description: ""
date: "2022-05-29T16:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 함수의 매개변수로 항상 비슷한 값을 입력하는 경우가 있음
    - 이러한 경우 매개변수에 기본값을 지정하는 **기본 매개변수**를 사용
    
    ```jsx
    함수이름(매개변수, 매개변수=기본값, 매개변수=기본값)
    ```
    
- 매개변수는 **왼쪽부터 입력**하므로 다음과 같이 함수를 작성하면 기본 매개변수의 의미가 없음.
    - b에 값을 전달하기 위해서는 a에 값을 채워야 하기 때문
    
    ```jsx
    function sample(a=기본값, b) { }
    ```
    
    - 기본 매개변수는 **오른쪽 매개변수에 사용**
- 기본 매개변수의 활용
    - 함수 이름 : earnings
    - 매개 변수 : name,(이름), wage(시급), hours(시간)
    - 함수의 역할
        - 이름, 시급, 시간을 출력
        - 시급과 시간을 곱한 최종 급여 출력
    - wage와 hours를 입력하지 않고 실행 시, wage에 최저 임금이 들어가고, hours에 법정근로 시간 1주일 40시간이 기본 매개변수로 입력
    
    ```jsx
    <script>
    	function earnings (name, wage=8590, hours=40) {
    		console.log(`# ${name} 급여 정보`)
    		console.log(`- 시급 : ${wage}원`)
    		console.log(`- 근무 시간 : ${hours}시간`)
    		console.log(`- 급여 : ${wage * hours}원`)
    		console.log('')
    	}
    
    	earnings('구름')
    	earnings('별', 10000)
    	earnings('달', 10000, 52)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study/JavaScript_기본_매개변수/Untitled.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판