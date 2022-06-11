---
title: "[JavaScript] 객체"
description: ""
date: "2022-06-03T15:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 자바스크립트에서 여러 자료를 다룰 때는 **객체**(**object**)를 사용
- **배열**(**array**)도 여러 자료를 다룰 수 있음
    - 배열도 객체이기 때문
        
        ![Untitled](/images/lang_javascript/study/JavaScript_객체/Untitled.png)
        
    
    - 배열은 객체를 기반으로 만들어져 객체와 상당히 비슷
- 배열과 객체의 차이점
    - 배열 : 요소에 접근 시 **인덱스** 사용
    - 객체 : 요소에 접근 시 **키(key)** 사용

- 객체는 중괄호 […]로 생성, 자료를 쉼표( , )로 연결해서 입력
    
    ```jsx
    키: 값
    ```
    
    ```jsx
    <script>
    	const product = {
    		제품명: '건조 망고',
    		유형: '당절임',
    		성분: '망고, 설탕',
    		원산지: '필리핀'
    	}
    </script>
    ```
    
    - 위에서 생성한 객체를 표로 표현
        - 배열과 비슷함을 확인
        
        | 키 |   속성    |
        |:-------:| :---: |
        | 제품명 |  건조 망고  |
        | 유형 |   당절임   |
        | 성분 | 망고, 설탕  |
        | 원산지 |   필리핀   |
    - 객체 뒤에 **대괄호 […]를 사용**하고 **키를 입력**하면 객체의 요소에 접근
        
        ```jsx
        product['제품명']   -> '건조 망고'
        product['유형']     -> '당절임'
        product['성분']     -> '망고, 설탕'
        product['원산지']   -> '필리핀'
        ```
        
    - 대괄호를 사용하는 방법 이외에 **온점(.)**을 사용 가능
        
        ```jsx
        product.제품명   -> '건조 망고'
        product.유형     -> '당절임'
        product.성분     -> '망고, 설탕'
        product.원산지   -> '필리핀'
        ```
        
        - 온점을 사용하면 **보조 기능을 활용**할 수 있음
            
            ![Untitled](/images/lang_javascript/study/JavaScript_객체/Untitled%201.png)
            
    - **식별자로 사용할 수 없는 단어**를 키로 사용할 경우
        - 객체를 생성할 때 키(key)는 식별자와 문자열을 모두 사용 가능
        - 식별자로 사용할 수 없는 단어를 키로 사용할 때는 **문자열을 사용**해야함
        - 식별자가 아닌 문자열을 키로 사용했을 때문 **무조건 대괄호[…]를 사용해야 객체의 요소에 접근 가능**
        
        ```jsx
        <script>
        	const object = {
        		"with space": 123,
        		"with ~!@# $%^&*()_+": 456
        	}
        
        	object["with space"]
        	object["with ~!@# $%^&*()_+"]
        </script>
        ```
        
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판