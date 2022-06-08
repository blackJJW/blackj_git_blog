---
title: "[JavaScript] 불 자료형"
description: ""
date: "2022-05-07T20:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

### <u>불 만들기</u>

- 참과 거짓 값을 표현할 때 **불**(**boolean**) **자료형**을 사용
- **true** 와 **false**

    ![Untitled](/images/lang_javascript/study/JavaScript_불_자료형/Untitled.png)

- 두 대상을 비교할 수 있는 **비교 연산자**를 사용해 불을 만들 수 있음

    | 연산자 |       설명        |
    |:---:|:---------------|
    | === |     양쪽이 같다.     |
    | !== |    양쪽이 다르다.     |
    | \>  |    왼쪽이 더 크다.    |
    |  <  |   오른쪽이 더 크다.    |
    | \>= |  왼쪽이 더 크거나 같다.  |
    | <=  | 오른쪽이 더 크거나 같다.  |

    ![Untitled](/images/lang_javascript/study/JavaScript_불_자료형/Untitled%201.png)

- 불 자료형의 사용
    - **조건문**
        
        ```jsx
        if(불 표현식){
        	불 표현식 = true 일 때 실행할 문장
        }
        ```
        
    - 불 표현식 이해
        
        ```jsx
        <script>
        	if (10 < 5) {
        		alert('10은 5보다 작음')
        	}
        	if (10 > 5) {
        		alert('10은 5보다 큼')
        	}
        </script>
        ```
        
        ![Untitled](/images/lang_javascript/study/JavaScript_불_자료형/Untitled%202.png)
        

### <u>불 부정 연산자</u>

- **논리 부정 연산자**
    - **! 기호**를 사용하며 true를 false로, false를 true로 바꾼다.
        
        ![Untitled](/images/lang_javascript/study/JavaScript_불_자료형/Untitled%203.png)
        

- **단항 연산자**
    - 연산자는 피연산자 개수에 따라 **단항 연산자**, **이항 연산자**, **삼항 연산자**로 구분
    
    ```jsx
    !true          // 피연산자가 true로 1개  -> 단항 연산자
    
    1 + 2          // 피연산자가 1과 2로 2개 -> 이항 연산자
    
    true ? 1 : 2   // 피연산자 true, 1, 2로 3개  -> 삼항 연산자
    ```
    

### <u>불 논리합 / 논리곱 연산자</u>

| 연산자 |   설명 |
|:---:|:---:|
| &&  | 논리곱 연산자 |
| ll  | 논리합 연산자 |

- **&& 연산자**
    - 양쪽 변의 값이 **모두 true**일 때 **true**를 출력
    - 이외에는 **모두 false**
    
    | 좌 |   우    |   결과   |
    |:------:|:------:|:---:|
    | true |  true  |  true  |
    | true | false  | false  |
    | false |  true  | false  |
    | false | false  | false  |

- **|| 연산자**
    - 양쪽 변의 값 중 **하나만 true**여도 **true**를 출력
    
    | 좌 |   우    |   결과   |
    |:------:|:------:| :---: |
    | true |  true  |  true  |
    | true | false  |  true  |
    | false |  true  |  true  |
    | false | false  | false  |

    ![Untitled](/images/lang_javascript/study/JavaScript_불_자료형/Untitled%204.png)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_불_자료형/Untitled%205.png)

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판