---
title: "[Java] 반복문 - continue문"
description: ""
date: "2022-07-10T17:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

# continue문

---

- for문, while문, do-while문에서만 사용
- 블록 내부에서 continue문이 실행되면 **for문의 증감식** 또는 **while문, do-while문의 조건식**으로 **이동**
    
    ## continue문의 실행 흐름
    
    ---
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_continue문/Untitled.png)
    
- continue문은 반복문을 종료하지 않고 계속 반복을 수행
- 대게 **if문과 같이 사용**
    - 조건을 만족하는 경우
        - continue문을 실행
        - **그 이후의 문장을 실해하지 않고 다음 반복으로 넘어감**
- ex)
    
    ```java
    public class ContinueEx {
    
    	public static void main(String[] args) {
    		for(int i = 1; i <= 10; i++) {
    			if(i % 2 != 0) {
    				continue;
    			}
    			System.out.println(i);
    		}
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_continue문/Untitled%201.png)
    

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판