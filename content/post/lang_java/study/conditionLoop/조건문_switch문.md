---
title: "[Java] 조건문 - switch문"
description: ""
date: "2022-07-10T10:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

# Switch 문

---

- if 문과 마찬가지로 조건 제어문
- 변수가 어떤 값을 갖느냐에 따라 실행문이 선택
    
    ## switch 문의 형식과 실행 흐름
    
    ---
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_switch문/Untitled.png)
    
    - 괄호 안의 값과 동일한 값을 갖는 case로 가서 실행문을 실행
    - 만약 괄호 안의 값과 동일한 값을 갖는 case가 없으면 default로 가서 실행문을 실행
    - default는 생략 가능
    - case 끝에 break가 붙어 있는 이유는 다음 case를 실행하지 말고, switch문을 빠져나가기 위해서이다.
    - break가 없다면 다음 case가 연달아 실행된다
    
    ## 사용 예
    
    ---
    
    ### 1. 주사위 번호 하나 뽑기
    
    ```java
    public class SwitchEx {
    
    	public static void main(String[] args) {
    		int num = (int)(Math.random() * 6) + 1;
    		
    		switch(num) {
    			case 1:
    				System.out.println("1번");
    				break;
    			case 2:
    				System.out.println("2번");
    				break;
    			case 3:
    				System.out.println("3번");
    				break;
    			case 4:
    				System.out.println("4번");
    				break;
    			case 5:
    				System.out.println("5번");
    				break;
    			default:
    				System.out.println("6번");
    				break;
    		}
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_switch문/Untitled%201.png)
    
    ### 2. break문이 없는 case
    
    ```java
    public class SwitchNoBreakEx {
    
    	public static void main(String[] args) {
    		int time = (int)(Math.random() * 4) + 8; 
    			// 8<= ... <=11 사이의 정수 뽑기
    		System.out.println("[현재시간 : " + time + " 시]");
    		
    		switch(time) {
    			case 8:
    				System.out.println("출근");
    			case 9:
    				System.out.println("회의");
    			case 10:
    				System.out.println("업무");
    			default:
    				System.out.println("외근");
    		}
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_switch문/Untitled%202.png)
    
    ### 3. char 타입 변수를 이용한 switch 문
    
    ```java
    public class SwitchCharEx {
    
    	public static void main(String[] args) {
    		char grade = 'B';
    		
    		switch(grade) {
    			case 'A':
    			case 'a':
    				System.out.println("우수 회원");
    				break;
    			case 'B':
    			case 'b':
    				System.out.println("일반 회원");
    				break;
    			default:
    				System.out.println("게스트");
    		}
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_switch문/Untitled%203.png)
    
    ### 4. String 타입 변수를 이용한 switch 문
    
    ```java
    public class SwitchStringEx {
    
    	public static void main(String[] args) {
    		String position = "과장";
    		
    		switch(position) {
    			case "부장":
    				System.out.println("700만원");
    				break;
    			case "과장":
    				System.out.println("500만원");
    				break;
    			default:
    				System.out.println("300만원");
    		}
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_switch문/Untitled%204.png)
    

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판