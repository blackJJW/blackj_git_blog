---
title: "[Java] 반복문 - while문"
description: ""
date: "2022-07-10T14:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

# while문

---

- 조건식인 true일 경우에 계속해서 반복
- 조건식에는 비교 또는 논리 연산식이 주로 온다.
- 조선식이 false가 되면 반복 행위를 멈추고 while문은 종료된다.
    
    ## while문을 작성하는 형식과 실행 흐름
    
    ---
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_while문/Untitled.png)
    
    ### 실행과정
    
    1. while문이 처음 실행될 때, 조건식을 평가
    2. 평가 결과가 true이면 실행문을 실행한다.
    3. 실행문이 모두 실행되면 다시 조건식으로 되돌아가서 조건식을 다시 평가한다.
    4. 조건식이 true라면, 다시 실행문을 실행하고, false라면 while 문을 종료한다.
- ex)
    
    ```java
    public class WhileEx {
    
    	public static void main(String[] args) {
    		int i = 1;
    		while(i <= 10) {
    			System.out.println(i);
    			i++;
    		}
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_while문/Untitled%201.png)
    
- ex) 1부터 100까지의 합
    
    ```java
    public class WhileEx {
    
    	public static void main(String[] args) {
    		int sum = 0;
    		int i = 1;
    		
    		while(i <= 100) {
    			sum += i;
    			i++;
    		}
    		System.out.println("sum : " + sum);
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_while문/Untitled%202.png)
    
- 조건식에는 boolean 변수나 true / false 값을 산출하는 어떠한 연산식이든 올 수 있다.
- 조건식에 true를 사용하면 `while(true) {…}` 가 되어 무한 루프를 돌게된다.
- 무한 루프는 무한히 반복 실행하기 때문에 언젠가는 while문을 빠져나가기 위한 코드가 필요
    
    ## 키보드 키 코드
    
    ---
    
    - 키보드에서 입력된 키가 어떤 키인지 확인하는 방법
        
        ```java
        int keyCode = System.in.read();
        ```
        
        - 키보드로부터 입력된 키 코드를 리턴
    - 키 코드
        
        ![Untitled](/images/lang_java/conditionLoop/반복문_while문/Untitled%203.png)
        
    - ex)
        - while 문의 조건식에 run이라는 boolean 타입 변수를 사용
        - 변수 run의 초기값은 true이므로 while문은 무한 루프를 돌게 된다.
        - while문을 종료하기 휘ㅐ 키보드로 3번을 입력하여 변수 run의 값을 false로 만든다.
        
        ```java
        public class WhileKeyControlEx {
        
        	public static void main(String[] args) throws Exception {
        		boolean run = true;
        		int speed = 0;
        		int keyCode = 0;
        		
        		while(run) {
        			if(keyCode != 13 && keyCode != 10) { // Enter키가 입력되면 
        												 //캐리지리턴(13)과 라인피드(10)가 입력되므로
        												 // 이 값을 제외시킴
        				System.out.println("-------------------------");
        				System.out.println("1. 증속 | 2. 감속 | 3. 중지");
        				System.out.println("-------------------------");
        				System.out.println("선택 : ");
        			}
        			keyCode = System.in.read(); // 키보드의 키 코드를 읽음
        			
        			if(keyCode == 49) { // 1을 읽었을 경우
        				speed++;
        				System.out.println("현재 속도 : " + speed);
        			} else if (keyCode == 50) { // 2를 읽었을 경우 
        				speed--;
        				System.out.println("현재 속도 : " + speed);
        			} else if(keyCode == 51) { // 3을 읽었을 경우
        				run = false;
        			}
        		}
        		System.out.println("프로그램 종료");
        
        	}
        
        }
        ```
        
        ![Untitled](/images/lang_java/conditionLoop/반복문_while문/Untitled%204.png)
        

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판