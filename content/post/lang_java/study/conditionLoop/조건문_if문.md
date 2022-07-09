---
title: "[Java] 조건문 - if문"
description: ""
date: "2022-07-09T20:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

# if문

---

- if문은 조건식의 결과에 따라 **블록 실행 여부가 결정**
    
    ## i**f문의 형식과 실행 흐름**
    
    ---
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_if문/Untitled.png)
    
    - 조건식에는 **true 또는 false 값을 산출할 수 있는 연산식이나, boolean 변수**가 올 수 있다.
    - 조건식이 **true이면 블록을 실행, false이면 블록을 실행하지 않는다.**
    
    ```java
    if (조건식) {
    	실행문;
    	실행문;
    	...
    }
    //-----------------
    if (조건식)
    	실행문;
    ```
    
    - **중괄호 { } 블록은 여러 개의 실행문을 하나로 묶기 위해 작성**
    - 만약 조건식이 true가 될 때 **실행해야 할 문장이 하나 밖에 없다면 중괄호 생략 가능**
        - 중괄호 { } 블록을 생략하지 않고 작성하는 것을 추천
        - 중괄호 블록을 작성하지 않느면 코드의 가독성(코드 해석)이 좋지 않고, 버그 발생의 원인이 될 수 있다.
- ex)
    
    ```java
    public class IfEx {
    
    	public static void main(String[] args) {
    		int score = 93;
    		
    		if (score >= 90) {
    			System.out.println("90점 이상");
    			System.out.println("A 등급");
    		}
    		
    		if (score < 90) 
    			System.out.println("90점 미만");
    			System.out.println("B 등급");  // if문과는 상관없는 실행문
    		
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_if문/Untitled%201.png)
    

# if-else 문

---

- if문은 else 블록과 함께 사용되어 **조건식의 결과에 따라 실행 블록을 선택**
- if문의 조건식이 true이면 if문의 블록이 실행되고, 조건식이 false이면 else 블록이 실행
- 조건식의 결과에 따라 **이 두 개의 블록 중 어느 한 블록의 내용만 실행하고 전체 if 문을 벗어나게 된다.**
    
    ## if-else 문의 형식과 실행 흐름
    
    ---
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_if문/Untitled%202.png)
    
- ex)
    
    ```java
    public class IfElseEx {
    
    	public static void main(String[] args) {
    		int score = 85;
    		
    		if(score >= 90) {
    			System.out.println("90점 이상");
    			System.out.println("A 등급");
    		} else {
    			System.out.println("90점 미만");
    			System.out.println("B 등급");
    		}
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_if문/Untitled%203.png)
    

# if-else if-else 문

---

- **조건문이 여러 개인 if문**
- 처음 if문의 조건식이 false인 경우 **다른 조건식의 결과에 따라 실행 블록 선택 가능**
    - if 블록 끝에 else if문을 붙인다.
    - else if 블록의 수는 제한이 없다.
    - 여러 개의 조건 식 중 true가 되는 블록만 실행하고 전체 if문을 벗어나게 된다.
    - else if 블록 마지막에는 else 블록을 추가 가능
        - 모든 조건식이 false일 경우 else 블록을 실행, if 문을 벗어나게 된다.
    
    ## if-else if-else 문의 형식과 실행 흐름
    
    ---
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_if문/Untitled%204.png)
    
- ex)
    
    ```java
    public class IfElseIfIfEx {
    
    	public static void main(String[] args) {
    		int score = 75;
    		
    		if (score >= 90) {
    			System.out.println("90 ~ 100");
    			System.out.println("A 등급");
    		} else if(score >= 80) {
    			System.out.println("80 ~ 89");
    			System.out.println("B 등급");
    		} else if(score >= 70) {
    			System.out.println("70 ~ 79");
    			System.out.println("C 등급");
    		} else {
    			System.out.println("70 미만");
    			System.out.println("D 등급");
    		}
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_if문/Untitled%205.png)
    
    ## 활용 예
    
    ---
    
    - 주사위를 굴려서 1, 2, 3, 4, 5, 6 중 하나의 수를 뽑는 프로그램
        
        ### 임의의 정수를 뽑는 방법
        
        ---
        
        - `Math.random()` 메소드 활용
            - 0.0.과 1.0 사이에 속하는 double 타입의 난수 하나를 리턴
            - 0.0은 범위에 포함되고 1.0은 포함되지 않는다.
            - 비교 연산자로 표현
                
                ```java
                0.0 <= Math.random() < 1.0
                ```
                
        - 1 ~ 10까지 정수 중에서 하나의 정수를 얻기 위한 방법
            1. 각 변에 10를 곱하여 범위에 속하는 하나의 double 타입의 값을 얻을 수 있다.
                
                ```java
                0.0 * 10 <= Math.random() * 10 < 1.0 * 10
                  (0.0)                           (10.0)
                ```
                
            2. 각 변을 int 타입으로 강제 타입 변환하여 같은 범위에 속하는 정수 하나를 얻을 수 있다. 
                
                ```java
                (int) (0.0 * 10) <= (int) (Math.random() * 10) < (int) (1.0 * 10)
                    (0)                (0, 1, 2, 3, ... 10)          (10)
                ```
                
            3. 각 변에 1을 더하여, 1 ~ 10까지 정수 중에서 하나의 정수를 얻는다.
                
                ```java
                0 + 1 <= (int) (Math.random() * 10) + 1 < 10 + 1
                 (1)         (1, 2, 3, ... 10)              (11)
                ```
                
    - **start부터 시작하는 n개의 정수 중에서 임의의 정수 하나를 얻기 위한 연산식**
        
        ```java
        int num = (int) (Math.random() * n) + start;
        ```
        
        - **주사위 번호 하나를 뽑기 위한 연산식 사용**
            
            ```java
            int num = (int) (Math.random() * 6) + 1;
            ```
            
        - **로또 번호 하나를 뽑기 위한 연산식 사용**
            
            ```java
            int num = (int) (Math.random() * 45) + 1;
            ```
            
    - 주사위의 번호를 뽑는 예제
        
        ```java
        public class IfDiceEx {
        
        	public static void main(String[] args) {
        		int num = (int) (Math.random() * 6) + 1; // 주사위 번호 하나 뽑기
        		
        		if (num == 1) {
        			System.out.println("1번");
        		} else if(num == 2) {
        			System.out.println("2번");
        		} else if(num == 3) {
        			System.out.println("3번");
        		} else if(num == 4) {
        			System.out.println("4번");
        		} else if(num == 5) {
        			System.out.println("5번");
        		} else {
        			System.out.println("6번");
        		}
        
        	}
        
        }
        ```
        
        ![Untitled](/images/lang_java/conditionLoop/조건문_if문/Untitled%206.png)
        

# 중첩 if문

---

- **if문의 블록 내부에 또 다른 if문을 사용 가능**
    - 이것을 중첩 if문
    - 중첩의 단계에는 제한이 없다.
    - if문, switch문, for문, while문, do-while문은 서로 중첩할 수 있다.
    
    ## 중첩 if문의 실행 흐름
    
    ---
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_if문/Untitled%207.png)
    
- ex)
    
    ```java
    public class IfNestedEx {
    
    	public static void main(String[] args) {
    		int score = (int) (Math.random() * 20) + 81;
    		System.out.println("score : " + score);
    		
    		String grade;
    		
    		if (score >= 90) {
    			if(score >= 95) {
    				grade = "A+";
    			} else {
    				grade = "A";
    			}
    		} else {
    			if(score >= 85) {
    				grade = "B+";
    			} else {
    				grade = "B";
    			}
    		}
    		
    		System.out.println("grade : " + grade);
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/조건문_if문/Untitled%208.png)
    

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판