---
title: "[Java] 반복문 - for문"
description: ""
date: "2022-07-10T13:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

# for문

---

- 프로그램을 작성하다 보면 똑같은 실행문을 반복적으로 실행해야 할 경우가 많이 발생
    
    ```java
    int sum = 0;
    sum = sum + 1; // 5개의 실행문
    sum = sum + 2;
    sum = sum + 3;
    sum = sum + 4;
    sum = sum + 5;
    
    System.out.println("1 ~ 5까지의 합 : " + sum);
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_for문/Untitled.png)
    
    - 위 코드는 1부터 5까지 같은 실행문이 반복됨
    - 만약 5가 아니라 그 이상의 수를 반복한다면 코드의 수가 많아지고, 그 수 만큼 코드가 반복된다.
    - 이런 경우 for문을 사용하면 코드를 획기적으로 줄여준다.
    - **1부터 100까지의 합**
        
        ```java
        int sum = 0;
        for (int i = 1; i <= 100; i++) { // 100번 반복
        	sum = sum + i;
        }
        System.out.println("1 ~ 100까지의 합 :" + sum);
        ```
        
        ![Untitled](/images/lang_java/conditionLoop/반복문_for문/Untitled%201.png)
        
- 반복문은 한 번 작성된 실행문을 여러 변 반복 실행해주기 때문에 코드를 절감하고 간결하게 만들어준다.
- **for문은 주어진 횟수만큼 실행문을 반복 실행할 때 적합한 반복문**
    
    ## for문의 형식과 실행 흐름
    
    ---
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_for문/Untitled%202.png)
    
    ### 실행과정
    
    1. 처음 실행 시, 초기화식이 제일 먼저 실행
    2. 조건식을 평가해서 true / false를 판단
    3. 조건식이 true이면 실행문을 실행
        - false이면 for문을 실행하지 않고 종료
    4. 블록 내부의 실행문들이 모두 실행되면, 증감식을 실행시키고 다시 조건식을 평가
    5. 평가 결과가 true이면 다시 실행문 → 증감식 → 조건식 과정을 다시 진행
    6. 조건식이 false이면 for문 종료
- ex)
    
    ```java
    public class ForLoop {
    
    	public static void main(String[] args) {
    		for (int i = 1; i <= 10; i++) {
    			System.out.println(i);
    		}
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_for문/Untitled%203.png)
    
- **초기화식의 역할**
    - 조건식과 실행문, 증감식에서 사용할 **변수를 초기화하는 역할**
    - 초기화식이 **필요 없을 경우**에는 다음과 같이 초기화식을 생략 가능
        
        ```java
        int i = 1;
        for(; i <= 100; i++) {...}
        ```
        
    - 초기화식이 둘 이상 있을 수 있고, 증감식도 둘 이상 있을 수 있다.
        - 이런 경우에는 **쉼표( , )로 구분**해서 작성
        
        ```java
        for(i = 0, j = 100; i <= 50 && j >= 50; i++, j--) {...}
        ```
        
- 초기화식에 선언된 변수는 for문 블록 내부에서 사용되는 로컬 변수
    - for문을 벗어나서는 사용 불가
    
    ```java
    public class ForLoop {
    
    	public static void main(String[] args) {
    		int sum = 0;
    				
    		for (int i = 1; i <= 100; i++) {
    			sum += i;
    		}
    		System.out.println("i : " + i); // 에러 발생
    		System.out.println("1 ~ 100까지의 합 : " + sum);
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_for문/Untitled%204.png)
    
    - 변수 i가  for문 내부가 아니라 외부에서 선언되었다면 for문 외부에서도 사용 가능
    
    ```java
    public class ForLoop {
    
    	public static void main(String[] args) {
    		int sum = 0;
    		int i = 0;  // 변수 i 선언
    				
    		for (i = 1; i <= 100; i++) {
    			sum += i;
    		}
    		System.out.println("i : " + i);
    		System.out.println("1 ~ 100까지의 합 : " + sum);
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_for문/Untitled%205.png)
    
    ## 주의할 점
    
    ---
    
    - 초기화식에서 루프 카운트 변수를 선언할 때 부동소수점 타입을 사용하지 말아야 한다.
    - ex) 이론적으로 다음 for문은 10번 반복해야 한다.
        
        ```java
        public class ForFloatCounterEx {
        
        	public static void main(String[] args) {
        		for(float x = 0.1f; x <= 1.0f; x +=0.1f) {
        			System.out.println(x);
        		}
        
        	}
        
        }
        ```
        
        ![Untitled](/images/lang_java/conditionLoop/반복문_for문/Untitled%206.png)
        
        - 0.1은 float 타입으로 정확하게 표현할 수 없기 때문에 x에 더해지는 실제 값은 0.1보다 약간 크다.
        - 결국 루프는 9번만 실행
    
    ## 중첩된 for문
    
    ---
    
    - **for문은 또다른 for문을 내포 가능**
    - 바깥쪽 for문이 한 번 실행될 때마다 중첩된 for문은 지정된 횟수만큼 반복해서 돌다가 다시 바깥쪽 for문으로 돌아간다.
    - ex) 구구단
        
        ```java
        public class ForMultiplicationTableEx {
        
        	public static void main(String[] args) {
        		for (int i = 2; i <= 9; i++) {
        			System.out.println("*** " + i + "단 ***");
        			for (int j = 1; j <=9; j++) {
        				System.out.println(i + " x " + j + " = " + (i * j));
        			}
        		}
        
        	}
        
        }
        ```
        
        ![Untitled](/images/lang_java/conditionLoop/반복문_for문/Untitled%207.png)
        

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판