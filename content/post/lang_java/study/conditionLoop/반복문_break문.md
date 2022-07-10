---
title: "[Java] 반복문 - break문"
description: ""
date: "2022-07-10T16:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

# break문

---

- 반복문인 for문, while문, do-while문을 실행 중지할 때 사용
    
    ## 반복문에서 break문을 사용할 때의 실행 흐름
    
    ---
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_break문/Untitled.png)
    
- break문은 대개 **if문과 같이 사용**되어 **if문의 조건식에 따라 for문과 while문을 종료할 때 사용**
- ex)
    
    ```java
    public class BreakEx {
    
    	public static void main(String[] args) {
    		while (true) {
    			int num = (int)(Math.random() * 6) + 1;
    			System.out.println(num);
    			if(num == 6) {
    				break;
    			}
    		}
    		System.out.println("프로그램 종료");
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_break문/Untitled%201.png)
    
- 반복문이 중첩되어 있을 경우 break문은 가장 가까운 반복문만 종료하고 바깥쪽 반복문은 종료시키지 않는다.
- 중첩된 반복문에서 바깥쪽 반복문까지 종료시키려면 **바깥쪽 반복문에 이름(라벨)을 붙이고, `break 이름;` 을 사용**한다.
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_break문/Untitled%202.png)
    
- ex)
    
    ```java
    public class BreakOutterEx {
    
    	public static void main(String[] args) {
    		Outter: for(char upper = 'A'; upper <= 'Z'; upper++) {
    			for(char lower = 'a'; lower <= 'z'; lower++) {
    				System.out.println(upper + "-" + lower);
    				if(lower == 'g') {
    					break Outter;
    				}
    			}
    		}
    		System.out.println("프로그램 종료");
    
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_break문/Untitled%203.png)
    

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판