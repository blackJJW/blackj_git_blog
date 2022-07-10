---
title: "[Java] 반복문 - do-while문"
description: ""
date: "2022-07-10T15:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

# do-while문

---

- 조건식에 의해 반복 실행
- 블록 내부의 **실행문을 우선 실행**시키고 **실행 결과에 따라서 반복 실행을 계속할 지 결정**
    
    ## do-while문의 작성 형식과 실행 흐름
    
    ---
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_do-while문/Untitled.png)
    
    ## 콘솔에서 입력한 문자열을 읽는 방법
    
    ---
    
    - `System.in.read()` 메소드는 하나의 키 코드만 읽기 때문에 콘소레 입력된 문자열을 한 번에 읽을 수 없다.
    - `Scanner` 객체를 생성하고 `nextLine()` 메소드를 호출하면 콘솔에 입력된 문자열을 한 번에 읽을 수 있다.
        - `nextLine()` 메소드로 읽은 문자열을 저장하기 위해서는 String 변수가 필요
            
            ```java
            Scanner scanner = new Scanner(System.in); // Scanner 객체 생성
            String inputString = scanner.nextLine(); // nextLine() 메소드 호출
            ```
            
- ex)
    
    ```java
    import java.util.Scanner; // Scanner 클래스를 사용하기 위해 필요
    
    public class DoWhileEx {
    
    	public static void main(String[] args) {
    		System.out.println("메시지를 입력");
    		System.out.println("프로그램을 종료하려면 q를 입력하세요.");
    		
    		Scanner scanner = new Scanner(System.in); // Scanner 객체 생성
    		String inputString;
    		
    		do {
    			System.out.print(">");
    			inputString = scanner.nextLine(); // 키보드로 입력한 문자열을 얻음
    			System.out.println(inputString);
    			
    		} while(!inputString.equals("q")); // 문자열과 비교할 때는 equals() 메소드 사용
    		
    		System.out.println();
    		System.out.println("프로그램 종료");
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/conditionLoop/반복문_do-while문/Untitled%201.png)
    

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판