---
title: "[Java] 향상된 for문"
description: ""
date: "2022-07-14T18:00:45+09:00"
thumbnail: ""
categories:
  - "Java"
tags:
  - "Java"


---
<!--more-->

- 자바 5부터 **배열 및 컬렉션 객체를 좀 더 쉽게 처리할 목적**으로 향상된 for문을 제공
- 향상된 for문은 **반복 실행을 하기 위해 카운터 변수와 증감식을 사용하지 않는다.**

## 향상된 for문을 작성하는 형식과 실행 흐름

---

![Untitled](/images/lang_java/referenceType/향상된_for문/Untitled.png)

- for문의 괄호( )에는 배열에서 꺼낸 항목을 저장할 **변수 선언**과 **콜론**( : ) 그리고 배열을 나란히 작성
- for문이 처음 실행될 때
    - 배열에서 가져올 첫 번째 값이 존재하는지 평가
    - 가져올 값이 존재하면 해당 값을 변수에 저장
    - 실행문을 실행
    - 다시 루프를 돌아 배열에서 가져올 값이 있는지 평가
        - 값이 존재할 경우 다시 진행
        - 값이 없을 경우 for문 종료
- for문의 반복 횟수는 **배열의 항목수**
- ex)
    
    ```java
    public class AdvancedForEx {
    
    	public static void main(String[] args) {
    		int[] scores = {95, 71, 84, 93, 87};
    		
    		int sum = 0;
    		for (int score : scores) {
    			sum = sum + score;
    		}
    		System.out.println("점수 총합 : " + sum);
    		
    		double avg = (double) sum / scores.length;
    		System.out.println("점수 평균 : " + avg);
    	}
    
    }
    ```
    
    ![Untitled](/images/lang_java/referenceType/향상된_for문/Untitled%201.png)
    

---

## References

- 이것이 자바다 신용권의 Java 프로그래밍 정복 - 신용권 지음, 한빛미디어 출판