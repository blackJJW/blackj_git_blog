---
title: "[JavaScript] 같은 형태의 객체 만들기"
description: ""
date: "2022-06-17T17:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 학생 성적 관리 프로그램을 만든다고 가정
- ‘**학생**’이라는 객체가 필요
    - 학생들로부터 성적 관리에 필요한 **공통사항을 추출**
    - 이 과정을 **추상화**라고 함
    - 학생들이 여러 명이므로 추출한 요소는 **배열**로 관리


- 객체와 배열 조합
    
    ```jsx
    <script>
    	// 객체를 선언
    	const students = []
    	students.push({ 이름: 'John', 국어: 80, 영어: 85, 수학: 90, 과학: 95 })
    	students.push({ 이름: 'Amy', 국어: 76, 영어: 90, 수학: 96, 과학: 91 })
    	students.push({ 이름: 'Jack', 국어: 98, 영어: 60, 수학: 93, 과학: 77 })
    	students.push({ 이름: 'Ann', 국어: 70, 영어: 87, 수학: 87, 과학: 65 })
    	students.push({ 이름: 'Tom', 국어: 87, 영어: 94, 수학: 76, 과학: 82 })
    
    	// 출력
    	alert(JSON.stringify(students, null, 2))
                    // 객체를 JSON 문자열로 변환할 때 사용하는 메소드
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_같은_형태의_객체_만들기/Untitled.png)

    
- 객체 활용
    
    ```jsx
    <script>
    	// 객체를 선언
    	const students = []
    	students.push({ 이름: 'John', 국어: 80, 영어: 85, 수학: 90, 과학: 95 })
    	students.push({ 이름: 'Amy', 국어: 76, 영어: 90, 수학: 96, 과학: 91 })
    	students.push({ 이름: 'Jack', 국어: 98, 영어: 60, 수학: 93, 과학: 77 })
    	students.push({ 이름: 'Ann', 국어: 70, 영어: 87, 수학: 87, 과학: 65 })
    	students.push({ 이름: 'Tom', 국어: 87, 영어: 94, 수학: 76, 과학: 82 })
    
    	// 출력
    	let output = '이름\t총점\t평균\n'
    	for (const s of students) {
    		const sum = s.국어 + s.영어 + s.수학 + s.과학
    		const average = sum / 4
    		output += `${s.이름} \t${sum}점 \t${average}점\n`
    	}
    	console.log(output)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_같은_형태의_객체_만들기/Untitled%201.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판