---
title: "[JavaScript] 객체를 처리하는 함수"
description: ""
date: "2022-06-17T18:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 기능을 단순하게 계산하는 것 보다는 **함수로 구현**하는 것이 **확장성을 고려했을 때 좋은 방법**
    
    ```jsx
    <script>
    	// 객체를 선언
      const students = []
      students.push({ 이름: 'John', 국어: 80, 영어: 85, 수학: 90, 과학: 95 })
      students.push({ 이름: 'Amy', 국어: 76, 영어: 90, 수학: 96, 과학: 91 })
      students.push({ 이름: 'Jack', 국어: 98, 영어: 60, 수학: 93, 과학: 77 })
      students.push({ 이름: 'Ann', 국어: 70, 영어: 87, 수학: 87, 과학: 65 })
      students.push({ 이름: 'Tom', 국어: 87, 영어: 94, 수학: 76, 과학: 82 })
    
    	// 객체를 처리하는 함수를 선언
    	function getSumOf (student) {
    		return student.국어 + student.영어 + studnet.수학 + student.과학
    	}
    
    	function getAverageOf (student) {
    		return getSumOf(student) / 4
    	}
    
    	// 출력
    	let output = '이름\t총점\t평균\n'
      for (const s of students) {
      	output += `${s.이름} \t${getSumOf(s)}점 \t${getAverageOf(s)}점\n`
      }
      console.log(output)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_객체를_처리하는_함수/Untitled.png)
    
    - 전체적인 코드의 길이는 길어졌진만, **객체를 만드는 부분**과 **활용하는 부분**으로 나뉨
    - 객체에 더 많은 기능을 추가하게 되었을 때 **객체를 쉽게 유지보수가 가능**
    - 객체를 활용할 때도 거 **간단하게 코드를 작성 가능**
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판