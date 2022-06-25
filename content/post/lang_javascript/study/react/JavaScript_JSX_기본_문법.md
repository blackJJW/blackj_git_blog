---
title: "[JavaScript][Library][React] JSX 기본 문법"
description: ""
date: "2022-06-24T16:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "JavaScript Library"
  - "HTML"
  - "React"


---
<!--more-->

- **JSX 문법**
    - 태그를 만드는 기능
    - 태그 내부에 **표현식을 삽입해서 출력**하는 기능
        - 표현식을 출력할 때는 {…} 기호 사용
            - **속성으로 표현식을 출력할 때는 따옴표를 사용하면 안 된다.**
        - 표현식 출력
            
            ```html
            <태그>{표현식}</태그>
            <태그 속성={표현식} /> // 따옴표 사용하면 안된다. 
            ```
            
        - 표현식 출력
            
            ```jsx
            <script type="text/babel">
            	// 상수 선언
            	const name = 'John'
            	const imgUrl = 'https://placedog.net/400/200'
            	
            	// 컴포넌트와 컨테이너 생성
            	const component = <div>
            		<h1>{name} Hello!!!</h1>
            		<img src={imgUrl} />
            	</div>
            	const container = document.getElementById('root')
            
            	// 출력
            	ReactDOM.render(component, container)
            	
            </script>
            ```
            
            ![Untitled](/images/lang_javascript/study_3/JavaScript_JSX_기본_문법/Untitled.png)
            
    
## References
    
- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판