---
title: "[JavaScript][Library][React] 루트 컴포넌트 출력"
description: ""
date: "2022-06-24T15:30:45+09:00"
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

- **컴포넌트**(**component**) : 리액트에서 **화면에 출력되는 요소**
- **루트 컴포넌트**(**root component**) : **가장 최상위에 배치**하는 컴포넌트
- 리액트는 컴포넌트를 만들 때 **HTML 요소를 만든 것과 동일한 문법 사용**
- **컴포넌트 생성**
    
    ```html
    <컴포넌트_이름></컴포넌트_이름>
    ```
    
- 이렇게 생성한 **컴포넌트를 출력**할 때는 **ReactDOM.render() 메소드**를 사용
- **컨테이너**(**container**) : 컴포넌트를 출력할 상자
- **컴포넌트 출력**
    
    ```jsx
    ReactDOM.render(컴포넌트, 컨테이너)
    ```
    

- 루트 컴포넌트 출력
    - h1 컴포넌트 생성
    - h1 컴포넌트를 출력할 div#root를 읽어들임
    - ReactDOM.render() 메소드로 컴포넌트를 div#root에 출력
    
    ```jsx
    <script type="text/babel">
    	// 컴포넌트와 컨테이너 생성
    	const component = <h1>리액트 생성</h1> // 바벨 덕분에 사용가능한 코드
    	const container = document.getElementById('root')
    
    	// 출력
    	ReactDOM.render(component, container)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_루트_컴포넌트_출력/Untitled.png)
    
- **JSX**(**자바스크립트 확장 문법**)
    - 자바스크립트 **코드 내부에 HTML 코드를 사용**
    - 웹 브라우저는 이러한 코드를 읽고 실행하지 못함
        - 바벨이 JSX 코드를 읽고 **일반적인 자바스크립트 문법으로 변환한 뒤 실행**해줌
        - **바벨 REPL 도구**를 사용하면 바벨이 어떤 식으로 코드를 바꾸는지 확인 가능
            
            [바벨 REPL 도구 링크](https://babeljs.io/repl)
            

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판