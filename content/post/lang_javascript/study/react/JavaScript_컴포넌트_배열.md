---
title: "[JavaScript][Library][React] 컴포넌트 배열"
description: ""
date: "2022-06-25T15:30:45+09:00"
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

- 리액트는 컴포넌트를 요소로 갖는 **배열을 사용해서 한 번에 여러 개의 컴포넌트를 출력 가능**
- 컴포넌트 배열 사용(1)
    
    ```jsx
    <script type="text/babel">
    	// 애플리케이션 클래스 생성
    	class App extends React.Component {
    		render () {
    			const list = [
    				<li>Python</li>,
    				<li>JAVA</li>,
    				<li>C++</li>,
    				<li>JavaScript</li>,
    				<li>HTML</li>
    			]
    			return <ul>{list}</ul>
    		}
    	}
    	// 출력
    	const container = document.getElementById('root')
    	ReactDOM.render(<App />, container)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_컴포넌트_배열/Untitled.png)
    

---

- 일반적으로 **this.state에 값 배열을 만들고 render() 메소드 내부에 map() 메소드를 사용해서 이를 컴포넌트 배열로 변환해서 출력하는 코드를 많이 사용**
- 컴포넌트 배열 사용(2)
    
    ```jsx
    <script type="text/babel">
    	// 애플리케이션 클래스 생성
    	class App extends React.Component {
    		constructor (props) {
    			super(props)
    			this.state = {
    				languages: ['Python', 'JAVA', 'C++', 'JavaScript', 'HTML']
    			}
    		}
    		render () {
    			const list = this.state.languages.map((item) => {
    				return <li>{item}</li>
    			})
    			return <ul>{list}</ul>
    		}
    	}
    	// 출력
    	const container = document.getElementById('root')
    	ReactDOM.render(<App />, container)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_컴포넌트_배열/Untitled%201.png)
    

---

- map() 메소드를 **표현식으로 삽입해서 사용**하는 경우
- 컴포넌트 배열 사용(3)

  ```jsx
  <script type="text/babel">
      // 애플리케이션 클래스 생성
      class App extends React.Component {
          constructor (props) {
              super(props)
              this.state = {
                  languages: ['Python', 'JAVA', 'C++', 'JavaScript', 'HTML']
              }
          }
          render () {
              return <ul> {
                  this.state.languages.map((item) => {
                      return <li>{item}</li>
                  })
              }</ul>
          }
      }
      // 출력
      const container = document.getElementById('root')
      ReactDOM.render(<App />, container)
  </script>
  ```

  ![Untitled](/images/lang_javascript/study_3/JavaScript_컴포넌트_배열/Untitled%202.png)

---

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판