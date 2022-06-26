---
title: "[JavaScript][Library][React] 여러 개의 컴포넌트 사용"
description: ""
date: "2022-06-25T16:30:45+09:00"
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

- **Item 컴포넌트**를 생성하여 사용
- Item 컴포넌트 만들기
    
    ```jsx
    <script type="text/babel">
    	// 애플리케이션 클래스 생성
    	class App extends React.Component {
    		render () {
    			return <ul>
    				<Item />
    				<Item />
    				<Item />
    			</ul>
    		}
    	}
    	class Item extends React.Component {
    		render () {
    			return <li>Item 컴포넌트</li>
    		}
    	}
    	// 출력
    	const container = document.getElementById('root')
    	ReactDOM.render(<App />, container)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_여러개의_컴포넌트_사용/Untitled.png)
    

---

- App 컴포넌트에서 **Item 컴포넌트로 어떤 데이터를 전달하고 싶을 때는 컴포넌트의 속성을 사용**
- **App 컴포넌트**에서 **Item 컴포넌트**로 **value 속성을 전달**하고, **Item 컴포넌트**에서 **value 속성을 출력**하는 예
    
    ```jsx
    <script type="text/babel">
    	// 애플리케이션 클래스 생성
    	class App extends React.Component {
    		render () {
    			return <ul>
    				<Item value="Item 컴포넌트 1" />
    				<Item value="Item 컴포넌트 2" />
    				<Item value="Item 컴포넌트 3" />
    			</ul>
    		}
    	}
    	class Item extends React.Component {
    		constructor (props) {
    			super(props)
    		}
    		render () {
    			return <li>{this.props.value}</li>
    		}
    	}
    	// 출력
    	const container = document.getElementById('root')
    	ReactDOM.render(<App />, container)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_여러개의_컴포넌트_사용/Untitled%201.png)
    

---

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판