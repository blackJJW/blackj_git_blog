---
title: "[JavaScript][Library][React] 이벤트 연결"
description: ""
date: "2022-06-25T13:30:45+09:00"
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

- 컴포넌트에 **이벤트를 연결**
    1. **메소드를 선언**
    2. 메소드에 **this를 바인드**
        - **이벤트 연결시 반드시 사용**
        - 사용하지 않으면 이벤트 헨들러에서 this를 입력했을 때 undefined가 출력
    3. render() 메소드에서 출력하는 **태그의 이벤트 속성에 메소드를 입력해서 이벤트를 연결**
    
    ```jsx
    class App extends React.Component {
    	constructor (props) {
    		super(props)
    		this.메소드_이름 = this.메소드_이름.bind(this) 
    										// 메소드에 this를 바인드
    	}
    	render () {
    		return <h1 이벤트_이름={this.메소드_이름}></h1>
    							// 이벤트를 연결
    	}
    	메소드_이름(event) {  // 메소드를 선언
    		// 이벤트가 호출될 때 실행할 코드
    	}
    }
    ```
    

---

- 버튼을 클릭할 때 클릭한 횟수를 세는 프로그램
- 이벤트 연결
    
    ```jsx
    <script type="text/babel">
    	// 애플리케이션 클래스 생성
    	class App extends React.Component {
    		constructor (props) {
    			super(props)
    			this.state = { // 클릭한 횟수를 출력한 것을
    				count: 0     // state 속성에 일단 0을 저장
    			}
    			this.countUp = this.countUp.bind(this)
    		}
    		render () {
    			return <div>
    				<h1>클릭한 횟수 : {this.state.count}</h1>
    				<button onClick={this.countUp}>클릭</button>
    							// 대소문자 구분 필수
    			</div>
    		}
    		countUp (event) {
    			this.setState({
    				count: this.state.count + 1
    			})
    		}
    	}
    	// 출력
    	const container = document.getElementById('root')
    	ReactDOM.render(<App />, container)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_이벤트_연결/Untitled.png)
    

---

- 리액트에서 JXS 문법으로 이벤트를 연결할 때는 이름의 대소문자를 확실하게 해야한다.
- 대소문자를 잘못 입력하여 오류발생 시 다음과 같은 오류가 발생한다.
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_이벤트_연결/Untitled%201.png)
    
    - 오류 중에 틀린 곳을 알려준다. 이 알려준 곳을 수정하면 된다.

---

- 리액트 이벤트 이름을 확인할 수 있는 주소
    
    [링크](https://ko.reactjs.org/docs/events.html#clipboard-events)
    

---

- `this.countUp = this.countUp,bind(this)` 를 사용하지 않고 다음과 같이 2가지 형태를 사용하는 방법도 존재
    - 이벤트 연결 : 다른 this 바인드 방법(1)
        
        ```jsx
        <script type="text/babel">
        	// 애플리케이션 클래스 생성
        	class App extends React.Component {
        		constructor (props) {
        			super(props)
        			this.state = {
        				count: 0
        			}
        		}
        		render () {
        			return <div>
        				<h1>클릭한 횟수: {this.state.count}</h1>
        				<button onClick={(e) => this.countUp(e)}>클릭</button>
        			</div>
        		}
        		countUp (event) {
        			this.setState({
        				count: this.state.count + 1
        			})
        		}
        	}
        	// 출력
        	const container = document.getElementById('root')
        	ReactDOM.render(<App />, container)
        </script>
        ```
        
        ![Untitled](/images/lang_javascript/study_3/JavaScript_이벤트_연결/Untitled%202.png)
        
    
    - 이벤트 연결하기: 다른 this 바인드방법(2)
        
        ```jsx
        <script type="text/babel">
        	// 애플리케이션 클래스 생성
        	class App extends React.Component {
        		constructor (props) {
        			super(props)
        			this.state = {
        				count: 0
        			}
        		}
        		render () {
        			return <div>
        				<h1>클릭한 횟수: {this.state.count}</h1>
        				<button onClick={this.countUp}>클릭</button>
        			</div>
        		}
        		countUp = (event) => {
        			this.setState({
        				count: this.state.count + 1
        			})
        		}
        	}
        	// 출력
        	const container = document.getElementById('root')
        	ReactDOM.render(<App />, container)
        </script>
        ```
        
        ![Untitled](/images/lang_javascript/study_3/JavaScript_이벤트_연결/Untitled%203.png)
        
    
    ---
    
    - 입력 양식 사용
    - 입력 양식의 값을 추출할 때는 **이벤트 객체를 활용**
        
        ```jsx
        <script type="text/babel">
        	// 애플리케이션 클래스 생성
        	class App extends React.Component {
        		constructor (props) {
        			super(props)
        			this.state = {
        				text: ''
        			}
        			this.handleChange = this.handleChange.bind(this)
        		}
        		render () {
        			return <div>
        				<input value={this.state.text} onChange={this.handleChange} />
        				<h1>{this.state.text}</h1>
        			</div>
        		}
        		handleChange (event) {
        			this.setState({             // 이벤트가 발생할 때
        				text: event.target.value  // this.state의 text 속성에
        			})                          // 입력 양식의 값을 넣음
        		}
        	}
        	// 출력
        	const container = document.getElementById('root')
        	ReactDOM.render(<App />, container)
        </script>
        ```
        
        ![Untitled](/images/lang_javascript/study_3/JavaScript_이벤트_연결/Untitled%204.png)
        

---

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판