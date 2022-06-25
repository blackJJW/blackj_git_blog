---
title: "[JavaScript][Library][React] 컴포넌트의 기본적인 속성과 메소드"
description: ""
date: "2022-06-25T12:30:45+09:00"
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

- React.Component 클래스는 여러 속성과 메소드를 제공
- **속성을 변경**하고 **메소드를 오버라이드**하고 **필요한 속성과 메소드를 클래스에 추가**해서 **컴포넌트를 생성**

### 클래스의 메소드 오버라이드

```jsx
class App extends React.Component {
	constructor (props) { // 생성자
		super(props)        // 생성자가 여러 일을 해주므로,
		// 생성자 코드       // super(props)를 사용해 부모 생성자를 호출
	}
	render () {
		// 출력할 것
	}
	componentDidMount () {              // 컴포넌트가 내부적으로 특정 상황에
		// 컴포넌트가 화면에 출력될 때 호출  // 호출하는 메소드
	}                                   // 이런 메소드를
                                      // 라이프사이클 메소드라고 부름
	componentWillUnmount () {
		// 컴포넌트가 화면에서 제거될 때 호출
	}
}
```

- 우리가 변경해서 사용해하는 속성으로는 **state 속성**이 있음
- state 속성에는 **출력할 값을 저장**
    - state **속성 값을 변경**할 때는 반드시 **setState() 메소드를 사용**
    - **setState() 메소드**로 속성의 값을 변경하면 컴포넌트는 **render() 메소드를 호출해서 화면에 변경 사항을 호출**
    
    ```jsx
    // 상태 선언(생성자 선언)
    this.state = { 속성: 값 }
    // 상태 변경(이외의 위치)
    this.setState({ 변경할 속성: 값})
    ```
    

---

- 리액트를 활용한 현재 시간 출력 프로그램
    
    ```jsx
    <script type="text/babel">
    	class App extends React.Component{
    		constructor (props) {
    			super(props)
    			this.state = {      // 시간을 출력할 것이므로
    				time: new Date()  // state 속성에 시간을 저장
    			}
    		}
    		render () {
    			return <h1>{this.state.time.toLocaleTimeString()}</h1>
    									// state 속성에 있는 값을 출력
    		}
    		componentDidMount () {
    			// 컴포넌트가 화면에 출력되었을 때
    			this.timerId = setInterval(() => {
    				this.setState({     // setState() 메소드를 사용해
    					time: new Date()  // 시간을 변경
    				})
    			}, 1000)
    		}
    		componentWillUnmount () {
    			//컴포넌트가 화면에서 제거될 때
    			clearInterval(this.timerId)
    		}
    	}
    	// 출력
    	const container = document.getElementById('root')
    	ReactDOM.render(<App />, container)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_컴포넌트의_기본적인_속성과_메소드/Untitled.png)
    
    - 이 예제는 컴포넌트가 화면에서 제거될 일이 없기 때문에 componentWillUnmount() 메소드를 오버라이드하지 않아도 된다.

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판