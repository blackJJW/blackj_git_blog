---
title: "[JavaScript][Library][React] 자식에서 부모의 state 속성 변경"
description: ""
date: "2022-06-26T19:30:45+09:00"
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

- **자식 컴포넌트에서 부모 컴포넌트의 상태를 변경할 때는 메소드를 사용**
- **부모 컴포넌트에서 자신(부모)의 속성을 변경하는 메소드를 자식에게 전달한 뒤, 자식에서 이를 호출하게 만드는 것**

---

- 자식에서 부모의 state 속성 변경
    
    ```jsx
    <script type="text/babel">
    	// 애플리케이션 클래스 생성
    	class App extends React.Component {
    		constructor (props) {
    			super(props)
    			this.state = {
    				value: ''
    			}
    			this.changeParent = this.changeParent.bind(this)
    		}
    		render () {
    			return <div>
    				<CustomInput onChange={this.changeParent} />
    				<h1>{this.state.value}</h1>
    			</div>
    		}
    		changeParent (event) {         // 자신의 속성을 변경하는 메소드
    			this.setState({              // 내부에서 this 키워드를 사용했으므로
    				value: event.target.value  // this 바인드 
    			})
    		}
    	}
    	class CustomInput extends React.Component {
    		render () {
    			return <div>
    				<input onChange={this.props.onChange} />
    				// input 태그에 변경 사항이 있을 때,
    				// 부모로부터 전달받은 메소드를 호출
    			</div>
    		}
    	}
    	// 출력
    	const container = document.getElementById('root')
    	ReactDOM.render(<App />, container)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_자식에서_부모의_state_속성_변경/Untitled.png)
    

---

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판