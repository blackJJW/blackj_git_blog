---
title: "[JavaScript][Library][React] 스타일 지정"
description: ""
date: "2022-06-25T14:30:45+09:00"
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

- 스타일을 지정할 때는 **style 속성**에 객체를 지정
    
    ```jsx
    render () {
    	const style = {}
    	return <h1 style={style}>글자</h1>
    }
    ```
    
    - style 객체에는 **캐멀 케이스**로 속성을 입력
    - 문서 객체 모델 때와 차이점이 있다면 **숫자를 입력할 때 단위를 입력하지 않아도 된다.**
        
        
        | CSS 스타일 속성 이름 | 가능한 형태(1) | 가능한 형태(2) |
        | --- | --- | --- |
        | color: red | { color: ‘red’ } | { ‘color’: ‘red’ } |
        | font-size: 2px | { fontSize: 2 } | { ‘font-size’; 2 } |

---

- 체크 상태에 따라서 스타일 지정
    
    ```jsx
    <script type="text/babel">
    	// 애플리케이션 클래스 생성
    	class App extends React.Component {
    		constructor (props) {
    			super(props)
    			this.state ={
    				checked: false
    			}
    			this.handleClick = this.handleClick.bind(this)
    		}
    		render () {
    			const textStyle = {
    				color: this.state.checked ? 'blue' : 'red'
    				// 체크 되어 있다면 blue, 아니면 red 
    			}
    			return <div>
    				<input type="checkbox" onClick={this.handleClick} />
    				<h1 style={textStyle}>글자</h1>
    			</div>
    		}
    		handleClick (event) {
    			this.setState({                  // 이벤트 객체를 활용해서 
    				checked: event.target.checked  // 체크 상태를 설정
    			})
    		}
    	}
    	// 출력
    	const container = document.getElementById('root')
    	ReactDOM.render(<App />, container)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_스타일_지정/Untitled.png)
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_스타일_지정/Untitled%201.png)
    

---

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판