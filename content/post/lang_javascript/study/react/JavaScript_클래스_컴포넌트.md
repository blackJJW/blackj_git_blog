---
title: "[JavaScript][Library][React] 클래스 컴포넌트"
description: ""
date: "2022-06-24T17:30:45+09:00"
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

- h1, h2, img 태그 등 **HTML 표준에 포함된 태그로 컴포넌트를 만들 수 있지만**, **사용자가 직접 클래스 또는 함수를 이용해 컴포넌트를 만들 수도 있다**.
- **함수 컴포넌트** : **함수**로 만드는 컴포넌트
- **클래스 컴포넌트** : **클래스**로 만드는 컴포넌트
    
    ```jsx
    class 컴포넌트_이름 extends React.Component {
    	render () {
    		return <h1>출력할 것</h1>
    	}
    }
    ```
    
    - **React.Component 클래스의 상속**을 받아야 컴포넌트로 동작할 수 있게 하는 속성과 메소드를 받을 수 있음
    - **React.Component 클래스**는 화면에 무언가를 출력할 때 render() 메소드를 호출
        - 이를 **오버라이드해서 원하는 것을 출력**

- 루트 컴포넌트 출력을 클래스 컴포넌트로 구현
    
    ```jsx
    <script type="text/babel">
    	// 애플리케이션 클래스 생성
    	class App extends React.Component{ // React.Component를 상속
    		render() {
    			return <h1>리액트 기본</h1>
    		}
    	}
    	// 출력
    	const container = document.getElementById('root')
    	ReactDOM.render(<App />, container)
    									// App 컴포넌트로 변경
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_클래스_컴포넌트/Untitled.png)
    
- 클래스 컴포넌트를 사용하면 **클래스 메소드 내부에서 this.props 속성을 사용가능**
- 컴포넌트 속성 사용
    
    ```jsx
    <script type="text/babel">
    	// 애플리케이션 클래스 생성
    	class App extends React.Component{
    		render (){
    			return <div>
    				<h1>{this.props.name} Hello!!!</h1> // 컴포넌트 속성으로
    				<img src={this.props.imgUrl} />  // 전달된 값을 사용
    			</div>
    		}
    	}
    	
    	// 출력
    	const container = document.getElementById('root')
    	ReactDOM.render(<App name="John" imgUrl="https://placedog.net/400/200" />, container)
                            // 속성을 지정
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_클래스_컴포넌트/Untitled%201.png)
    

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판