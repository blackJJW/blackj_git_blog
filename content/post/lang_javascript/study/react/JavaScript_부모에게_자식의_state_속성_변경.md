---
title: "[JavaScript][Library][React] 부모에게 자식의 state 속성 변경"
description: ""
date: "2022-06-26T19:00:45+09:00"
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

- 부모 컨포넌트에서 자식 컴포넌트로 어떤 데이터를 전달할 때는 **속성(this.props)을 사용**
- 부모 컴포넌트에서 자식으로 어떤 데이터를 전달한 뒤 **화면 내용을 변경할 때도 속성(this.prop)을 사용**

---

- 부모 컴포넌트에서 시간을 구하고, 이를 속성을 통해 자식 컴포넌트에게 전달하는 예
    - 중요 부분 : **componentDidUpdate( ) 메소드**
    
    ```jsx
    <script type="text/babel">
          // 애플리케이션 클래스 생성하기
          class App extends React.Component {
            constructor (props) {
              super(props)
              this.state = {
                time: new Date()
              }
            }
            componentDidMount () {
              // 컴포넌트가 화면에 출력되었을 때
              this.timerId = setInterval(() => {
                this.setState({
                  time: new Date()
                })
              }, 1000)
            }
            componentWillUnmount () {
              // 컴포넌트가 화면에서 제거될 때
              clearInterval(this.timerId)
            }
            render () {
              return <ul>
                <Item value={this.state.time.toLocaleString()} />
                <Item value={this.state.time.toLocaleString()} />
                <Item value={this.state.time.toLocaleString()} />
              </ul>
            }
          }
    
          class Item extends React.Component {
            constructor (props) {
              super(props)
              this.state = {
                value: props.value
              }
            }
            componentDidUpdate (prevProps) {              // 자식 컴포넌트에서
              if (prevProps.value !== this.props.value) { // 적용할 때 사용하는 
                this.setState({                           // 코드
                  value: this.props.value
                })
              }
            }
            render () {
              return <li>{this.state.value}</li>
            }   
          }
          // 출력
          const container = document.getElementById('root')
          ReactDOM.render(<App />, container)
        </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_3/JavaScript_부모에게_자식의_state_속성_변경/Untitled.png)
    
    - **componentDidUpdate() 메소드**는 컴포넌트에 **변경이 발생했을 때 호출**되는 메소드
        - 이를 **오버라이드해서 사용**
        - componentDidUpdate() 메소드는 **매개 변수로 변경 이전의 속성(prevProps)이 들어온다.**
        - 이 **속성 값과 현재 속성 값을 비교**해서 **변경이 있는 경우(다른 경우)에만 setState() 메소드를 호출해서 화면에 변경 사항을 출력**
        - componentDidUpdate() 메소드 부분이 없으면 시간은 변하지 않음.
    - render() 메소드는 단순하게 **컴포넌트를 조합해서 문서 객체를 만든 뒤 화면에 출력하는 메소드**가 아님
        - **내부적으로 쓸데없는 변경 등을 막아 애플리케이션의 성능을 높일 수 있게 다양한 처리**를 해줌

---

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판