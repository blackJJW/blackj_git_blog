---
title: "[JavaScript] private 속성과 메소드"
description: ""
date: "2022-06-20T18:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"


---
<!--more-->

- 사용자가 음수 길이를 입력한 경우
    
    ```jsx
    <script>
        // 정사각형 
        class Square {
            constructor (length){
                this.length = length
            }
            getPerimeter(){ return 4 * this.length }
            getArea(){ return this.length * this.length }
        }
    
        // 클래스 사용
        const square = new Square(-10) // 길이에 음수를 넣어서 사용
        console.log(`정사각형의 둘레 : ${square.getPerimeter()}`)
        console.log(`정사각형의 넓이 : ${square.getArea()}`)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_private_속성과_메소드/Untitled.png)
    
    - 길이는 음수가 나올 수 없는 값
    - **이러한 일을 막아야 함**

- **조건문을 활용**, 0 이하의 경우 예외를 발생시켜 클래스의 **사용자에게 할 수 없다고 인지**시킨다.
- 길이에 음수가 들어가지 않게 수정
    
    ```jsx
    <script>
        // 정사각형 
        class Square {
            constructor (length){
                if (length <= 0) {
                    throw '길이는 0보다 커야 된다.'
                    // throw 키워드를 사용하여 강제로 오류를 발생
                }
                this.length = length
            }
            getPerimeter(){ return 4 * this.length }
            getArea(){ return this.length * this.length }
        }
    
        // 클래스 사용
        const square = new Square(-10) // 길이에 음수를 넣어서 사용
        console.log(`정사각형의 둘레 : ${square.getPerimeter()}`)
        console.log(`정사각형의 넓이 : ${square.getArea()}`)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_private_속성과_메소드/Untitled%201.png)
    
    - 하지만 생성자로 객체를 생성한 이후에 사용자가 length 속성을 변경하는 것을 막을 수는 없다.
        - 사용자의 잘못된 사용 예
        
        ```jsx
        // 클래스 사용
        const square = new Square(10)
        square.length = -10 // 이렇게 음수를 지정하는 것은 막을 수 없다.
        console.log(`정사각형의 둘레 : ${square.getPerimeter()}`)
        console.log(`정사각형의 넓이 : ${square.getArea()}`)
        ```
        
- 클래스 사용자가 클래스 속성(또는 메소드)을 **의도하지 않는 방향으로 사용하는 것을 막아** 클래스의 **안정성을 확보**하기 위해 나온 문법이 **private 속성**과 **메소드**
    
    ```jsx
    class 클래스_이름 {
        #속성_이름
        #메소드_이름 () {
    
        }
    }
    ```
    
    - 속성과 메소드 **이름 앞에 ‘#’을 붙인다**.
    - ‘#’이 붙어있는 속성과 메소드는 모두 **private 속성과 메소드**가 된다.
    - **주의할 점**
        - private 속성은 **사용하기 전에 미리 외부에 어떤 속성을 private 속성으로 사용하겠다고 선언**해주어야 한다.

- private 속성 사용 - 1
    
    ```jsx
    <script>
        // 사각형 클래스
        class Square {
            #length // 이 위치에 해당 속성을 private 속성으로
                            // 사용하겠다고 미리 선언
    		
            constructor (length){
                if (length <= 0) {
                    throw '길이는 0보다 커야 된다.'
                }
                this.#length = length
            }
            getPerimeter(){ return 4 * this.#length }
            getArea(){ return this.#length * this.#length }
        }
    
        // 클래스 사용
        const square = new Square(10)
        console.log(`정사각형의 둘레 : ${square.getPerimeter()}`)
        console.log(`정사각형의 넓이 : ${square.getArea()}`)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_private_속성과_메소드/Untitled%202.png)
    
    - private 속성으로 변경하면 **클래스 외부에서는 해당 속성에 접근 불가**

- ex) square 객체의 length 속성 변경 시도
    - 변경해도 클래스 내부에서 **사용하고 있는 속성은 #length 속성**이지 length 속성이 아니므로 **결과에 영향을 주지 않는다.**
- private 속성 사용 - 2
    
    ```jsx
    <script>
        // 사각형 클래스
        class Square {
            #length // 이 위치에 해당 속성을 private 속성으로
                            // 사용하겠다고 미리 선언
    		
            constructor (length){
                if (length <= 0) {
                    throw '길이는 0보다 커야 된다.'
                }
                this.#length = length
            }
            getPerimeter(){ return 4 * this.#length }
            getArea(){ return this.#length * this.#length }
        }
    
        // 클래스 사용
        const square = new Square(10)
    
        square.length = -10 // 클래스 내부의 length 속성을 변경 시도
    
        console.log(`정사각형의 둘레 : ${square.getPerimeter()}`)
        console.log(`정사각형의 넓이 : ${square.getArea()}`)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_private_속성과_메소드/Untitled%203.png)
    
- \#length 속성을 사용하면 오류가 발생
- private 속성 사용 - 3
    
    ```jsx
    <script>
        // 사각형 클래스
        class Square {
            #length // 이 위치에 해당 속성을 private 속성으로
                            // 사용하겠다고 미리 선언
    		
            constructor (length){
                if (length <= 0) {
                    throw '길이는 0보다 커야 된다.'
                }
                this.#length = length
            }
            getPerimeter(){ return 4 * this.#length }
            getArea(){ return this.#length * this.#length }
        }
    
        // 클래스 사용
        const square = new Square(10)
    
        square.#length = -10 // 클래스 내부의 #length 속성을 변경 시도
    
        console.log(`정사각형의 둘레 : ${square.getPerimeter()}`)
        console.log(`정사각형의 넓이 : ${square.getArea()}`)
    </script>
    ```
    
    ![Untitled](/images/lang_javascript/study_2/JavaScript_private_속성과_메소드/Untitled%204.png)
    

- private 속성은 클래스 외부에서는 접근할 수 없으므로 클래스 사용자가 클래스를 잘못 사용하는 문제를 줄일 수 있다.

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판