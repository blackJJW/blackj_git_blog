---
title: "[JavaScript] 이벤트 발생 객체"
description: ""
date: "2022-06-13T15:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- 이벤트 내부에서 **문서 객체 변수를 사용**해 문서 객체와 관련된 정보를 추출
    
    ```jsx
    <script>
    	document.addEventListener('DOMContentLoaded', () => {
    		const textarea = document.querySelector('textarea');
    		const h1 = document.querySelector('h1');
    		
    		textarea.addEventListener('keyup', (event) => {
    			const length = textarea.value.length;
    			h1.textContent = `글자 수 : ${length}`;
    		};
    	};
    </script>
    ```
    
    - 상황에 따라서는 이벤트 리스너 내부에서 변수에 접근할 수 없는 경우가 존재
    
    - 다음 코드에서는 listener() 함수 내부에서 textarea 변수에 접근할 수 없어 오류가 발생한다.
    - **이벤트 리스너를 외부로 빼낸 경우**
        
        ```jsx
        <script>
        	const listener = (event) => {
        		const length = textarea.value.length // 현재 블록에서는 textarea 변수 사용불가
        		h1.textContent = `글자 수 : ${length}`;
        	};
        
        	document.addEventListener('DOMContentLoaded', () => {
        																								// 이벤트 리스너가 외부로 분리됨
        		const textarea = document.querySelector('textarea');
        		const h1 = document.querySelector('h1');
        		textarea.addEventListener('keyup', listener);
        	});
        </script>
        ```
        
    
    - 코드의 규모가 커지면 이벤트 리스너를 외부로 분리하는 경우가 많아진다.
    - 이러한 경우, 이벤트를 발생시킨 객체(현재 코드의 textarea)에 접근하는 방법
        - **event.currentTarget 속성을 사용**
            - ( ) => { } 와 function ( ) { } 형태 모두 사용 가능
            
            ```jsx
            <script>
            	const listener = (event) => {
            		const length = event.currentTarget.value.length
                                // event.currentTarget 이 textarea가 됨
            		h1.textContext = `글자 수 : ${length}`;
            	};
            
            	document.addEvenListener('DOMContentLoaded', () => {
            		const textarea = document.querySelector('textarea');
            		const h1 = document.querySelector('h1');
            		textarea.addEventListener('keyup', listener);
            	});
            </script>
            ```
            
        
        - **this 키워드를 사용**
            - 화살표 함수가 아닌 function ( ) { } 형태로 함수를 선언한 경우에 사용
            
            ```jsx
            <script>
            	const listener = (event) {
            		const length = this.value.length
                                    // this가 textarea가 됨
            		h1.textContext = `글자 수 : ${length}`;
            	};
            
            	document.addEvenListener('DOMContentLoaded', () => {
            		const textarea = document.querySelector('textarea');
            		const h1 = document.querySelector('h1');
            		textarea.addEventListener('keyup', listener);
            	});
            </script>
            ```
            
    
    ## References
    
    - 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판