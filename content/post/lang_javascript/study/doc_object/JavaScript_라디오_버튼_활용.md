---
title: "[JavaScript] 라디오 버튼 활용"
description: ""
date: "2022-06-14T17:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "HTML"


---
<!--more-->

- 여러 선택지 중 **하나만 선택 가능**
- **checked 속성 사용**
- 라디오 버튼 사용
    
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <script>
            document.addEventListener('DOMContentLoaded', () => {
                // 문서 객체 추출
                const output = document.querySelector('#output');
                const radios = document.querySelectorAll('[name=lang]');
    
                // 모든 라디오 버튼
                radios.forEach((radio) => {
                    // 이벤트 연결
                    radio.addEventListener('change', (event) => {
                        const current = event.currentTarget;
                        if(current.checked){
                            output.textContent = `프로그래밍 언어 : ${current.value}`;
                        }
                    });
                });
            });
        </script>
    </head>
    <body>
        <h3># 프로그래밍 언어를 선택</h3>
        <input type="radio" name="lang" value="Python">
        <span>Python</span>
        <input type="radio" name="lang" value="Java">
        <span>Java</span>
        <input type="radio" name="lang" value="C++">
        <span>C++</span>
        <input type="radio" name="lang" value="JavaScript">
        <span>JavaScript</span>
        <input type="radio" name="lang" value="HTML">
        <span>HTML</span>
        <hr>
        <h3 id="output"></h3> 
    </body>
    </html>
    ```
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_라디오_버튼_활용/Untitled.png)
    
    ![Untitled](/images/lang_javascript/study_1/JavaScript_라디오_버튼_활용/Untitled%201.png)
    
    - **name 속성이 없는 라디오 버튼**
        - name 속성을 입력하지 않으면 라디오 버튼을 여러개 선택 가능
        - 카테고리 구분 없이 선택 가능
        - **한번 선택하고 나면 선택 취소 불가**
        
        ![Untitled](/images/lang_javascript/study_1/JavaScript_라디오_버튼_활용/Untitled%202.png)
        
        - **라디오 버튼을 사용할 때는 꼭 name 속성과 함께 사용**해야된다.

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판