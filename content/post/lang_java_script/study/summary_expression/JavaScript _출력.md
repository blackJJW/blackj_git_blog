---
title: "[JavaScript] 출력"
description: ""
date: "2022-05-07T18:00:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

### <u>간단한 표현식 결과 확인</u>

1. 구글 크롬 주소창에 about:blank 입력

    ![Untitled](/images/lang_javascript/study/JavaScript_output/Untitled.png)

2. 단축키 **[ Ctrl + Shift + i ]** 를 눌러 개발자 환경을 띄움

    ![Untitled](/images/lang_javascript/study/JavaScript_output/Untitled1.png)

3. about:blank 에서 **[Console] 탭**을 클릭하여 **구글 크롬 개발자 도구**에 들어간다.

    ![Untitled](/images/lang_javascript/study/JavaScript_output/Untitled2.png)

    ![Untitled](/images/lang_javascript/study/JavaScript_output/Untitled3.png)

- 이곳에서 어떤 값을 입력하면 바로 그 결과를 출력한다.

  ![Untitled](/images/lang_javascript/study/JavaScript_output/Untitled4.png)

### <u>경고창에 출력</u>

- **개발 전용 에디터**를 사용할 때의 출력하는 방법
  - 파일을 생성했을 때 **가장 기본적인 자바스크립트 출력 방법**은 **alert() 함수를 사용**하는 것
      - alert() : 웹 브라우저에 경고창을 띄울 수 있음.

  ```jsx
    <script>
      alert('Hello World!')
    </script>
  ```

- alert() 함수를 입력하고 HTML 페이지를 저장

  ```jsx
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta http-equiv="X-UA-Compatible" content="IE=edge">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
  
      <!-- 자바스크립트 코드 -->
      <script>
          alert('Hello World!')
      </script>
      <!-- 자바스크립트 코드 -->
      
  </head>
  <body>
      
  </body>
  </html>
  ```

- 저장된 HTML 페이지를 실행

  ![Untitled](/images/lang_javascript/study/JavaScript_output/Untitled5.png)

- 페이지를 실행하여 경고창이 열린 것을 확인

  ![Untitled](/images/lang_javascript/study/JavaScript_output/Untitled6.png)

### <u>콘솔에 출력</u>

- **console.log()** 메소드를 사용
    - console.log() 메소드도 alert() 함수처럼 괄호 내부에 입력한 값을 출력

  ```jsx
  <script>
      console.log('Hello World!')
  </script>
  ```

- console.log() 메소드를 입력하고 HTML 페이지를 저장 후 실행

  ```jsx
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta http-equiv="X-UA-Compatible" content="IE=edge">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
  
      <!-- 자바스크립트 코드 -->
      <script>
          console.log('Hello World!')
      </script>
      <!-- 자바스크립트 코드 -->
  
  </head>
  <body>
      
  </body>
  </html>
  ```

- 실행하면 빈 페이지가 나타난다.

  ![Untitled](/images/lang_javascript/study/JavaScript_output/Untitled7.png)

- 단축키 **[ Ctrl + Shift + i ]** 또는 **[ F12 ]키** 를 눌러 개발자 도구를 띄운다.
- 그 다음 **[ Console ]** 탭을 클릭한다.

  ![Untitled](/images/lang_javascript/study/JavaScript_output/Untitled8.png)

- 입력한 결과가 출력된 것을 확인

  ![Untitled](/images/lang_javascript/study/JavaScript_output/Untitled9.png)

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판