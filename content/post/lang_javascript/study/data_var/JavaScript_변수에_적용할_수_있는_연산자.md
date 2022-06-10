---
title: "[JavaScript] 변수에 적용할 수 있는 연산자"
description: ""
date: "2022-05-08T18:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

### <u>복합 대입 연산자</u>

- **대입 연산자와 다른 연산자를 함께 사용하는 연산자**

  | 복합 대입 연산자  | 설명                |  사용 예   |     의미     |
  |:----------:|:------------------|:-------:|:----------:|
  |     +=     | 기존 변수의 값에 값을 더함   | a += 1  | a = a + 1  |
  |     -=     | 기존 변수의 값에 값을 뺌    | a -= 1  | a = a - 1  |
  |     *=     | 기존 변수의 값에 값을 곱함   | a *= 1  | a = a * 1  |
  |     /=     | 기존 변수의 값에 값을 나눔   | a /= 1  | a = a / 1  |
  |     %=     | 기존 변수의 값에 나머지를 구함 | a %= 1  | a = a % 1  |

- 복합 대입 연산자의 활용

  ```html
  <!DOCTYPE html>
  <html lang="en">
  <head>
      <meta charset="UTF-8">
      <meta http-equiv="X-UA-Compatible" content="IE=edge">
      <meta name="viewport" content="width=device-width, initial-scale=1.0">
      <title>Document</title>
      <script>
          let list = '' // 변수를 선언
  
          list += '<ul>'  // ul 태그는 Unordered List의 약자, HTML 태그
                          // 순서가 없는 목록을 의미
          list += ' <li>Hello</li>'   // li 태그는 List Item의 약자, HTML 태그
          list += ' <li>JavaScript...!</li>' // ul 태그 내부에서 사용
                                              // 실질적인 목록 내용
          list += '</ul>'
  
          document.write(list) // 문서에 출력
      </script>
  </head>
  <body>
      
  </body>
  </html>
  ```
  
  ![Untitled](/images/lang_javascript/study/JavaScript_변수에_적용할_수_있는_연산자/Untitled.png)

- 복합 대입 연산자를 사용하지 않은 경우 - 1

  ```jsx
  <script>
      const list = '<ul><li>Hello</li><li>JavaScript...!</li></ul>'
      // 상수를 선언
      
      document.write(list)
      // 문서에 출력
  </script>
  ```

- 복합 대입 연산자를 사용하지 않은 경우 - 2

  ```jsx
  <script>
      // 상수를 사용하지 않고 문서에 출력
      document.write('<ul><li>Hello</li><li>JavaScript...!</li></ul>')
  </script>
  ```

### <u>증감 연산자</u>

- 변수와 함께 사용할 수 있는 증감 연산자
- 복합 대입 연산자를 간략하게 사용한 형태

  | 증감 연산자  | 설명                 |
  |:-------:|:-------------------|
  |  변수++   | 기존 변수 값에 1을 더함(후위) |
  |  ++변수   | 기존 변수 값에 1을 더함(전위) |
  |   변수—   | 기존 변수 값에 1을 뺌(후위)  |
  |   —변수   | 기존 변수 값에 1을 뺌(전위)  |

- 사용 예 - 1

  ```jsx
  <script>
    let a = 10 // 변수를 선언
  
    a++ // 증감자 사용
  
    alert(a) // 출력
  </script>
  ```
  
  ![Untitled](/images/lang_javascript/study/JavaScript_변수에_적용할_수_있는_연산자/Untitled%201.png)

- 사용 예 - 2
    - **후위**(**postfix**) : 해당 문장을 실행 후 값을 연산을 실행

  ```jsx
  <script>
    let a = 10 // 변수를 선언
  
    alert(a++) // 출력
    alert(a++)
    alert(a++)
  </script>
  ```

  ![Untitled](/images/lang_javascript/study/JavaScript_변수에_적용할_수_있는_연산자/Untitled%202.png)
  
  ![Untitled](/images/lang_javascript/study/JavaScript_변수에_적용할_수_있는_연산자/Untitled%203.png)
  
  ![Untitled](/images/lang_javascript/study/JavaScript_변수에_적용할_수_있는_연산자/Untitled%204.png)

- 사용 예 - 3

  ```jsx
  <script>
    let a = 10 // 변수를 선언
  
    alert(a); a += 1 
    alert(a); a += 1
    alert(a); a += 1
  </script>
  ```
  
  ![Untitled](/images/lang_javascript/study/JavaScript_변수에_적용할_수_있는_연산자/Untitled%205.png)
  
  ![Untitled](/images/lang_javascript/study/JavaScript_변수에_적용할_수_있는_연산자/Untitled%206.png)
  
  ![Untitled](/images/lang_javascript/study/JavaScript_변수에_적용할_수_있는_연산자/Untitled%207.png)

- 사용 예 - 4
    - **전위**(**prefix**) : 해당 문장을 실행하기 전 연산을 먼저 실행

  ```jsx
  <script>
    let a = 10 // 변수를 선언
  
    alert(++a) // 출력
    alert(++a)
    alert(++a)
  </script>
  ```

  ![Untitled](/images/lang_javascript/study/JavaScript_변수에_적용할_수_있는_연산자/Untitled%208.png)
  
  ![Untitled](/images/lang_javascript/study/JavaScript_변수에_적용할_수_있는_연산자/Untitled%209.png)
  
  ![Untitled](/images/lang_javascript/study/JavaScript_변수에_적용할_수_있는_연산자/Untitled%2010.png)

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판