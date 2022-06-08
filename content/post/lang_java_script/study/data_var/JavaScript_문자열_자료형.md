---
title: "[JavaScript] 문자열 자료형"
description: ""
date: "2022-05-07T18:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
 

---
<!--more-->

- 자바스크립트에서는 문자가 하나든 여러 개든 모두 **문자열(string) 자료형**이라고 한다.

- **함수**와 **메소드**
    - **특정 기능을 동작**시키도록 작성된 **코드의 집합**
    - **메소드** : **클래스**가 가지고 있는 함수
    - **함수** : **메소드를 포괄**하는 의미

### <u>문자열 만들기</u>

1. **큰따옴표**(**” ”**)를 사용
2. **작은따옴표**(**’ ’**)를 사용

    ![Untitled](/images/lang_javascript/study/JavaScript_문자열_자료형/Untitled.png)

- 큰따옴표와 작은따옴표 모두 사용하여 문자열 자료형을 만들 수 있지만, **따옴표의 종류는 일관되게 사용하는 것이 좋다.**
- **내부에 작은따옴표**를 사용하려면 **외부에 큰따옴표**를 사용해야한다.
    - 반대로 **내부에 큰따옴표**를 사용하려면 **외부에 작은따옴표**를 사용해야한다.

  ![Untitled](/images/lang_javascript/study/JavaScript_문자열_자료형/Untitled%201.png)

- 따옴표를 문자 그대로 사용하고 싶다면 따옴표 앞에 특수한 기능을 하는 **이스케이프 문자**(**\\**)를 사용하면 된다.

  ![Untitled](/images/lang_javascript/study/JavaScript_문자열_자료형/Untitled%202.png)

- 이스케이프 문자의 여러 가지 특수 기능
    - **\n** : 줄바꿈
    - **\t** : 탭
    - **\\\\** : 역슬래시(\\)

  ```jsx
  <script>
      alert('Hello\nWorld!')
  </script>
  ```

  ![Untitled](/images/lang_javascript/study/JavaScript_문자열_자료형/Untitled%203.png)

  ```jsx
  <script>
      alert('Hello \t World!')
  </script>
  ```

  ![Untitled](/images/lang_javascript/study/JavaScript_문자열_자료형/Untitled%204.png)

  ```jsx
  <script>
    alert(" \ , \\ \\ \\")
  </script>
  ```

  ![Untitled](/images/lang_javascript/study/JavaScript_문자열_자료형/Untitled%205.png)

  ![Untitled](/images/lang_javascript/study/JavaScript_문자열_자료형/Untitled%206.png)

- “ \ ” → 역슬래시를 인식하지 않아 빈 문자열이 출력
- “\” → \”를 이스케이프 무자로 인식해서 문자열이 닫히지 않았다고 오류를 낸다.

### <u>문자열 연산자</u>

- ( **+** ) : **문자열 연결 연산자**
    - 문자열 + 문자열

  ![Untitled](/images/lang_javascript/study/JavaScript_문자열_자료형/Untitled%207.png)

- **문자 선택 연산자** : 문자열 **내부의 문자 하나를 선택**
    - 문자열[숫자]
    - 문자열 뒤에 대괄호[…]를 입력하고 괄호 안에 선택할 문자의 위치를 지정
        - **인덱스**(Index) : 위치를 나타내는 숫자

  | A | B | C | D | E |
  | --- | --- | --- | --- | --- |
  | [0] | [1] | [2] | [3] | [4] |

  ![Untitled](/images/lang_javascript/study/JavaScript_문자열_자료형/Untitled%208.png)

### <u>문자열 길이 구하기</u>

- **문자열 길이 :** 문자열 내부의 문자 개수
    - 문자열 길이를 구할 때는 **length 속성**을 사용
    - **빈 문자열도 문자열**

  ![Untitled](/images/lang_javascript/study/JavaScript_문자열_자료형/Untitled%209.png)

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판