---
title: "[JavaScript][Library][Lodash] Lodash"
description: ""
date: "2022-06-07T13:30:45+09:00"
thumbnail: ""
categories:
  - "JavaScript"
tags:
  - "JavaScript"
  - "JavaScript Library"
  - "HTML"
  - "Lodash"


---
<!--more-->

- 다른 사람이 만든 **외부 자바스크립트 파일을 사용**하는 것이 가능
- **외부 라이브러리** : 다른 사람들이 만든 **다양한 함수와 클래스를 묶어서 제공**해주는 것

- **유틸리티 라이브러리** : 개발할 때 보조적으로 사용하는 함수들을 제공해주는 라이브러리
    - underscore, Lodash 등 다양한 라이브러리가 존재
    - 최근 많이 사용되는 것이 **Lodash 라이브러리**

### Lodash 라이브러리

1. lodash 라이브러리 다운로드 페이지 접속
    
    [lodash 라이브러리 다운로드](https://lodash.com/)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_Lodash_라이브러리/Untitled.png)
    

1. 다운로드 페이지에서 [**Full Build**]를 **클릭**하면 웹 브라우저에 따라 다음과 같은 2가지 상황이 발생
    1. **곧바로 파일 다운로드가 될 경우** : 다운로드한 파일을 HTML 파일과 같은 위치에 놓고 읽어들인다.
    2. **파일 내용이 출력될 경우** : 마우스 우 클릭을 하고 [다른 이름으로 저장]을 선택한 뒤 HTML 파일과 같은 위치에 놓고 다운로드한다. 

1. 다른 방법이 존재한다. 
    - 다운로드 페이지에서 [**CDN copies**]를 클릭하여 다음 페이지로 이동한다.
        
        [Lodash CDN 링크 페이지](https://www.jsdelivr.com/package/npm/lodash)
        
        ![Untitled](/images/lang_javascript/study/JavaScript_Lodash_라이브러리/Untitled%201.png)
        
        - 하이라이트된 부분의 오른쪽에 있는 [Copy to Clipboard] 아이콘을 클릭하여 **URL을 선택**해 **CDN 링크를 복사**한다.
            
            ![Untitled](/images/lang_javascript/study/JavaScript_Lodash_라이브러리/Untitled%202.png)
            
    
    1. 복사한 링크를 script 태그의 src 속성에 입력하면 라이브러리를 읽어들일 수 있다. 
        
        ```html
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Document</title>
            <script src="https://cdn.jsdelivr.net/npm/lodash@4.17.21/lodash.min.js">
        		</script>
        </head>
        <body>
            
        </body>
        </html>
        ```
        
    
- 더 자세한 정보는 다음 링크를 참조
    
    [Lodash 라이브러리 문서](https://lodash.com/docs/4.17.15)
    
    ![Untitled](/images/lang_javascript/study/JavaScript_Lodash_라이브러리/Untitled%203.png)
    
    - 수많은 함수를의 사용방법과 결과를 볼 수 있다.
- Lodash 라이브러리 안에 ‘ **_** ’를 포함한 **메소드**를 갖고 있는 것을 알 수 있음
    - 자바스크립트는 ‘**_**’와 ‘**$**’ 기호를 **식별자로 사용 가능**한데, 이때 ‘_’기호를 사용하여 식별자를 생성한 것이다.
- _.sortBy() 메소드
    - 배열을 어떤 것으로 **정렬할지 지정**하면, 지정한 것을 기반으로 **배열을 정렬해서 리턴**
    
    ```html
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta http-equiv="X-UA-Compatible" content="IE=edge">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
        <script src="https://cdn.jsdelivr.net/npm/lodash@4.17.21/lodash.min.js">        
        </script>
        <script>
            const people = [{
                "name" : "John",
                "age" : 31,
                "language" : "Python"
            }, {
                "name" : "Amy",
                "age" : 29,
                "language" : "JAVA"
            }, {
                "name" : "Paul",
                "age" : 25,
                "language" : "C++"
            }, {
                "name" : "Jack",
                "age" : 33,
                "language" : "C"
            }, {
                "name" : "James",
                "age" : 23,
                "language" : "JavaScript"
            }]
    
            const output = _.sortBy(people, (person) => person.age)
            console.log(JSON.stringify(output, null, 2))
        </script>
    </head>
    <body>
        
    </body>
    </html>
    ```
    
    - 나이(age)으로 정렬한 결과를 출력
        
        ![Untitled](/images/lang_javascript/study/JavaScript_Lodash_라이브러리/Untitled%204.png)
        

- 이외에도 다양한 기능이 있음

## References

- 혼자 공부하는 자바스크립트 - 윤인성 지음, 한빛미디어 출판
- [https://lodash.com/](https://lodash.com/)
- [https://www.jsdelivr.com/package/npm/lodash](https://www.jsdelivr.com/package/npm/lodash)
- [https://lodash.com/docs/4.17.15](https://lodash.com/docs/4.17.15)