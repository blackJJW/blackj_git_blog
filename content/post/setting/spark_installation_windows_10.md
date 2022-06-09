---
title: "[Spark][setting] Spark Installation(Windows 10)"
description: ""
date: "2022-04-23T16:35:45+09:00"
thumbnail: ""
categories:
  - "Setting"
tags:
  - "Spark"

---
<!--more-->

## 자바 설치

- Oracle 회원가입 (로그인 필요)
- 다운로드 파일 : [https://www.oracle.com/java/technologies/javase/javase8u211-later-archive-downloads.html](https://www.oracle.com/java/technologies/javase/javase8u211-later-archive-downloads.html)

- 관리자 권한으로 다운로드 파일 실행.
- Next 버튼을 클릭 후 설치 경로 설정에 관하여 나온다.
    - 초기 설정 : C:\Program Files\Java\jdk1.8.0_301\
    - 이 경로를 변경 → C:\jdk
        - 경로 이름 중에 공백이 있으면 환경 설치 시 문제 발생 가능성 있음.
- 다시 Next 버튼을 클릭 후, 자바 런타일 환경의 폴더의 경로도 변경해준다.
    - 초기 설정 :  C:\Program Files\Java\jre1.8.0_301\
    - 이 경로를 변경 → 해당 드라이브에 jre폴더 생성 → C:\jre 로 설정

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled.png)

(본인은 C드라이브 용량의 문제로 D드라이브에 저장을 했다.)

## Spark 설치

### (1) Spark 다운로드

- 다운로드 주소 : [https://spark.apache.org/downloads.html](https://spark.apache.org/downloads.html)

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_1.png)

(Spark 버전을 확인 후 기억해 둔다.)

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_2.png)

표시된 곳을 선택하여 다운로드를 실행한다. 

### (2) WinRAR 프로그램 다운로드

- 다운로드한 Spark가 .tgz 형식의 압축파일이므로 WinRAR을 설치.
    - 다운로드 파일 :  [https://www.rarlab.com/download.htm](https://www.rarlab.com/download.htm)
    
    ![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_3.png)
    
    컴퓨터 환경에 맞는 파일을 선택한 후 다운로드를 진행한 후, 설치를 진행한다.
    

### (3) spark 폴더 생성 및 파일 이동

- 드라이브 하단에 spark 폴더 생성

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_4.png)

- spark-3.3.0-bin-hadoop3.2 폴더 내의 모든 파일을 복사

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_5.png)

### (4) conf 폴더 내의 있는 [log4j.properties](http://log4j.properties) 파일 수정

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_6.png)

해당 파일을 메모장으로 연다. 

- INFO 를 ERROR로 수정

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_7.png)

### (5) winutils 설치

- 스파크가 윈도우 로컬 컴퓨터가 Hadoop으로 착각하게 만들 프로그램 필요
    - 다운로드 파일 : https://github.com/cdarlint/winutils
    - spark의 버전에 맞추어 다운로드한다.
- 설치할 드라이브에 winutils 폴더를 생성 후, 그 폴더 내에 bin 폴더를 생성한다.
- bin 폴더 내에 다운로드한 파일을 저장한다.

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_8.png)

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_9.png)

- 이 파일이 Spark 실해 시, 오류 없이 실행 될 수 있도록, 사용 권한을 얻도록 한다.
    - 관리자 권한으로  cmd 실행
    - 드라이브 하단에 tmp폴더와 그 내부에 hive 폴더를 생성
    
    ![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_10.png)
    
    ![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_11.png)
    
    위 과정을 수행한다.
    

## 환경변수 설정

- 시스템 환경변수 설정
    - 사용자 변수 - 새로 만들기
    
    ![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_12.png)
    

- SPARK_HOME 환경 변수

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_13.png)

- JAVA_HOME 환경 변수

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_14.png)

- HADOOP_HOME 환경 설정

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_15.png)

### Path 변수 편집

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_16.png)

아래 코드를 추가

- %SPARK_HOME%\bin
- %JAVA_HOME%\bin

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_17.png)

## 파이썬 환경설정

- Python 환경설정

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_18.png)

- 또는 파이썬의 경로를 직접 설정해준다.

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_19.png)

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_20.png)

## 스파크 테스트

- cmd를 열고 spark 파일로 경로를 설정한다.

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_21.png)

- pyspark 를 입력

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_22.png)

- 아래 코드를 입력 후 실행하여 정상적으로 작동하는 지 확인한다.

rd = sc.textFile(”README.md”)

rd.count()

![Untitled](/images/setting_images/spark_installation_windows_10/Untitled_23.png)