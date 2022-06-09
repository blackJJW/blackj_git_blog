---
title: "[Oracle][setting] Oracle Database 19c Installation and Setting"
description: ""
date: "2022-04-23T18:30:45+09:00"
thumbnail: ""
categories:
  - "Setting"
tags:
  - "Oracle"
  - "DB"
---
<!--more-->

## 1. Oracle 19c 다운로드

- 다운로드 링크 : [https://www.oracle.com/kr/database/technologies/oracle19c-windows-downloads.html](https://www.oracle.com/kr/database/technologies/oracle19c-windows-downloads.html)

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled.png)

## 2. 드라이브에 새 폴더 생성 및 다운로드 파일 옮기기

- 드라이브 내에 새 폴더 생성(본인은 ‘oracle_db_sql’로 지정했다.)
- 다운로드된 압축파일을 옮긴 후 압축 해제

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_1.png)

## 3. 압축 해제된 폴더로 이동

- setup 실행(관리자 권한)

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_2.png)

## 4. 설치

- 데이터베이스 설치 옵션
    - ‘단일 인스턴스 데이터베이스 생성 및 구성 ‘선택

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_3.png)

- 시스템 클래스
    - ‘데스크톱 클래스’ 선택

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_4.png)

- Oracle 홈 사용자
    - ‘가상 계정 사용’  선택

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_5.png)

- 기본 설치
    - 파일 경로 확인
    - 전역 데이터베이스 이름 : myoracle
    - 비밀번호 입력
    - ‘컨테이너 데이터베이스로 생성’ 체크 해제

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_6.png)

- 필요 조건 검사, 요약 후 제품 설치 진행

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_7.png)

## 5. 설치 완료 후 SQL Plus 관리자 권한 실행

- 계정명 : system
- 비밀번호 입력

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_8.png)

- TABLESPACE 생성
    - `CREATE TABLESPACE myts DATAFILE 'D:\oracle_db_sql\oradata\MYORACLE\myts.dbf' SIZE 100M AUTOEXTEND ON NEXT 5M;`

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_9.png)

- USER 생성
    - `CREATE USER ora_user IDENTIFIED BY jjw DEFAULT TABLESPACE MYTS TEMPORARY TABLESPACE TEMP;`

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_10.png)

- DBA 권한 부여
    - `GRANT DBA TO ora_user;`

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_11.png)

- USER connect
    - `connect ora_user/jjw;`

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_12.png)

- dual table에서 user 출력 확인
    - `select user from dual;`

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_13.png)

- user 출력
    - 'show user;'

![Untitled](/images/setting_images/oracle_database_19c_setting/Untitled_14.PNG)