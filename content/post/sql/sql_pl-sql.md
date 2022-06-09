---
title: "[SQL] SQL과 PL/SQL"
description: ""
date: "2022-04-25T19:33:45+09:00"
thumbnail: ""
categories:
  - "SQL"
tags:
  - "SQL"

---
<!--more-->

## 1. SQL(Structured Query Language)

- 구조화된 질의 언어
- DBMS 상에서 데이터를 읽고 쓰고 삭제하는 등, **데이터를 관리**하기 위한 프로그램 언어
- **집합적 언어** : 데이터를 특정 집합 단위로 분류해 이 단위별로 한 번에 처리하는 언어
    - **절차적 언어**
        
        프로그래밍 순서대로 로직이 처리되는 언어
        
        ex) C, Java
        
- **DDL,** **DML, DCL**
    - **DDL(Data Definition Language)**
        
        데이터베이스 객체를 생성, 삭제, 변경하는 언어
        
        1. **CREATE** : 테이블, 인덱스, 뷰 등 데이터베이스 객체를 생성
        2. **DROP** : 생성된 데이터베이스 객체를 영구히 삭제
        3. **ALTER** : 이미 생성된 데이터베이스 객체를 수정
        4. **TRUNCATE** : 테이블이나 클러스터의 데이터를  통째로 삭제
    - **DML(Data Manipulation Language)**
        
        실제 데이터를 조작하는 언어
        
        1. **SELECT** : 테이블이나 뷰에 있는 데이터를 조회
        2. **INSERT** : 데이터를 신규로 생성
        3. **UPDATE** : 이미 생성된 데이터를 수정
        4. **DELETE** : 데이터를 삭제
        5. **COMMIT** : 트랜잭션 처리. 변경된 데이터를 최종 적용
        6. **ROLLBACK** : 트랜잭션 처리. 변경된 데이터를 적용하지 않고 이전으로 되돌림
    - **DCL(Data Control Language)**
        
        데이터 제어 언어
        
        1. **GRANT** : 사용자에게 특정 권한 부여
        2. **REVOKE** : 사용자에게 부여된 권한을 회수

## 2. PL/SQL

- 절차적 언어
- SQL을 이용해 집합적으로 데이터를 맞게 처리, 이렇게 처리한 SQL을 절차적으로 사용
- 변수 할당, 예외처리, 함수 처리, 프로시저 생성 등이 가능
- DB 서버에 코드가 올라가 컴파일되어 수행되는 것이 특징

## Reference

- "오라클 SQL과 PL/SQL을 다루는 기술", 홍형경 지음, 길벗 출판사