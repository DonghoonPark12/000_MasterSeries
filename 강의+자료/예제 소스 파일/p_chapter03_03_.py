# Chapter03-03
# 파이썬 심화
# 데이터 모델(Data Model)
# 참조 : https://docs.python.org/3/reference/datamodel.html
# Namedtuple 실습
# 파이썬의 중요한 핵심 프레임워크 -> 시퀀스(Sequence), 반복(Iterator), 함수(Functions), 클래스(Class)

# 객체 -> 파이썬의 데이터를 추상화
# 모든 객체 -> id, type -> value

# 일반적인 튜플 사용
pt1 = (1.0, 5.0)
pt2 = (2.5, 1.5)

from math import sqrt

l_leng1 = sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2)

print(l_leng1)

from collections import namedtuple

Point = namedtuple('Point', 'x y') # 튜플인데, 딕셔너리와 같이 사용할 수 있다.

# 두 점 선언
pt3 = Point(1.0, 5.0)
pt4 = Point(2.5, 1.5)

print(pt3[0])
print(pt4.x)

l_leng2 = sqrt((pt4.x  - pt3.x) ** 2 + (pt4.y - pt3.y) ** 2)

# 출력
print(l_leng2)
print(l_leng1 == l_leng2)

# 네임드 튜플 선언 방법
Point1 = namedtuple('Point', ['x', 'y'])
Point2 = namedtuple('Point', 'x, y')
Point3 = namedtuple('Point', 'x y')
Point4 = namedtuple('Point', 'x y x class', rename=True) # Default=False

print(Point1, Point2, Point3, Point4)

print()
print()

# Dict to Unpacking
temp_dict = {'x': 75, 'y': 55}

p1 = Point1(x=10, y=35)
p2 = Point2(20, 40)
p3 = Point3(45, y=20)
p4 = Point4(10, 20, 30, 40)
p5 = Point2(**temp_dict)

# 출력
print(p1, p2, p3, p4, p5)

print()
print()

print(p1[0] + p2[1])
print(p1.x + p2.y)

# Unpacking
x, y = p3
print(x+y)

# Rename 테스트
print(p4)

print()
print()

temp = [52, 38]
# _make() : 새로운 객체 생성
p4 = Point1._make(temp) # 리스트를 이용해 namedtuple에 넣을때
print(p4)

# _fields : 필드 네임 확인
print(p1._fields, p2._fields, p3._fields)

# _asdict() : OrderedDict 반환
print(p1._asdict())
print(p1._asdict(), p4._asdict()) # namedtuple to dictionary

# 네임드 튜플 선언
Classes = namedtuple('Classes', ['rank', 'number'])

# 그룹 리스트 선언
numbers = [str(n) for n in range(1, 21)]
ranks = 'A B C D'.split()

students = [Classes(rank, number) for rank in ranks for number in numbers]

print(len(students))
print(students)

#추천
students = [Classes(rank, number)
                for rank in 'A B C D'.split()
                    for number in [str(n) for n in range(1, 21)]]

# 출력
for s in students:
    print(s)
