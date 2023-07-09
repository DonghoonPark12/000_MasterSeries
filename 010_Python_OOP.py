"""
배운 것들
 - 클래스 변수
 - 클래스 매소드
 - 스태틱 매소드
 - 각종 매타 함수들
"""

class Car():
    """
    Car Class
    Author: dhpark
    """
    
    # 클래스 변수는 모든 인스턴스가 공유한다!!
    car_count = 0

    def __init__(self, company, details):
        self._company = company # 인스턴스 변수는 _를 붙여 관리해주면 클래스 변수와 구분이 된다!
        self._details = details
        Car.car_count += 1 # 전체 인스턴스 갯수를 관리할 때는 클래스 변수를 활용한다!

    def __str__(self):
        return 'str : {} - {}'.format(self._company, self._details)

    def __repr__(self):
        return 'repr : {} - {}'.format(self._company, self._details)
    
    def detail_info(self):
        #print(self) # 인스턴스와 주소 값이 나온다.
        print('Current ID: {}'.format(id(self)))
        print('Car Detail Info : {} {}'.format(self._company, self._details.get('price')))

    def __del__(self):
        Car.car_count -= 1

    def get_price(self):
        return 'Before Car Price -> company : {}, price : {}' \
                .format(self._company, self._details.get('price'))

    # Instance Method
    def get_price_culc(self):
        return 'After Car Price -> company : {}, price : {}'.format(self._company, self._details.get('price') * Car.price_per_raise)

    """
    클래스 매소드는 인스턴스 한곳에 정의하면, 모든 인스턴스에 공통 적용?
    """
    @classmethod
    def raise_price(cls, per):
        if per <=1:
            print('Please Enter 1 or More')
            return
        cls.price_per_raise = per
        return 'Succeed! price increased.'

    # Static Method
    @staticmethod
    def is_bmw(inst):
        if inst._company == 'Bmw':
            return 'OK! This car is {}.'.format(inst._company)
        return 'Sorry. This car is not Bmw.'
    
car1 = Car('Ferrari', {'color' : 'White', 'horsepower': 400, 'price': 8000})
car2 = Car('Bmw', {'color' : 'Black', 'horsepower': 270, 'price': 5000})
car3 = Car('Audi', {'color' : 'Silver', 'horsepower': 300, 'price': 6000})

# 1) 인스턴스 고유값 부여
print(id(car1))
print(id(car2))
print(id(car3))

print(isinstance(car1, Car))
print(isinstance(car2, Car))
print(isinstance(car3, Car))

# 2) 매소드 및 속성 호출
print(dir(car1))    # 모든 매소드를 리스트로 보여준다.
print(dir(car2))    # 클래스 안의 모든 매타 정보 프린트
print(dir(car3))

print(car1.__dict__) # 클래스 맴버변수가 Key로 저장된다.
print(car2.__dict__) # 클래스 안에 어떤 값이 담겨 있는지 보여준다.
print(car3.__dict__) # 사용자가 정의한 속성만 보여준다.

# 3) 주석, 설명 호출
print(Car.__doc__)

car1.detail_info()
car2.detail_info()
car3.detail_info()

# 4) 클래스로 부터 함수 호출 시
Car.detail_info(car1) # 클래스로 부터 함수를 호출 할때는 인스턴스를 명시해준다.

print(id(car1.__class__), id(car2.__class__), id(car3.__class__)) # 하나의 클래스로 부터 왔으니 모두 동일

# 5) __repr__, __str__
car_list = []
car_list.append(car1)
car_list.append(car2)
car_list.append(car3)

for x in car_list:
    print(repr(x)) # __repr__ 호출
    print(x)       # __str__ 호출


# 6) 클래스 변수 특성 (★)
# del car1
# del car2

#print(car1.car_count) # 네임스페이스에는 없지만, 출력은 된다.
#print(car2.car_count) # 네임스페이스에는 없지만, 출력은 된다.
print(car3.car_count) # 네임스페이스에는 없지만, 출력은 된다.
print(Car.car_count)

'''
인스턴스 네임스페이스에 없으면 상위에서 검색
즉, 동일한 이름으로 변수 생성 가능(인스턴스 검색 후 -> 상위(클래스 변수, 부모 클래스 변수))
'''

#===============================================================================#
# 가격 정보(인상 전)
print(car1._details.get('price')) # 이렇게 내부 속성 변수의 이름을 직접 등장 시키는 것은 옳지 못하다
print(car1._details['horsepower'])

print(car1.get_price())
print(car2.get_price())
print()

# 가격 인상(클래스 메소드 미사용)
Car.price_per_raise = 1.2

# 가격 정보(인상 후)
print(car1.get_price_culc())
print(car2.get_price_culc())

# 가격 인상(클래스 메소드 사용)
Car.raise_price(1.6) # 클래스로 부터 호출하면, 첫번째 인자로 cls가 들어가는 클래스 매소드 호출된다.
print()

# 가격 정보(인상 후 : 클래스메소드)
print(car1.get_price_culc())
print(car2.get_price_culc())
print()

# Bmw 여부(스테이틱 메소드 미사용)
def is_bmw(inst):
    if inst._company == 'Bmw':
        return 'OK! This car is {}.'.format(inst._company)
    return 'Sorry. This car is not Bmw.'

# 별도의 메소드 작성 후 호출
print(is_bmw(car1))
print(is_bmw(car2))
print()

# Bmw 여부(스테이틱 메소드 사용)
print('Static : ', Car.is_bmw(car1))
print('Static : ', Car.is_bmw(car2))
print()

# print('Static : ', car1.is_bmw(car1))
# print('Static : ', car2.is_bmw(car2))