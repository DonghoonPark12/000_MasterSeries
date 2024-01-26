# 클래스 재 선언
class Car():
    """
    Car Class
    Author : dhpark
    """

    # Class Variable
    price_per_raise = 1.0

    def __init__(self, company, details):
        self._company = company # 인스턴스 변수는 _를 붙여 관리해주면 클래스 변수와 구분이 된다!
        self._details = details

    def __str__(self):
        return 'str : {} - {}'.format(self._company, self._details)

    def __repr__(self):
        return 'repr : {} - {}'.format(self._company, self._details)

    # Instance Method: 객체의 고유한 속성 값을 사용
    # self가 인자로 들어가 있다면, 인스턴스 메소드이다.
    def detail_info(self):
        print('Current Id : {}'.format(id(self)))
        print('Car Detail Info : {} {}'.format(self._company, self._details.get('price')))

    # Instance Method
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
    
# 자동차 인스턴스    
car1 = Car('Bmw', {'color' : 'Black', 'horsepower': 270, 'price': 5000})
car2 = Car('Audi', {'color' : 'Silver', 'horsepower': 300, 'price': 6000})

# 전체 정보
car1.detail_info()
car2.detail_info()
print()

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
Car.raise_price(1.6)
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