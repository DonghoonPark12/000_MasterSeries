class Car():
    """
    Car Class
    Author: dhpark
    """

    # 클래스 변수
    car_count = 0

    def __init__(self, company, details):
        self._company = company
        self._details = details
        Car.car_count += 1
    
    def __str__(self) -> str:
        return 'str : {} - {}'.format(self._company, self._details)

    def __repr__(self) -> str:
        return 

    def detail_info(self):
        print(self) # 인스턴스와 주소 값이 나온다.
        print('Current ID: {}'.format(id(self)))
        print('Car Detail Info : {} {}'.format(self._company, self._details.get('price')))

    def __del__(self):
        Car.car_count -=1
    
car1 = Car('Ferrari', {'color' : 'White', 'horsepower': 400, 'price': 8000})
car2 = Car('Bmw', {'color' : 'Black', 'horsepower': 270, 'price': 5000})
car3 = Car('Audi', {'color' : 'Silver', 'horsepower': 300, 'price': 6000})

print(id(car1))
print(id(car2))
print(id(car3))

# print(dir(car1))    # 모든 매소드를 리스트로 보여준다.
# print(dir(car2))
# print(dir(car3))

print(car1.__dict__) # 사용자가 정의한 속성만 보여준다
print(car2.__dict__)
print(car3.__dict__)

print(isinstance(car1, Car))
print(isinstance(car2, Car))
print(isinstance(car3, Car))

print(Car.__doc__)

car1.detail_info()
car2.detail_info()
car3.detail_info()

print()

print(id(car1.__class__), id(car2.__class__), id(car3.__class__))

Car.detail_info(car1) # 클래스로 부터 함수를 호출 할때는 인스턴스를 명시해준다.


print(car1.__dict__) # 사용자가 정의한 속성만 보여준다
print(car2.__dict__)
print(car3.__dict__)

del car1

#print(car1.car_count) # 네임스페이스에는 없지만, 출력은 된다.
print(car2.car_count) # 네임스페이스에는 없지만, 출력은 된다.
print(car3.car_count) # 네임스페이스에는 없지만, 출력은 된다.

# print(dir(car1))    # 모든 매소드를 리스트로 보여준다. 클래스 변수까지 포함된다.
# print(dir(car2))
# print(dir(car3))

'''
인스턴스 네임스페이스에 없으면 상위에서 검색
즉, 동일한 이름으로 변수 생성 가능(인스턴스 검색 후 -> 상위(클래스 변수, 부모 클래스 변수))
'''