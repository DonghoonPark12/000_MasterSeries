# def coroutine1():
#     print('>>> coroutine started.')
#     i = yield
#     print('>>> coroutine received : {}'.format(i))

# # 메인 루틴, 제너레이터 선언
# cr1 = coroutine1()

# print(cr1, type(cr1))

#next(cr1)

#cr1.send # 아무 것도 전달하지 않으면 None이 전달된다.
#cr1.send(100) # 데이터를 메인 루틴 -> 서브 루틴 전달
#next(cr1)

print('#' * 10)

def coroutine2(x):
    print(">>> coroutine started : {}".format(x))
    y = yield x
    print(">>> coroutine received : {}".format(y))
    z = yield x + y
    print(">>> coroutine received : {}".format(z))

cr3 = coroutine2(10)
print(next(cr3))
cr3.send(100)   # y가 100을 받음
cr3.send(1000)  # z가 1000을 받음