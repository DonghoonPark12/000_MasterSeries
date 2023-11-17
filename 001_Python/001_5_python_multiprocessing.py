from multiprocessing import Pool

# def f(x):
#     return x ** 2

# if __name__ == '__main__':
#     with Pool(5) as p:
#         print(p.map(f, [1, 2, 3]))

#---------------------------------------------------------------------------------------------------------#
# def info(title):
#     print('title:', title)
#     print('module name:', __name__)
#     print('parent process:', os.getppid())
#     print('process id:', os.getpid())

# def f(name):
#     info('function f')
#     print('hello', name)

# if __name__ == '__main__':
#     info('main line')
#     p = Process(target=f, args=('bob',))
#     p.start()
#     p.join()
#---------------------------------------------------------------------------------------------------------#
# import multiprocessing as mp

# def foo(q):
#     q.put('hello')

# if __name__ == '__main__':
#     mp.set_start_method('spawn') # set_start_method() 는 프로그램에서 한 번만 사용되어야 한다.
#     q = mp.Queue()
#     p = mp.Process(target=foo, args=(q,))
#     p.start()
#     print(q.get())
#     p.join()
#---------------------------------------------------------------------------------------------------------#
# import multiprocessing as mp

# def foo(q):
#     q.put('hello2')

# if __name__ == '__main__':
#     ctx = mp.get_context('spawn')
#     q = ctx.Queue()
#     p = ctx.Process(target=foo, args=(q,))
#     p.start()
#     print(q.get())
#     p.join()
#---------------------------------------------------------------------------------------------------------#
# import time, os
# import multiprocessing as mp
# from multiprocessing import Pool
# print(mp.cpu_count(), mp.current_process().name)

# def work_func(x):
#     print("work_func:", x, "PID", os.getpid())
#     time.sleep(1)
#     return x ** 5

# if __name__ == "__main__":
#     start = int(time.time())
#     cpu = None
#     pool = Pool(cpu)
#     print(pool.map(work_func, range(0, 12)))

#     print("***run time(sec) :", int(time.time()) - start)
#---------------------------------------------------------------------------------------------------------#
import multiprocessing as mp
import time
import random
import sys

def calculate(func, args):
    result = func(*args)
    return '%s says that %s%s = %s' % (
        mp.current_process().name, func.__name__, args, result
        )

def calculatestar(args):
    '''
    args -> (mul, (i, 7))
    이렇게 튜플로 받아서 풀어서 전달하거나,
    아래처럼 함수 내부에서 풀어서 전달한다.
    '''
    #print(*args)
    return calculate(*args) # *args(arguments) : list of arguments 즉, [mul, (i, 7)]

def calculate_rev(args):
    func, arg = args[0], args[1]
    res = func(*arg) # arg가 (0, 1) 의 튜플이기 때문에 (a, b) 인자를 받는 함수에 전달하려면 풀어서 전달해야 한다.
    return '%s says that %s%s = %s' % (mp.current_process().name, func.__name__, args, res)

def mul(a, b):
    time.sleep(0.5 * random.random())
    return a * b

def plus(a, b):
    time.sleep(0.5 * random.random())
    return a + b

if __name__ == '__main__':
    cpu = 4
    num_of_tasks = 10

    with mp.Pool(cpu) as pool:
        TASKS = [(mul, (i, 7)) for i in range(num_of_tasks)] + [(plus, (i, 8)) for i in range(num_of_tasks)] 
        '''
        [ (mul, (0, 7)), (mul, (1, 7)), ..., (mul, (9, 7)), (plut, (0, 8)), ..., (plus, (9, 8)) ]
        '''
        #results = pool.map(calculatestar, TASKS)
        map_result = pool.map(calculate_rev, TASKS) # TASK에는 일괄 적용할 인자가 들어간다!
        async_result = [pool.apply_async(calculate, t) for t in TASKS]
        imap_result = pool.imap(calculate_rev, TASKS)
        imap_unordered_result = pool.imap_unordered(calculatestar, TASKS)

        print('Ordered results - map():')
        for r in map_result:
            print('\t', r)
        print()

        print('Ordered async_results - apply_async():')
        for r in async_result:
            print('\t', r.get())
        print()

        print('Ordered results - imap():')
        for x in imap_result:
            print('\t', x)
        print()

        print('Unordered results - imap_unordered():')
        for x in imap_unordered_result:
            print('\t', x)
        print()
