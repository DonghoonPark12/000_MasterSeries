#===================================================================================#
# 프로세스 spawning
# - Spawning은 부모 프로세스가 OS가 요청하여 자식 프로세스를 만들어 내는 과정
#===================================================================================#
# import multiprocessing as mp
# import time

# def worker():
#     proc = mp.current_process()
#     print(proc.name)
#     print(proc.pid)
#     time.sleep(5)
#     print("SubProcess End")

# if __name__ == "__main__":
#     # main process
#     proc = mp.current_process()
#     print(proc.name)
#     print(proc.pid)
    
#     # process spawning
#     p = mp.Process(name="SubProcess", target=worker)
#     p.start()

#     print("MainProcess End")
#===================================================================================#
