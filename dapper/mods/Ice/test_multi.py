from multiprocessing import Process
from datetime import datetime, timedelta
from time import sleep

var0 = 1

def func(x):
    print('enter ',x, datetime.now())
    sleep(5)
    print('exit ',x, datetime.now())
    
print('var0 pre',var0)
if __name__ == '__main__':
    print("In main")
    p0 = Process(target=func,args=(0,))
    p1 = Process(target=func,args=(1,))
    p0.start()
    p1.start()
    p0.join()
    p1.join()
    
print('var0 post',var0)
    
    