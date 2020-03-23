import os,sys,logging,datetime
test = 1 
# 打印时间的装饰器
def logging_time(func):
    def wrapper(*args, **kwargs):
        start = datetime.datetime.now()
        print("this function <",func.__name__,">is running")
        res = func(*args, **kwargs)
        print("this function <",func.__name__,"> takes time：",datetime.datetime.now()-start)
        return res
    return wrapper

if __name__ == "__main__":
    pass
