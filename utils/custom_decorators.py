import time
from functools import wraps

def calc_execute_time(show_time=True):
    def deco(func):
        @wraps(func)
        def time_measurement(*args, **kwargs):
            st = time.time()
            res = func(*args, **kwargs)
            ed = time.time()
            if show_time:
                print("It took {} seconds to execute function {}".format(ed-st, func.__name__))
            return res
        return time_measurement
    return deco