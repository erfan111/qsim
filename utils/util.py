

def clamp(val, lo, hi):
    if val < lo:
        return lo
    else:
        if val > hi:
            return hi
    return val

def u_saturation_sub(a, b):
    if a > b:
        return a - b
    return 0