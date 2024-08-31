



def to_sec(h, m, s):
    return h * 60 * 60 + m * 60 + s

def to_hms(sec):
    h = sec // 3600
    sec %= 3600
    m = sec // 60
    s = sec % 60
    return h, m, s

print(to_hms(to_sec(23, 25, 18)))




