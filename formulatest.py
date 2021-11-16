import math

"""
checking formula for calculating nb of times window of size width fits into signal of size length with offset offset
"""

def formula(length, width, offset):
    return math.floor((length-width)/offset) + 1

def brute_force(length, width, offset):
    x = 0
    n = 0
    while x <= length - width:
        x += offset
        n += 1
    return n

for length in range (1, 100):
    # print(f'processing length {length}')

    for width in range (1, length):
        # print(f'\tprocessing width {width}')

        for offset in range (1, length):

            true = brute_force(length, width, offset)
            test = formula(length, width, offset)

            if true != test:
                print(length, width, offset, 'mismatch')
