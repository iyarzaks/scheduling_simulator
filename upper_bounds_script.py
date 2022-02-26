import math
import matplotlib.pyplot as plt
import numpy as np


def f_1(M):
    return (M - 2) // 3 + 1


def f_2(M):
    return math.ceil(2 * (M - 2) / 3) + 1

x = []
y = []
y_2 = []
for i in range(2, 200):
    # print(i, f_1(i), f_2(i), 1 + i / f_1(i), 1 + 2 * i / f_2(i))
    x.append(i)
    y.append(max(1 + i / f_1(i), 1 + 2 * i / f_2(i)))
    y_2.append(4 + 4 / i)
plt.plot(x, y, label="Hybrid-WSVF competitive ratio")
plt.plot(x, y_2, label="4+4/M")
plt.legend(loc='best')
plt.xlabel("Number of machines")
plt.ylabel("Competitive ratio")
plt.show()

