#from PYchenka-Gynetik.pichenka-gynetik import *
import random
from numpy.random import uniform as u
import math
import numpy as np
import time

#log = [4, [7*6, 256], [256, 64], [64, 32], [32, 7]]          or
#log = [4, [256*256, 1024], [1024, 256], [256, 32], [32, 10]] or
#log = [3, [360, 999], [999, 222], [222, 123]]                or

def sig(x):
    list_sig = []
    for i in range(len(x)):
        list_sig.append(1/(1 + np.exp(-x[i])))
    return list_sig


def relu(x):
    list_relu = []
    for i in range(len(x)):
        list_relu.append(max(0, x[i]))
    return list_relu


# w = [ag1.w, ag2.w, ag3.w, ...]
# glasses = [int/float, int/float, int/float, ...]
def upgrade_w(w, glasses, log, mutashen_procent=3):
    ochky = []
    for i in range(len(glasses)):
        ochky.append(glasses[i].out)

    rand = []
    for i in range(len(ochky)):
        rand.append(ochky[i]/sum(ochky))

    ww = get_w(log)

    new_w = []

    for i in range(len(ww)):
        delta_w = r.choices(range(len(ww)), weights=rand, k=len(ww[0]))
        new_w.append([])
        for x in range(len(delta_w)):
            if r.random()*100 <= mutashen_procent:
                new_w[i].append(u(-1, 1))
            else:
                new_w[i].append(ww[i][delta_w[x]])

    ws = []
    for i in range(len(new_w)):
        ws.append(open_w(new_w[i], log))
    return ws


def zip_w(n_w, log):
    w = []
    for i in range(log[0]):
        for x in range(log[i+1][0]):
            for y in range(log[i+1][1]):
                w.append(n_w[i][x][y])
    return w


def open_w(n_w, log):
    w = []
    m = 0
    for i in range(log[0]):
        w.append([])
        for x in range(log[i+1][0]):
            w[i].append([])
            for y in range(log[i+1][1]):
                w[i][x].append(n_w[m+(x+1)*(y+1)-1])
        m += log[i+1][0]*log[i+1][1]
    return w


def get_w(list_w, log):
    ws = []
    for i in range(len(list_w)):
        ws.append(zip_w(list_w[i].w, log))
    return ws


def save(ws, log):
    file = open("agents.txt", "w")
    ws = get_w(ws, log)
    for i in range(len(ws)):
        for q in range(len(ws[i])):
            file.write(str(ws[i][q])+" ")
        file.write("\n")


def load(log, path="agents.txt"):
    file = open(path, "r").read().split("\n")
    ws = []
    for i in range(0, len(file)-1):
        ag = Agent()
        ag.w = open_w(list(map(float, file[i].split(" ")[0:-1])), log)
        ws.append(ag.w.copy())
    return ws


class Agent():
    def __init__(self):
        self.time = time.time()
        self.w = []

    def get_w(self):
        delta_w = []
        for i in range(len(self.w)):
            for j in range(len(self.w[i])):
                delta_w.append(w[i][j])
        return self.w

    def append(self, new_w):
        self.w.append(new_w)

    def input(self, x):
        self.x = x

    def update(self):
        self.s = np.dot(self.x, self.w[0])
        for i in range(len(self.w)-1):
            self.s = np.dot(self.s, self.w[i+1])
        

    def out(self):
        return relu(self.s)#/sig(self.s)


"""
ag = Agent()
ag.append(u(-1, 1, size=(32, 16)))
ag.append(u(-1, 1, size=(16, 8)))
ag.append(u(-1, 1, size=(8, 3)))
log = [3, [32, 16], [16, 8], [8, 3]]
save([ag], log)
ag.w = load(log)[0]
print("load!")
"""


if __name__ == "__main__":
    ag = Agent()
    ag.append(u(-1, 1, size=(10000, 1000)))
    ag.append(u(-1, 1, size=(1000, 256)))
    ag.append(u(-1, 1, size=(256, 128)))
    ag.append(u(-1, 1, size=(128, 64)))
    ag.append(u(-1, 1, size=(64, 32)))
    ag.append(u(-1, 1, size=(32, 16)))
    ag.append(u(-1, 1, size=(16, 8)))
    ag.append(u(-1, 1, size=(8, 3)))
    #log = [8, [10000, 1000], [1000, 256], [256, 128], [128, 64], [64, 32], [32, 16], [16, 8], [8, 3]]
    while True:
        start_time = time.time()
        for i in range(1000):
            ag.input(u(0, 1, size=(10000)))
            ag.update()
        print("total time:", time.time()-start_time, "|", "update time:", round((time.time()-start_time)/1000, 4))