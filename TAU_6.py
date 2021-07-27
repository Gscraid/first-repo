import numpy as np
import matplotlib.pyplot as plt
import math

def Filtr(s, sign):
    buf_x = np.zeros(3)
    buf_y = np.zeros(2)
    sign_new = np.zeros(len(sign))
    for n in range(len(sign)):
        buf_x[2] = buf_x[1]
        buf_x[1] = buf_x[0]
        buf_x[0] = sign[n]
        sign_new[n] = s[0] * buf_x[0] + s[1] * buf_x[1] + s[2] * buf_x[2] - s[4] * buf_y[0] - s[5] * buf_y[1]
        buf_y[1] = buf_y[0]
        buf_y[0] = sign_new[n]
    return sign_new


def noise(length):
    sign = np.zeros(length)
    t = np.arange(length)
    for k in range(length // 2):
        if k == 0:
            sign = 0.5
        else:
            sign = sign + np.cos(2 * np.pi * t * k / length)
    return sign

def DFT(sign):
    characteristics = {'imag': np.zeros([len(sign)//2]),
                       'real': np.zeros([len(sign)//2]),
                       'amp': np.zeros([len(sign)//2]),
                       'phase': np.zeros([len(sign)//2]),
                       'log_amp': np.zeros([len(sign)//2])}
    for k in range(len(sign) // 2):
        S = 0
        for n in range(len(sign)):
            S = S + sign[n] * math.e ** complex(0, -2 * math.pi * n * k / len(sign))
        characteristics['imag'][k] = S.imag
        characteristics['real'][k] = S.real
        characteristics['amp'][k] = 2 * abs(S) / len(sign)
        characteristics['log_amp'][k] = 20 * math.log10(2 * abs(S) / len(sign))
        if characteristics['amp'][k] < 0.01 and Zer:
            characteristics['phase'][k] = 0
        else:
            characteristics['phase'][k] = math.atan2(S.imag, S.real)
    return characteristics

N = 2000
x = noise(N)
Zer = False

s = [[0.54, 0.334, 0.54, 1, 0.334, 0.081]]

for i in range(len(s)):
     x = Filtr(s[i], x)
X = DFT(x)

fig = plt.figure(figsize=(10, 5), dpi=100)

plt.subplot(2, 2, 1) 
plt.title('АЧХ') 
plt.plot(X['amp'],'g-') 
plt.grid(True)  
ax = plt.gca()  

plt.subplot(2, 2, 2) 
plt.title('ЛАХ') 
plt.plot(X['log_amp'],'g-')  
plt.grid(True)
plt.xscale('log') 
ax = plt.gca()  

plt.subplot(2, 2,(3,4))
plt.title('ФЧХ') 
plt.plot(X['phase'],'g-')  
plt.grid(True)  
ax = plt.gca()  


plt.tight_layout()  
plt.show()   