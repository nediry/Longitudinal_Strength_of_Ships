import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt

length = 72
breadth = length / 8
draft = breadth / 2.5
depth = 1.5 * draft
cb = .7
rho = 1.025
w = length * breadth * draft * cb * rho
offset = np.loadtxt('s60.txt', dtype=float) * breadth / 2

wl = np.array([0, .3, 1, 2, 3, 4, 5, 6]) * draft/4
pn0 = np.array([0, .5, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9.5, 10]) * length/10
pn = np.linspace(0, length, 101)

# offset genişletme 
offset2 = np.zeros((101, 8))
for i in range(8):
    f = interp1d(pn0, offset[:, i], kind='cubic')
    offset2[:, i] = f(pn)
alan = np.zeros((101, 8))
for i in range(101):
    alan[i, 1:] = 2 * cumtrapz(offset2[i], wl)

# atalet moment dağılımı
Iy = 3 * .189 * length / 100 # orta kesit atalet momenti
Ix = np.empty(101)
Ix[:5] = np.linspace(0, Iy/2, 5)
Ix[5:35] = np.linspace(Iy/2, Iy, 30)
Ix[35:76] = Iy
Ix[76:96] = np.linspace(Iy, Iy/4, 20)
Ix[96:] = np.linspace(Iy/4, 0, 5)

" denize indirme "

# çelik tekne ağırlığı dağılımı
N = length * breadth * depth
cs = (0.21 - 0.026*np.log10(N)) * (1 + 0.025 * (length / depth - 12))
G = cs * N * (1 + (2/3) * (cb - 0.70))
a = 0.680 * G / length
b = 1.185 * G / length
c = 0.580 * G / length
qx = np.zeros(101)
qx[:33] = np.linspace(a, b, 33)
qx[33:68] = b
qx[68:] = np.linspace(b, c, 33)

for i in range(35, 101):
    r =  pn[i] * np.tan(2.5 * np.pi / 180)
    batan = np.linspace(r, 0, i)
    y = np.zeros(i)
    for j in range(i):
        y[j] = np.interp(batan[j], wl, alan[j, :]) * rho
    mesafe = np.linspace(pn[:i], 0, len(pn[:i]))
    M1 = 0
    for j in range(i-1):
        M1 +=  pn[j] * y[j]
    M2 = 0
    for j in range(i-1):
        M2 +=  pn[j] * qx[j]
    cd = abs( (M1/np.trapz(qx[:i], pn[:i]))
       - (M2/np.trapz(qx[:i], pn[:i])) )
    if cd < 0.7:
        ax = np.zeros(101)
        ax[:i] = y
        break

# sephiye ve ağırlık dağılımının plot çıktısı
plt.figure(figsize = (10, 3))
plt.title("Denize İndirme")
plt.plot( pn, ax,  pn, qx)
plt.fill_between(pn, ax, color="g", alpha = .5, hatch="//", edgecolor="r")
plt.fill_between(pn, qx, color="r", alpha = .5, hatch="o", edgecolor="g")
plt.legend(["ax", "qx"])

px = ax - qx
Qx = np.array([0, *cumtrapz(px,  pn)])
Mx = np.array([0, *cumtrapz(Qx,  pn)])

# gerilme hesabı
ymax = .6 * depth
gerilme = -9.81 * Mx[1:-1] * ymax / (Ix[1:-1] * 1000)
gerilme = [0, *gerilme, 0]

# Baş papet geminin tam başında 0.posta old. kabul edildi
# Kesme kuvvetin max old. postada max kayma gerilmesi
# Bu postada eninde perde olduğunu kabul edildi
# Perde sanki dikörtgen kesitli gibi hesaplama yapıldı
n = -15 # baş papatten bir kaç potsa önce
A1 = depth * breadth
# Tarafsız eksenin 0.4xD old. kabul edildi
A2 = .4 * depth * breadth
S = (A1 - A2) * (.6 * depth) / 2
kayma = -9.81 * Qx[n] * S / (breadth * Ix[n] * 1000)
print('Denize indirme')
print('Kesme kuvvetin max old. postada max kayma gerilmesi')
print(round(kayma, 3))

# Gerilme dağılımının plot çıktısı
plt.figure(figsize = (10, 3))
plt.title("Denize İndrime")
plt.plot( pn[:-7], gerilme[:-7])
plt.fill_between(pn[:-7], gerilme[:-7], color="g", alpha=.4)
plt.legend(["gerilme"])


" dipten yaralanma durumu "

# toplam gemi ağırlık dağılımı
qx = np.zeros(101)
a = .68 * w / length
b = 1.187 * w / length
c = .58 * w / length
qx[:33] = np.linspace(a, b, 33)
qx[33:68] = b
qx[68:] = np.linspace(b, c, 33)

Ix = Ix - 0.15 * Ix

ax = alan[:, 7] * rho
ax[36:64] = np.zeros(28)

# sephiye ve ağırlık dağılımının plot çıktısı
plt.figure(figsize = (10, 3))
plt.title("Dipten Yaralanma")
plt.plot( pn, ax,  pn, qx)
plt.fill_between(pn, ax, color="g", alpha = .5, hatch="//", edgecolor="r")
plt.fill_between(pn, qx, color="r", alpha = .5, hatch="o", edgecolor="g")
plt.legend(["ax", "qx"])

px = ax - qx
Qx = np.zeros(101)
Qx[1:] = cumtrapz(px,  pn)

# lineer düzenleme 3%max(Q)
lineer = np.linspace(0, Qx[-1], 101)
Qx = Qx - lineer
Mx = np.zeros(101)
Mx[1:] = cumtrapz(Qx,  pn)

# lineer düzenleme 6%max(M)
lineer = np.linspace(0, Mx[-1], 101)
Mx = Mx - lineer

# gerilme hesabı
gerilme = 9.81 * Mx[1:-1] * ymax / (Ix[1:-1] * 1000)
gerilme = [0, *gerilme, 0]

# Kesme kuvvetin max old. postada max kayma gerilmesi
# Bu postada eninde perde olduğunu kabul edildi
# Perde sanki dikörtgen kesitli gibi hesaplama yapıldı
A1 = depth * breadth
# Tarafsız eksenin 0.4xD old. kabul edildi
A2 = .4 * depth * breadth
S = (A1 - A2) * (.6 * depth) / 2
kayma = -9.81 * Qx[n] * S / (breadth * Ix[n] * 1000)
print('\nDipten yaralanma')
print('Kesme kuvvetin max old. postada max kayma gerilmesi')
print(round(kayma, 3))

# gerilme dağılımının plot çıktısı
plt.figure(figsize = (10, 3))
plt.title("Dipten Yaralanma")
plt.plot(pn, gerilme)
plt.legend(["gerilme"])
plt.fill_between(pn, gerilme, color="g", alpha=.4)