import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from dmp_functions import *
## parameters
dt = 0.01
alpha_x = 1
alpha_y = 8
beta_y = 10
m = 8
h = 1
## get y
y_input = pd.read_csv("./slice-y-sample.csv", usecols = [6])
y_input = np.array(y_input)

y = []
for y_value in y_input:
	y.append(y_value[0])
time_step = len(y)
## get dy
dy_input = pd.read_csv("./slice-dy-sample.csv", usecols = [6])
dy_input = np.array(dy_input)
dy = []
for dy_value in dy_input:
	dy.append(dy_value[0])
## get ddy
ddy = [0]
for i in range(1, time_step):
	ddy.append((dy[i]-dy[i-1])/dt)
## get x
x = canonical_func(alpha_x, dt, time_step)
# plt.plot(range(1, len(y) + 1), x)
# plt.show()
## get s
s = []
for n in range(time_step):
	s.append(x[n]*(y[len(y) - 1] - y[0]))
s = np.array(s)
s = s.reshape(time_step, 1)
s_trans = s.transpose()

## get phi
phi = []
for i in range(m):
	phi_i = np.zeros((time_step, time_step))
	for j in range(time_step):
		phi_i[j][j] = basis_func((i + 1)/m, h, x[j])
	phi.append(phi_i)
## get fd
fd = np.zeros((time_step, 1))
for i in range(time_step):
	fd[i][0] = ddy[i] - alpha_y * (beta_y * (y[len(y) - 1] - y[0]) - dy[i])
## get weight
weight = []
for i in range(m):
	weight_i = (np.matmul(np.matmul(s_trans, phi[i]), fd))/(np.matmul(np.matmul(s_trans, phi[i]), s))
	weight.append(weight_i[0][0])
print(weight)
## get estimated f list
f_list = []
for i in range(time_step):
	upper = 0
	lower = 0
	for j in range(m):
		upper = upper + basis_func((j + 1)/m, h, x[i]) * weight[j]
		lower = lower + basis_func((j + 1)/m, h, x[i])
	f_i = (upper/lower) * x[i] * (y[len(y) - 1] - y[0])
	f_list.append(f_i)
## get results
ddy_list = [ddy[0]]
dy_list = [dy[0]]
y_list = [y[0]]
for i in range(time_step - 1):
	ddy_next = alpha_y * (beta_y * (y[len(y) - 1] - y_list[i]) - dy_list[i]) + f_list[i + 1]
	dy_next = dy_list[i] + ddy_list[i] * dt
	y_next = y_list[i] + dy_list[i] * dt
	ddy_list.append(ddy_next)
	dy_list.append(dy_next)
	y_list.append(y_next)

plt.plot(range(1, time_step + 1), y, label = "Raw")
plt.plot(range(1, time_step + 1), y_list, label = "DMP", linestyle = "--")
plt.xlabel("Time (ms)")
plt.ylabel("Angular Position (rad)")
plt.legend(loc = "upper right")
plt.show()


	
		













