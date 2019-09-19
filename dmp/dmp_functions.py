import math
def canonical_func(alpha_x, dt, time_steps):
	x = [1]
	for i in range(1, time_steps):
		x.append(0)
		x[i] = (1 - dt)*x[i - 1]
	return x

def basis_func(center, variance, x_value):
	return math.exp(-variance * (x_value - center)**2)


