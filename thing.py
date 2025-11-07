import random
import math

def check(a, b, c, d):
    arr = str(a) + str(b) + str(c) + str(d)

    match arr:
        case "0000": return 0
        case "0001": return 1
        case "0010": return 0
        case "0011": return 1
        case "0100": return 0
        case "0101": return 1
        case "0110": return 0
        case "0111": return 1
        case "1000": return 1
        case "1001": return 1
        case "1010": return 1
        case "1011": return 1
        case "1100": return 0
        case "1101": return 0
        case "1110": return 0
        case "1111": return 1
    pass

neural = [] # weights for hidden nodes (5: bias + 4 inputs * 8) = 8 groups of 5
neural2 = [] # weights for public node (9: bias + 8 inputs from hidden) = 9

n_input = []
n_input2 = []
deltas = []
for i in range(8):
	neural.append([
		random.random()- 1, 
		random.random() * 2 - 1, 
		random.random() * 2 - 1, 
		random.random() * 2 - 1, 
		random.random() * 2 - 1])
for i in range(9):
	neural2.append(random.random() * 2 - 1)


def run_neural():
	a = 0 
	b = 1
	c = 1
	d = 0
	ret = 0
	error = 0
	for i in range(16):
		n_input = [a, b, c, d]
		n_input2 = []
		for i in range(8):
			val1 = neural[i][0]
			val2 = neural[i][1] * a
			val3 = neural[i][2] * b
			val4 = neural[i][3] * c
			val5 = neural[i][4] * d
			exp = val1 + val2 + val3 + val4 + val5
			if exp > 600: n_input2.append(1.0)
			elif exp < -600: n_input2.append(0.0)
			else: n_input2.append(1.0 / (1.0 + math.exp((-exp)) ))
		sigmoid = neural2[0]
		for i in range(len(n_input2)):
			sigmoid = sigmoid + n_input2[i] * neural2[i+1]
		ret = 1.0 / (1.0 + math.exp(-sigmoid))
		if (a + b + c + d < 5):
			#print("expected value for " + str(a) + " " + str(b) + " " + str(c) + " " + str(d) + ": " + str(ret))
			print("error is: " + str(check(a, b, c, d) - ret) )
		learn(n_input, n_input2, ret, sigmoid)

		d = d + 1
		if d > 1: c = c + 1; d = 0
		if c > 1: b = b + 1; c = 0
		if b > 1: a = a + 1; b = 0
	pass

def learn(n_input, n_input2, actual, sigmoid):
	global neural
	global neural2
	a = 0.1
	error = check(n_input[0], n_input[1], n_input[2], n_input[3]) - actual # error value
	d = actual * (1.0 - actual) * (error)
	if math.isnan(actual): print(str(neural))
	#wf = wi + 0.1 * err * g(x) * (1 - g(x)) * x
	#output nodes: 
	for i in range(len(neural2)):
		if i > 0: neural2[i] = neural2[i] + 0.1 * d * n_input2[i-1]
		else: neural2[i] = neural2[i] + 0.1 * d * 1
		pass
	#hidden nodes: err is output * weight: neural2[i] * neural[i][j]
	for i in range(len(neural)):
		for j in range(len(neural[0])):
			d2 = neural2[i+1] * d
			if j > 0:
				neural[i][j] = neural[i][j] + a * d2 * n_input[j-1]
				#neural[i][j] = neural[i][j] + a * neural[i][j] * error * neural2[i+1] * (1 - neural2[i+1]) * n_input[j-1]
			else: 
				neural[i][j] = neural[i][j] + a * d2 * 1
				#neural[i][j] = neural[i][j] + a * neural[i][j] * error * neural2[1+1] * (1 - neural2[i+1]) * 1
	return error

stop = 1
for i in range(1000):
	run_neural()