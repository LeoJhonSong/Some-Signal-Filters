from Simple_Kalman_Filterimport simple_kalman as kal
import numpy as np
import matplotlib.pyplot as plt
from unit_transfer import unitTransfer


# initial the filter
motor = kal.myKalman(2, 2, 1, 1)

# set period = 4ms
t = 0.04

# set transition matrix and transform matrix
# the transition matrix and transform matrix for control vector is simply based on Newton's law
motor.setAH([[1, t], [0, 1]], 0)
motor.setB([[t * t / 2], [t]])

# set P, Q, R
P = 5e-2
Qa = 7e-1
Ra = 5e-6
Qv = 1e-5
Rv = 1e-2

motor.setP(P)
motor.setQ([Qa, Qv])
motor.setR([Ra, Rv])

velocity = 0
velocity_last = 0

# give a initial state (be careful the unit)
# motor.statePost = np.mat([[input[0, 0]], [input[0, 2]*2*np.pi]])  # better initial value

# use these two array to store the filtered data
theta = []
speed = []

window = []

input = './test_data2/with_cmd.txt'  # angle, control, velocity

standard = unitTransfer.motorReunit(input)

while (True):
    # transfer the units
    current = standard.new()
    # filter the data
    if(current):
        motor.new(current[0], current[1])
    else:
        break
    # store the filtered data
    theta.append(motor.statePost[0, 0])  # angle
    o = len(theta)
    speed.append(motor.statePost[1, 0])  # velocity

# save the filtered data
for i in range(len(theta)):
    with open('./dst/filtered.txt', 'w') as dst:
            dst.write(str(theta[i]))
            dst.write('\n')

input = np.loadtxt(input)

# plot them
time = np.linspace(0, len(theta), len(theta))

plt.subplot(211)
plt.plot(time, theta, color='green')  # filtered
plt.plot(time, input[:len(theta), 0], color='red', linestyle='--')  # unfiltered

plt.subplot(212)
plt.plot(time, speed, color='green')  # filtered
plt.plot(time, (input[:len(theta), 2]*2*np.pi/60), color='red', linestyle='--')  # unfiltered

plt.show()
