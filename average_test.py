import numpy as np
import matplotlib.pyplot as plt
from unit_transfer import unitTransfer


theta = []

input = './test_data2/with_cmd.txt'  # angle, control, velocity
standard = unitTransfer.motorReunit(input)

# save the filtered data
with open('./dst/filtered.txt', 'w') as dst:
            dst.write("")

for i in range(len(theta)):
    with open('./dst/filtered.txt', 'a') as dst:
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
