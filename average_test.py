import numpy as np
import matplotlib.pyplot as plt
import unitTransfer
import dropExtremeAverage


theta = []
input = './test_data2/agile_log.txt'  # angle, control, velocity
standard = unitTransfer.MotorReunit(input)
smooth = dropExtremeAverage.DPAverage(15, 9, 2, 8, 5, 1)

while(True):
    current = standard.new()  # reunit it first
    if(current):
        # refresh the stream and store it
        i = smooth.new(current[0][0])
        theta.append(i)  # angle
    else:
        break

# save the filtered data
with open('./dst/filtered.txt', 'w') as dst:  # clear the file first
            dst.write("")

for i in range(len(theta)):
    with open('./dst/filtered.txt', 'a') as dst:
            dst.write(str(theta[i]))
            dst.write('\n')

input = np.loadtxt(input)

# plot them
time = np.linspace(0, len(theta), len(theta))

plt.plot(time, theta, color='green')  # filtered
plt.plot(time, input[:len(theta), 0], color='red', linestyle='--')  # unfiltered

plt.show()
