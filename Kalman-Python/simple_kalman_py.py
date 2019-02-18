import numpy as np
import matplotlib.pyplot as plt


P = 6e-2
Qa = 4.5e-1
Ra = 1e-5
Qv = 5e-7
Rv = 5e-1


class myKalman(object):
    # Reference:    http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf
    # two parts:
    #   predict
    #       x_hat'(k) = A * x_hat(k-1) + B * u(k-1)
    #       P'(k) = A * P(k-1) * A.T + Q
    #   correct
    #       K(k) = P'(k) * H.T * (H * P'(k) * H.T +R).I
    #       x_hat(k) = x_hat'(k) + K(k) * (z(k) - H * x_hat'(k))
    #       P(k) = (I - K(k) * H) * P'(k)
    # some var:
    # (i for initial, u for update, t for have to be tried)
    # <i>   transition      (A)based on state
    # <i>   controlTrans    (B)transfer the form of control input
    # <u>   control         (u)
    # <u>   measure         (z)
    # <i>   measureTrans    (H)transfer the state to the form of measurement
    # <it>  prcssNsCov      (Q)
    # <it>  msrNsCov        (R)
    #       priEstErrCov    (P')priEstErrCov = transition  * postEECov_last  * transition.T + msrNsCov
    #       postEstErrCov   (P)the initial P is not important, can be a random, but DO NOT USE 0
    #                           P = Mat[ei*ej]n, e = measure - statePost
    #                           postEstErrCov = (I - gain * tansition) * priEstErrCov
    #       msrResCov       (S)msrResCov = measureTrans * priEstErrCov * measureTrans.T + msrNsCov
    #       gain            (K)gain = priEstErrCov * measureTrans.T * msrResCov
    #                           when the noise of measurement is very small, K=measureTrans.T, when the noise of prediction is very small, K=0
    #       statePre        (xk')statePre = transition * statePost_last + controlTrans * control_last
    #       statePost       (xk)is recommended to use a measurement to initialize
    #                           statePost = statePre + gain * (z(k) - measureTrans * statePre)
    #                           statePost rely more on statePre when Q is small, vise versa
    # controlDimen may not used, in this case input 0
    def __init__(self, stateDimen, measureDimen, controlDimen):
        self.stateDimen = stateDimen
        self.measureDimen = measureDimen
        if controlDimen:
            self.controlSwitch = True
            self.controlDimen = controlDimen
        else:
            self.controlSwitch = False

    # set transition and measureTrans
    def setAH(self, transition, measureTrans):  # if input 0, A/H will be identity
        if not transition:
            self.transition = np.identity(self.stateDimen)
        else:
            self.transition = np.mat(transition)
        if not measureTrans:
            self.measureTrans = np.identity(self.stateDimen)
        else:
            self.measureTrans = np.mat(measureTrans)

    def setQ(self, Q):
        Q = np.mat(Q)
        self.prcssNsCov = Q.reshape(-1, 1) * Q

    def setR(self, R):
        R = np.mat(R)
        self.msrNsCov = R.reshape(-1, 1) * R

    def setB(self, controlTrans):
        if self.controlSwitch:
            self.controlTrans = np.mat(controlTrans)

    def predict(self, control):
        #   predict
        #       x_hat'(k) = A * x_hat(k-1) + B * u(k-1)
        #       P'(k) = A * P(k-1) * A.T + Q
        self.control = np.mat(control)
        self.statePre = self.transition * self.statePost + self.controlTrans * self.control
        self.priEstErrCov = self.transition * \
            self.postEstErrCov * self.transition.T + self.prcssNsCov

    def correct(self, measure):
        #   correct
        #       msrResCov   (S)msrResCov = measureTrans * priEstErrCov * measureTrans.T + msrNsCov
        #       gain        (K)gain = priEstErrCov * measureTrans.T * msrResCov
        #       K(k) = P'(k) * H.T * S.I
        #       x_hat(k) = x_hat'(k) + K(k) * (z(k) - H * x_hat'(k))
        #       P(k) = (I - K(k) * H) * P'(k)
        self.measure = np.mat(measure)
        self.msrResCov = self.measureTrans * self.priEstErrCov * \
            self.measureTrans.T + self.msrNsCov
        self.gain = self.priEstErrCov * self.measureTrans.T * self.msrResCov
        self.statePost = self.statePre + self.gain * \
            (self.measure - self.measureTrans * self.statePre)
        self.postEstErrCov = (np.identity(
            self.stateDimen) - self.gain * self.measureTrans) * self.priEstErrCov

    def new(self, measure, control):  # control may not needed
        # initialing the state with measurement allows the signal converge more easily
        # initialing the post estimate error covariance with an identity matrix because the initial value does not matter that much
        self.predict(control)
        self.correct(measure)


if __name__ == '__main__':

    # ################initial the filter################
    # [[angle], [velocity]]
    motor = myKalman(2, 2, 1)

    #  ################set period = 4ms################
    t = 0.04

    # ################set transition matrix and transform matrix################
    # the transition matrix and transform matrix for control vector is simply based on Newton's law
    motor.setAH([[1, t], [0, 1]], 0)
    motor.setB([[t * t / 2], [t]])

    # ################set Q, R, P################
    motor.setQ([Qa, Qv])
    motor.setR([Ra, Rv])
    motor.postEstErrCov = np.full((motor.stateDimen, motor.stateDimen), P)

    velocity = 0
    velocity_last = 0

    x = []
    y = []
    # adjust Q, R with the test input
    input = np.loadtxt('./test_data2/without_cmd1.txt')  # angle, control, velocity

    # give a initial state (be careful the unit)
    motor.statePost = np.mat([[input[0, 0]], [input[0, 2]*2*np.pi]])

    for column in input:
        angleIn = column[0]
        controlIn = column[1]
        velocityIn = column[2]

        # then transfer the units
        angle = angleIn
        control = (controlIn * 2 * np.pi - velocity)  # indicate acceleration by increacesment
        velocity = velocityIn * 2 * np.pi

        motor.new(([angle], [velocity]), control)

        x.append(motor.statePost[0, 0])
        y.append(motor.statePost[1, 0])
        # with open('./dst/1.txt', 'a') as dst:
        #     dst.write(str(motor.statePost[0, 0]))
        #     dst.write('\n')

    # 然后plot之类的
    # 定义 x 变量的范围, 数量
    time = np.linspace(0, input.shape[0], input.shape[0])

    plt.subplot(211)
    plt.plot(time, x, color='green')
    plt.plot(time, input[:, 0], color='red', linestyle='--')

    plt.subplot(212)
    plt.plot(time, y, color='green')
    plt.plot(time, (input[:, 2]*2*np.pi/60), color='red', linestyle='--')

    plt.show()

    i = 1