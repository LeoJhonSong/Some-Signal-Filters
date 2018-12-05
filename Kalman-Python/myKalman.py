import numpy as np


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
    # <i>   transition  (A)based on state
    # <i>   controlTrans(B)transfer the form of control input
    # <u>   control     (u)
    # <u>   measure     (z)
    # <i>   measureTrans(H)trnsfer the state to the form of measurement
    # <it>  prcssNsCov  (Q)
    # <it>  msrNsCov    (R)
    #       priEECov    (P')priEECov = transition  * postEECov_last  * transition.T + msrNsCov
    #       postEECov   (P)the initial P is not important, can be a random, but DO NOT USE 0
    #                      P = Mat[ei*ej]n, e = measure - statePost
    #                      postEECov = (I - gain * tansition) * priEECov
    #       msrResCov   (S)msrResCov = measureTrans * priEECov * measureTrans.T + msrNsCov
    #       gain        (K)gain = priEECov * measureTrans.T * msrResCov
    #                      when the noise of measurement is very small, K=measureTrans.T, when the noise of prediction is very small, K=0
    #       statePre    (xk')statePre = transition * statePost_last + controlTrans * control_last
    #       statePost   (xk)is recommended to use a measurement to initialize
    #                       statePost = statePre + gain * (z(k) - measureTrans * statePre)
    #                       statePost rely more on statePre when Q is small, vise versa
    def __init__(self, stateDimen, measureDimen, controlDimen):  # controlDimen may not used, in this case input 0
        self.start = False
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
        self.prcssNsCov = np.mat(Q)

    def setP(self, R):
        self.msrNsCov = np.mat(R)

    def setB(self, controlTrans):
        if self.controlSwitch:
            self.controlTrans = np.mat(controlTrans)

    def predict(self, control):
        #   predict
        #       x_hat'(k) = A * x_hat(k-1) + B * u(k-1)
        #       P'(k) = A * P(k-1) * A.T + Q
        self.statePre = self.transition * self.statePost + self.controlTrans * self.control
        self.control = np.mat(control)
        self.priEECov = self.transition * self.postEECov * self.transition.T + self.prcssNsCov

    def correct(self, measure):
        #   correct
        #       K(k) = P'(k) * H.T * (H * P'(k) * H.T +R).I
        #       x_hat(k) = x_hat'(k) + K(k) * (z(k) - H * x_hat'(k))
        #       P(k) = (I - K(k) * H) * P'(k)
        #       msrResCov   (S)msrResCov = measureTrans * priEECov * measureTrans.T + msrNsCov
        #       gain        (K)gain = priEECov * measureTrans.T * msrResCov
        self.measure = np.mat(measure)
        self.msrResCov = self.measureTrans * self.priEECov * self.measureTrans.T + self.msrNsCov
        self.gain = self.priEECov * self.measureTrans.T * self.msrResCov
        self.statePost = self.statePre + self.gain * (self.measure - self.measureT)

    def new(self, measure, control):  # control may not needed
        if not self.start:
            self.statePost = measure
        self.predict(control)
        self.correct(measure)


t = 1  # period
motor = myKalman(2, 2, 1)  # [[angle], [velocity]]
motor.setAH([[1, 0], [0, 1]], 0)
motor.setB([[t * t / 2], [t]])
motor.setP(1e-5)
motor.setQ(1e-5)
velocity = 0
while True:
    velocity_last = velocity
    control = (velocity - velocity_last) / t
    motor.new(_, control)
    # 然后plot之类的
