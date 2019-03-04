import numpy as np


class myKalman(object):
    """
    if zeroinit is not 0, the initial value of posteriori state. Or you have to
    initialize it by your value.\n
    Reference:  http://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf\n
    two parts:
        predict
          x_hat'(k) = A * x_hat(k-1) + B * u(k-1)
          P'(k) = A * P(k-1) * A.T + Q
        correct
          K(k) = P'(k) * H.T * (H * P'(k) * H.T +R).I
          x_hat(k) = x_hat'(k) + K(k) * (z(k) - H * x_hat'(k))
          P(k) = (I - K(k) * H) * P'(k)
    some var:\n
    (i for initial, u for update, t for have to be tried)\n
    `transition` (A) need initial
        based on state
    `controlTrans` (B) need initial
        transfer the form of control input
    `measureTrans` (H) need initial
        transfer the state to the form of measurement
    `prcssNsCov` (Q) need initial and the value have to be tried out\n
    `msrNsCov` (R) need initial and the value have to be tried out\n
    `control` (u) need update
    `measure` (z) need update
    `priEstErrCov` (P')
        priEstErrCov = transition  * postEECov_last  * transition.T + msrNsCov\n
    `postEstErrCov` (P)
        postEstErrCov = (I - gain * tansition) * priEstErrCov\n
        P = Mat[ei*ej]n, e = measure - statePost. the initial P is not important, can be a random, but DO NOT USE 0\n
    `msrResCov` (S)
        msrResCov = measureTrans * priEstErrCov * measureTrans.T + msrNsCov\n
    `gain` (K)
        gain = priEstErrCov * measureTrans.T * msrResCov
        when the noise of measurement is very small, K=measureTrans.T, when the noise of prediction is very small, K=0\n
    `statePre` (xk')
        statePre = transition * statePost_last + controlTrans * control_last\n
    `statePost` (xk)
        statePost = statePre + gain * (z(k) - measureTrans * statePre)
        is recommended to use a measurement to initialize
        statePost rely more on statePre when Q is small, vise versa
    """
    def __init__(self, stateDimen, measureDimen, controlDimen, zeroinit):
        self.stateDimen = stateDimen
        self.measureDimen = measureDimen
        self.controlDimen = controlDimen
        if zeroinit:
            self.statePost = np.zeros((self.stateDimen, 1))

    # set transition and measureTrans
    def setAH(self, transition, measureTrans):
        """if transition/measureTrans is 0, A/H will be identity matrix"""
        if not transition:
            self.transition = np.identity(self.stateDimen)
        else:
            self.transition = np.mat(transition)
        if not measureTrans:
            self.measureTrans = np.identity(self.stateDimen)
        else:
            self.measureTrans = np.mat(measureTrans)

    def setP(self, P):
        """
        initial the posteriori estimate error covariance
        it can be theoretically initialed with any value but 0
        """
        self.postEstErrCov = np.full((self.stateDimen, self.stateDimen), P)

    def setQ(self, Q):
        """
        set argument of process noise covariance\n
        Q should be a array of standard divisions of process noise of all dimension
        """
        Q = np.mat(Q)
        self.prcssNsCov = Q.reshape(-1, 1) * Q

    def setR(self, R):
        """
        set argument of measurement noise covariance\n
        R should be a array of standard divisions of measurement noise of all dimension
        """
        R = np.mat(R)
        self.msrNsCov = R.reshape(-1, 1) * R

    def setB(self, controlTrans):
        """
        set transition matrix of control input
        """
        self.controlTrans = np.mat(controlTrans)

    def predict(self, control):
        """
        x_hat'(k) = A * x_hat(k-1) + B * u(k-1)\n
        P'(k) = A * P(k-1) * A.T + Q
        """
        self.control = np.mat(control)
        self.statePre = self.transition * self.statePost + self.controlTrans * self.control
        self.priEstErrCov = self.transition * self.postEstErrCov * self.transition.T + self.prcssNsCov

    def correct(self, measure):
        """
        K(k) = P'(k) * H.T * S.I\n
        x_hat(k) = x_hat'(k) + K(k) * (z(k) - H * x_hat'(k))\n
        P(k) = (I - K(k) * H) * P'(k)
        """
        self.measure = np.mat(measure)
        self.msrResCov = self.measureTrans * self.priEstErrCov * \
            self.measureTrans.T + self.msrNsCov
        self.gain = self.priEstErrCov * self.measureTrans.T * self.msrResCov
        self.statePost = self.statePre + self.gain * \
            (self.measure - self.measureTrans * self.statePre)
        self.postEstErrCov = (np.identity(self.stateDimen) - self.gain * self.measureTrans) * self.priEstErrCov

    def new(self, measure, control):
        """
        initialing the state with measurement allows the signal converge more easily\n
        initialing the post estimate error covariance with an identity matrix
        because the initial value does not matter that much
        """
        self.predict(control)
        self.correct(measure)


if __name__ == '__main__':
    pass
