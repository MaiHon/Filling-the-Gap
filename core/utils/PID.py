from math import exp

class PIDController():
    def __init__(self, cfg):
        self.I_k = 0.0
        self.W_k = 1.0
        self.e_k = 0.0
        
        self.Kp = cfg.Kp
        self.Ki = cfg.Ki
        self.Kd = cfg.Kd
        
    
    def _calc_P(self, Kp, err, scale=1):
        return Kp/(1.0 + float(scale)*exp(err))
    
    
    def pid(self, C, kld):
        error_k = C - kld
        
        P = self._calc_P(self.Kp, error_k, scale=1) + 1
        I = self.I_k + self.Ki * error_k
        
        if self.W_k < 1:
            I = self.I_k
        
        W = P + I
        self.W_k = W
        self.I_k = I
        
        if W < 1:
            W =1
            
        return W, error_k