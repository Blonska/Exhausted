import numpy as np


class Transformation:
    n=2
    tay=2


    def _koef(self,x1_ash_conc,x2_nitr_conc):
        k1L = 0.002 + 0.009 * x1_ash_conc + 0.003 * x2_nitr_conc
        k2L = lambda : 0.00114 - 0.00028 * x2_nitr_conc if x2_nitr_conc > 0.5 else 0.002 * x2_nitr_conc
        k3L = lambda : 0.04 - 0.003 * x1_ash_conc if x1_ash_conc > 0.5 else 0.005 * x1_ash_conc
        k4L = lambda: 0.01 if x2_nitr_conc > 2 else 0.005 * x2_nitr_conc
        k5 = lambda: 0 if x2_nitr_conc <= 0.5 else(0.00462 * x2_nitr_conc - 0.00231 if 0.5 < x2_nitr_conc <= 2  else 0.007)
        k5S = lambda: 0.2 if x1_ash_conc <= 5 else k5
        k1S = 1.6 * k1L
        k3S = 1.35 * k3L()
        k4S = k4L()
        k6 = 0.00006
        return k1L,k2L(),k3L(),k4L(),k5S(),k1S,k3S,k4S,k6
    
    def _M_miner(self,Mov,x1_star_nitr,x2_nitr):
         ML = lambda: 0.1 * Mov if x2_nitr - 1.16 * x1_star_nitr <= 0.44 else(0.5 * Mov if 0.44 < x2_nitr-1.16*x1_star_nitr <= 1.5  else Mov)
         return ML()

    def _L_above(self,L0,x1_ash_conc,x2_nitr_conc):
        L = [0 for i in range(self.n+1)]
        for i in range(0,self.n):
            L[i+1] = (L0 * self.tay + L[i]) / (1 + self.tay * (self._koef(x1_ash_conc,x2_nitr_conc)[0] + self._koef(x1_ash_conc,x2_nitr_conc)[2]))
        return L[i+1]

    def _NL_above(self,NL0,x1_ash_conc,x2_nitr_conc,Mov,x1_star_nitr,x2_nitr):
        NL = [0 for i in range(self.n+1)]
        for i in range(0,self.n):
            NL[i+1] = (NL0 * self.tay + NL[i]) / (1 + self.tay * (self._koef(x1_ash_conc,x2_nitr_conc)[0] * self._M_miner(Mov,x1_star_nitr,x2_nitr) + self._koef(x1_ash_conc,x2_nitr_conc)[2]))
        return NL[i+1]

    def _Lu_below(self,Lu0,x1_ash_conc,x2_nitr_conc):
        Lu = [0 for i in range(self.n+1)]
        for i in range(0,self.n): 
            Lu[i+1] = (Lu0 * self.tay + Lu[i]) / (1 + self.tay * (self._koef(x1_ash_conc,x2_nitr_conc)[5] + self._koef(x1_ash_conc,x2_nitr_conc)[6]))
        return Lu[i+1]

    def _NLu_below(self,NLu0,x1_ash_conc,x2_nitr_conc,Mov,x1_star_nitr,x2_nitr):
        NLu = [0 for i in range(self.n+1)]
        for i in range(0,self.n):
            NLu[i+1] = (NLu0 * self.tay + NLu[i]) / (1 + self.tay * (self._koef(x1_ash_conc,x2_nitr_conc)[5] * self._M_miner(Mov,x1_star_nitr,x2_nitr) + self._koef(x1_ash_conc,x2_nitr_conc)[6]))
        return NLu[i+1]

    def _F_above(self,x1_ash_conc,x2_nitr_conc,L0):
        F = [0 for i in range(self.n+1)]
        for i in range(0,self.n):
            F[i+1] = (F[i] + self._koef(x1_ash_conc,x2_nitr_conc)[2] * self._L_above(L0,x1_ash_conc,x2_nitr_conc) * self.tay) / (1 + self.tay * (self._koef(x1_ash_conc,x2_nitr_conc)[1] + self._koef(x1_ash_conc,x2_nitr_conc)[3] + self._koef(x1_ash_conc,x2_nitr_conc)[4]))
        return F[i+1]

    def _NF_above(self,x1_ash_conc,x2_nitr_conc,NL0,Mov,x1_star_nitr,x2_nitr):
        NF = [0 for i in range(self.n+1)]
        for i in range(0,self.n):
            NF[i+1] = (NF[i] + self._koef(x1_ash_conc,x2_nitr_conc)[2] * self._NL_above(NL0,x1_ash_conc,x2_nitr_conc,Mov,x1_star_nitr,x2_nitr) * self.tay) / (1 + self.tay * (self._koef(x1_ash_conc,x2_nitr_conc)[1] * self._M_miner(Mov,x1_star_nitr,x2_nitr) + self._koef(x1_ash_conc,x2_nitr_conc)[3] + self._koef(x1_ash_conc,x2_nitr_conc)[4]))
        return NF[i+1]

    def _Fu_below(self,Hm,x1_ash_conc,x2_nitr_conc,Lu0,k2S):
        k2S = self._koef(x1_ash_conc,x2_nitr_conc)[1] * (1.22 + 0.488 * Hm)
        Fu = [0 for i in range(self.n+1)]
        for i in range(0,self.n):
            Fu[i+1] = (Fu[i] + self._koef(x1_ash_conc,x2_nitr_conc)[6] * self._Lu_below(Lu0,x1_ash_conc,x2_nitr_conc) * self.tay) / (1 + self.tay * (k2S + self._koef(x1_ash_conc,x2_nitr_conc)[7] + self._koef(x1_ash_conc,x2_nitr_conc)[4]))
        return Fu[i+1]

    def _NFu_below(self,NLu,k2S,MFu,Hm,Mov,x1_star_nitr,x2_nitr,NLu0,x1_ash_conc,x2_nitr_conc):
        k2S = self._koef(x1_ash_conc,x2_nitr_conc)[1] * (1.22 + 0.488 * Hm)
        NFu = [0 for i in range(self.n + 1)]
        for i in range(0,self.n):
            NFu[i+1] = (NFu[i] + self._koef(x1_ash_conc,x2_nitr_conc)[6] * self._NLu_below(NLu0,x1_ash_conc,x2_nitr_conc,Mov,x1_star_nitr,x2_nitr) * self.tay) / (1 + self.tay * (k2S * self._M_miner(Mov,x1_star_nitr,x2_nitr) + self._koef(x1_ash_conc,x2_nitr_conc)[7] + self._koef(x1_ash_conc,x2_nitr_conc)[4]))
        return NFu[i+1]

object=Transformation()
print(object._NFu_below(1,1,1,1,1,1,1,1,1,1))



