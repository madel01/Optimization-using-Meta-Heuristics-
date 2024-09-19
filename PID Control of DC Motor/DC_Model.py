# -*- coding: utf-8 -*-
"""
Created on Fri May  3 16:28:25 2024

@author: mohamed
"""
import numpy as np

class DCmotorPID:
    
    def __init__(self, la, ra, jeq, beq, k1, k2, k3):
        
        self.La = la
        self.Ra = ra
        self.Jeq = jeq
        self.Beq = beq
        self.K1 = k1
        self.K2 = k2
        self.K3 = k3
    
    
    def __str__(self):
        
        return f"""Motor specifications:\n\narmature inductance = {self.La}\narmature resistance = {self.Ra}\nviscosity = {self.Jeq}\nmomentum = {self.Beq}\nConstant1 = {self.K1}\nConstant2 = {self.K2}\nConstant3 = {self.K3}\n"""
        
    
    def evaluate(self,inputs=[]):
        
        a1 = self.La*self.Jeq
        a2 = (self.La*self.Beq) + (self.Ra*self.Jeq)
        a3 = (self.Ra*self.Beq) + (self.K3*self.K2)
        a4 = (self.K1*self.K2)
        
        factors = [0]*5
        factors[0] = a1
        factors[1] = a2
        factors[2] = a3 + a4*inputs[1] 
        factors[3] = a4*inputs[0]
        factors[4] = a4*inputs[2]
        
        ##### finding the roots

        roots = np.roots(factors)
        max_root = max(roots.real)
        fobj = np.exp(max_root)
        return fobj

        


    
 


