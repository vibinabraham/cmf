import numpy as np
import scipy 
import itertools as it


class Cluster(object):

    def __init__(self):

        self.orb_list = 0

        self.oei  = 0
        self.tei  = 0
        self.n_orb = 0
        self.p_evec = 0
        self.p_eval = 0
        self.n_range = 0 
        self.n_a = 0
        self.n_b = 0
        self.S2 = 0
        self.Ham = 0


        self.tdm = {}
        self.h_eff = 0

        
    def init(self,bl,n_elec, oei, tei, Sz=None, S2 = None):

        self.orb_list = bl
        print(n_elec)
        self.n_a  = n_elec[0]
        self.n_b  = n_elec[1]
        self.oei  = oei
        self.tei  = tei
        self.n_orb = oei.shape[0]


    def tucker_vecs(self,p,e):
        #self.Ham = H
        self.p_evec = p 
        self.p_eval = e


    def store_tdm(self,key,tdm):
        self.tdm[key] = tdm

    def update_t(self,heff):
        self.h_eff += heff


class TuckerBlock(object):

   def __init__(self, veclist):
      self.veclist = veclist
      self.vector = {}
      self.start = 0
      self.stop = 0
      self.shape = 0
      self.core = 0

   def readvectors(self,vec,i):
      self.vector[i] =  vec

   def vec_startstop(self,start,stop,shape):
      self.start =  start
      self.stop  =  stop
      self.shape = shape 

   def makecore(self,core):
      self.core = core
