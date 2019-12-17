import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

from hubbard_fn import *
from cmf import *
from myfci import *
from pyscf_helper import *
import pyscf
ttt = time.time()
np.set_printoptions(suppress=True, precision=4, linewidth=1500)
pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues


for ri in range(0,26):
    ###     PYSCF INPUT
    r0 = 0.95 + 0.05 * ri
    molecule = '''
    N   {1}  {1}   {1} 
    H   {0}   {0}   0
    H   0   {0}   {0}
    H   {0}   0   {0}
    '''.format(r0,r0/2)
    charge = 0
    spin  = 0
    basis_set = 'sto-3g'

    ###     TPSCI BASIS INPUT
    orb_basis = 'ibmo'
    cas = True
    cas_nstart = 1
    cas_nstop =  8
    loc_start = 1
    loc_stop = 8
    cas_nel = 8

    ###     TPSCI CLUSTER INPUT
    blocks = [[0,1,2,3],[4,5,6,7]]
    init_fspace = ((2, 2), (2, 2))
    blocks = [[0,1],[2,3],[4,5],[6]]
    init_fspace = ((1, 1), (1, 1),(1, 1),(1,1))
    block_nel = [[1, 1], [1, 1],[1, 1],[1,1]]

    nelec = tuple([sum(x) for x in zip(*init_fspace)])
    if cas == True:
        assert(cas_nel == sum(nelec))
        nelec = cas_nel


    # Integrals from pyscf
    #Integrals from pyscf
    pmol = PyscfHelper()
    pmol.init(molecule,charge,spin,basis_set,orb_basis,
                    cas_nstart=cas_nstart,cas_nstop=cas_nstop,cas_nel=cas_nel,cas=True,
                    loc_nstart=loc_start,loc_nstop = loc_stop)

    h = pmol.h
    g = pmol.g
    ecore = pmol.ecore

    do_fci = 1
    do_hci = 1
    do_tci = 1

    if do_fci:
        efci, fci_dim = run_fci_pyscf(h,g,nelec,ecore=ecore)
    if do_hci:
        ehci, hci_dim = run_hci_pyscf(h,g,nelec,ecore=ecore,select_cutoff=1e-3,ci_cutoff=1e-3)

    #cluster using hcore
    idx = e1_order(h,cut_off = 3e-1)
    h,g = reorder_integrals(idx,h,g)
    if do_tci:
        Ecmf = run_cmf(h,g,blocks,block_nel,ecore=ecore)
        Edps = run_cmf(h,g,blocks,block_nel,ecore=ecore,miter=0)
        Escf = pmol.mf.e_tot
        print(" CMF:      %12.8f Dim:%6d" % (Ecmf, 1))
        print(" DPS:      %12.8f Dim:%6d" % (Edps, 1))
        print(" SCF:      %12.8f Dim:%6d" % (Escf, 1))

    if do_fci:
        print(" FCI:        %12.8f Dim:%6d" % (efci, fci_dim))
        print("%6.3f %16.8f   %16.8f  %16.8f  %16.8f"%(r0,Escf,Edps,Ecmf,efci))
    if do_hci:
        print(" HCI:        %12.8f Dim:%6d" % (ehci, hci_dim))


