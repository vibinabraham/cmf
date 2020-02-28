import numpy as np
from math import factorial
import copy as cp
from scipy import sparse
import pyscf
from Cluster import *

def get_cluster_eri(bl,h,g):
# {{{
    size_bl = len(bl)
    ha = np.zeros((size_bl,size_bl))
    ga = np.zeros((size_bl,size_bl,size_bl,size_bl))

    #AAAA
    for i,a in enumerate(bl):
        for j,b in enumerate(bl):
            ha[i,j] = h[a,b]
            for k,c in enumerate(bl):
                for l,d in enumerate(bl):
                    ga[i,j,k,l] = g[a,b,c,d]

    return ha,ga
# }}}

def get_block_eri_2(block,Cluster,tei,a,b,c,d):
# {{{
    """
    Gives the two electron integral living in respective blocks <AB|CD>
    """
    g_bl = np.zeros((Cluster[a].n_orb, Cluster[b].n_orb, Cluster[c].n_orb, Cluster[d].n_orb))

    #print("       a ")
    #print(Cluster[a].so_orb_list)
    #print("       b ")
    #print(Cluster[b].so_orb_list)
    #print("       c ")
    #print(Cluster[c].so_orb_list)
    #print("       d ")
    #print(Cluster[d].so_orb_list)

    for i,I in enumerate(Cluster[a].orb_list):
        for j,J in enumerate(Cluster[b].orb_list):
            for k,K in enumerate(Cluster[c].orb_list):
                for l,L in enumerate(Cluster[d].orb_list):
                    g_bl[i,j,k,l] = tei[I,J,K,L]

    return g_bl
# }}}

def form_Heff(blocks,Cluster,tei):
# {{{
    n_blocks = len(blocks)
    VJa = {}
    VKa = {}
    for a in range(0,n_blocks):
        #print("Cluster: ",a)

        for b in range(0,n_blocks):
            if b != a:

                gaabb = get_block_eri_2(blocks,Cluster,tei,a,a,b,b)
                gabba = get_block_eri_2(blocks,Cluster,tei,a,b,b,a)
                #print(gabab[1,:,1,:])

                Jtemp = np.einsum('prqs,qs->pr',gaabb,Cluster[b].tdm["ca_aa"])
                Ktemp = np.einsum('psqr,qs->pr',gabba,Cluster[b].tdm["ca_aa"])

                VJa[a,b] = Jtemp
                VKa[a,b] = Ktemp

                #print("Couloumb\n",Jtemp)
                #print("Exchange\n",Ktemp)
    return VJa,VKa
# }}}

def run_cmf_iter(blocks,Cluster,tei):
# {{{
    n_blocks = len(blocks)
    
    VJa, VKa = form_Heff(blocks,Cluster,tei)

    hnew =  {}
    mat =   {}

    bl_vec = {}

    EE = 0
    for a in range(0,n_blocks):

        damp = True
        damp = False
        if damp == True:
            dd = Cluster[a].p_evec.T @ Cluster[a].p_evec
            print(dd)

        #Cluster[a].h_eff = cp.deepcopy(Cluster[a].oei)
        heff = cp.deepcopy(Cluster[a].oei)
        for b in range(0,n_blocks):
            temp = 0
            if a != b:

                #Cluster[a].h_eff +=  VJa[a,b] - VKa[a,b]
                heff +=  2 * VJa[a,b] - VKa[a,b]

        #print("effective part")
        #print(heff)
        norb_a  = Cluster[a].n_orb
        ha = Cluster[a].oei
        ga = Cluster[a].tei

        n_a = Cluster[a].n_a
        n_b = Cluster[a].n_b

        if n_a+n_b != 0:
            from pyscf import fci
            cisolver = fci.direct_spin0.FCI()
            efci, ci = cisolver.kernel(heff, ga, norb_a, (n_a,n_b), ecore=0,nroots =1,verbose=100)
            print(efci)
            #efci = efci[0]
            #ci = ci[0]
            fci_dim = ci.shape[0]*ci.shape[1]
            ci = ci.reshape(1,fci_dim)
        else:
            efci, ci = cisolver.kernel(heff, ga, norb_a, (n_a,n_b), ecore=0,nroots =1,verbose=6)
            #print(ci)
            print(efci)
            fci_dim = ci.shape[0]*ci.shape[1]
            ci = ci.reshape(1,fci_dim)

        ## make tdm
        d1,d2 = cisolver.make_rdm1s(ci, ha.shape[1], (n_a,n_b))
        Cluster[a].store_tdm("ca_aa",d1)
        Cluster[a].store_tdm("ca_bb",d2)

        if damp == True:
            cv =  ci.reshape(1,fci_dim)
            d2 = cv.T @ cv
            ee, ci = np.linalg.eigh(0.5 * d2 + 0.5 * dd)

        ## fci
        Cluster[a].tucker_vecs(ci,efci) #Only one P vector right now

        

        EE += efci

        print("Cluster  :%6d" %(a))
        print("     Energy                      : %16.10f" %(efci))

    return EE, bl_vec
# }}}

def double_counting(blocks,Cluster,eri):
# {{{

    #eri = eri.swapaxes(1,2)
    n_blocks = len(blocks)
    ####Double counting in MF 
    e_double = 0
    e_d = np.zeros((n_blocks,n_blocks))
    e_extra2 = 0
    E1_cmf = 0
    E2_cmf = 0
    Eij = 0
    for a in range(0,n_blocks):

        for b in range(0,a):
            if a != b:
                gaabb = get_block_eri_2(blocks,Cluster,eri,a,a,b,b)
                gabba = get_block_eri_2(blocks,Cluster,eri,a,b,b,a)

                Pi = Cluster[a].tdm["ca_aa"]
                Pj = Cluster[b].tdm["ca_aa"]

                Eij += np.einsum("pqrs,pq,rs->",gaabb,Pi,Pj)
                Eij -= np.einsum("psrq,pq,rs->",gabba,Pi,Pj)
                #print(Eij)

                Pi = Cluster[a].tdm["ca_bb"]
                Pj = Cluster[b].tdm["ca_bb"]

                Eij += np.einsum("pqrs,pq,rs->",gaabb,Pi,Pj)
                Eij -= np.einsum("psrq,pq,rs->",gabba,Pi,Pj)
                #print(Eij)

                Pi = Cluster[a].tdm["ca_aa"]
                Pj = Cluster[b].tdm["ca_bb"]

                Eij += np.einsum("pqrs,pq,rs->",gaabb,Pi,Pj)
                #print(Eij)

                Pi = Cluster[a].tdm["ca_bb"]
                Pj = Cluster[b].tdm["ca_aa"]

                Eij += np.einsum("pqrs,pq,rs->",gaabb,Pi,Pj)
                #print("EEij",Eij)
                e_d[a,b] = Eij

    #print(e_d)
    return Eij
# }}}

def run_cmf(h,g,blocks,fspace,ecore=0,miter=50):
    # {{{

    print("-------------------------------------------------------------------")
    print("                     Start of n-body Tucker ")
    print("-------------------------------------------------------------------")
    n_blocks = len(blocks)
    Ecmf = 0

    n_orb = h.shape[0]

    assert(n_blocks == len(blocks))
    size_blocks = [len(i) for i in blocks]

    print("\nNumber of Blocks                               :%10i" %(n_blocks))
    print("\nBlocks :",blocks,"\n")



    #initialize the cluster class
    print("Initialize the clusters:")
    cluster = {}
    for a in range(n_blocks):
        ha, ga = get_cluster_eri(blocks[a],h,g)  #Form integrals within a cluster
        cluster[a] = Cluster()
        cluster[a].init(blocks[a],fspace[a],ha,ga)
        print(ha)


    ###One cluster Terms
    EE = 0
    d1 = {}

    for a in range(n_blocks):

        norb_a  = cluster[a].n_orb
        ha = cluster[a].oei
        ga = cluster[a].tei

        n_a = fspace[a][0]
        n_b = fspace[a][1]

        if 0:
            H = run_fci(ha,ga,norb_a, n_a, n_b)
            S2 = form_S2(norb_a, n_a, n_b)
            efci2,ci2 = sparse.linalg.eigsh(H +  S2, k = 4,which = 'SA')
            print("ham")
            print(H)
            #print(efci)
            #print(ci.T @ H @ ci)
            #print(ci.T @ S2 @ ci)
            #print(ci)
            #ci = ci.reshape(nCr(norb_a,n_a),nCr(norb_a,n_b))

        if 1:
            if n_a+n_b != 0:
                from pyscf import fci
                cisolver = fci.direct_spin0.FCI()
                efci, ci = cisolver.kernel(ha, ga, norb_a, (n_a,n_b), ecore=0,nroots =1,verbose=100)
                print(efci)
                #efci = efci[0]
                #ci = ci[0]
                fci_dim = ci.shape[0]*ci.shape[1]
                ci = ci.reshape(1,fci_dim)
            else:
                efci, ci = cisolver.kernel(ha, ga, norb_a, (n_a,n_b), ecore=0,nroots =1,verbose=6)
                #print(ci)
                print(efci)
                fci_dim = ci.shape[0]*ci.shape[1]
                ci = ci.reshape(1,fci_dim)

            ## fci
            cluster[a].tucker_vecs(ci,efci) #Only one P vector right now

            ## make tdm
            d1,d2 = cisolver.make_rdm1s(ci, ha.shape[1], (n_a,n_b))
            cluster[a].store_tdm("ca_aa",d1)
            cluster[a].store_tdm("ca_bb",d2)

        
        print("Diagonalzation of each cluster local Hamiltonian    %16.8f:"%efci)

        EE += efci

        print("cluster  :%6d" %(a))
        print("     Energy                      : %16.10f" %(efci))

        

    e_double = double_counting(blocks,cluster,g)
    print("Ground State nbT-0                             :%16.10f"   %(EE+ecore+e_double) )

    EE_old = EE
    Ecmf = EE + ecore + e_double
    converged = False

    print("\nBegin cluster Optimisation...\n")
    #print("    Iter                     Energy                    Error ")
    #print("-------------------------------------------------------------------")
    for i in range(0,miter):
        EE_old = EE
        print("-------------------------------------------------------------------")
        print(" CMF Iteration            : %4d" %i)
        print("-------------------------------------------------------------------")

        EE,bl_vec = run_cmf_iter(blocks,cluster,g)

        print("  %4i    Energy: %16.12f         Error:%16.12f" %(i,EE_old,EE - EE_old))
        print()
        if abs(EE - EE_old) < 1E-8:

            print("")
            print("CMF energy converged...")


            e_double = double_counting(blocks,cluster,g)

            print("Sum of Eigenvalues                   :%16.10f"%(EE))
            print("Nuclear Repulsion                    :%16.10f " %ecore)
            print("Removing Double counting...          :%16.10f"%(e_double))
            print("")
            Ecmf = EE + ecore - e_double
            #print("SCF                                         :%16.10f"%(Escf))
            print("CMF                                         :%16.10f"%(Ecmf))
            print("")
            converged = True
            break
        else:
            e_double = double_counting(blocks,cluster,g)
            print("Sum of Eigenvalues                   :%16.10f"%(EE))
            print("Nuclear Repulsion                    :%16.10f " %ecore)
            print("Removing Double counting...          :%16.10f"%(e_double))
            print("")
            Ecmf = EE + ecore - e_double
            #print("SCF                                         :%16.10f"%(Escf))
            print("CMF                                         :%16.10f"%(Ecmf))
    if miter==0:
        print("Energy for the reference state computed")
        print("Sum of Eigenvalues                   :%16.10f"%(EE))
        print("Nuclear Repulsion                    :%16.10f " %ecore)
        print("Removing Double counting...          :%16.10f"%(e_double))
        print("DPS                                         :%16.10f"%(Ecmf))
        print(" -------CMF did not converge--------")

    elif converged == False:
        print(" -------CMF did not converge--------")
    print("-------------------------------------------------------------------")
    print("                                                                   ")
    print("                   False modesty is a lie ")
    print("                                                                      ")
    print("                      - Ratatouille")
    print("                                                                   ")
    print("-------------------------------------------------------------------")


    #wfn.Ca().copy(psi4.core.Matrix.from_array(C))
    #wfn.Cb().copy(psi4.core.Matrix.from_array(C))

    #psi4.molden(wfn,"psi4.molden")

    return Ecmf
# }}} 

def lowdin(S):
# {{{
    print("Using lowdin orthogonalized orbitals")
    #forming S^-1/2 to transform to A and B block.
    sal, svec = np.linalg.eigh(S)
    idx = sal.argsort()[::-1]
    sal = sal[idx]
    svec = svec[:, idx]
    sal = sal**-0.5
    sal = np.diagflat(sal)
    X = svec @ sal @ svec.T
    return X
# }}}
