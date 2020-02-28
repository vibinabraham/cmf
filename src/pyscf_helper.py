import numpy as np
import scipy
import itertools
import time
from math import factorial
import copy as cp
import sys

class PyscfHelper(object):
    """
    Pyscf is used to generate 
    integrals etc for the TPSCI program. this is a class which keeps info about pyscf mol info and HF stuff
    """

    def __init__(self):

        self.mf  = None
        self.mol = None

        self.h      = None
        self.g      = None
        self.n_orb  = None
        #self.na     = 0
        #self.nb     = 0
        self.ecore  = 0
        self.C      = None
        self.S      = None
        self.J      = None
        self.K      = None
        self.Escf = None

    def init(self,molecule,charge,spin,basis_set,orb_basis='scf',cas=False,cas_nstart=None,cas_nstop=None,cas_nel=None,loc_nstart=None,loc_nstop=None):
    # {{{
        import pyscf
        from pyscf import gto, scf, ao2mo, molden, lo
        pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
        #PYSCF inputs
        print(" ---------------------------------------------------------")
        print("                      Using Pyscf:")
        print(" ---------------------------------------------------------")
        print("                                                          ")

        mol = gto.Mole()
        print("here")
        mol.atom = molecule

        mol.max_memory = 1000 # MB
        mol.symmetry = True
        mol.charge = charge
        mol.spin = spin
        mol.basis = basis_set
        mol.build()
        print("symmertry")
        print(mol.topgroup)

        #SCF 

        #mf = scf.RHF(mol).run(init_guess='atom')
        mf = scf.RHF(mol).run()
        #C = mf.mo_coeff #MO coeffs
        enu = mf.energy_nuc()
        self.Escf = mf.e_tot
       
        print(mf.get_fock())
        print(np.linalg.eig(mf.get_fock())[0])
        
        if mol.symmetry == True:
            from pyscf import symm
            mo = symm.symmetrize_orb(mol, mf.mo_coeff)
            osym = symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo)
            #symm.addons.symmetrize_space(mol, mo, s=None, check=True, tol=1e-07)
            for i in range(len(osym)):
                print("%4d %8s %16.8f"%(i+1,osym[i],mf.mo_energy[i]))

        #orbitals and lectrons
        n_orb = mol.nao_nr()
        n_b , n_a = mol.nelec 
        nel = n_a + n_b
        self.n_orb = mol.nao_nr()


        if cas == True:
            cas_norb = cas_nstop - cas_nstart
            from pyscf import mcscf
            assert(cas_nstart != None)
            assert(cas_nstop != None)
            assert(cas_nel != None)
        else:
            cas_nstart = 0
            cas_nstop = n_orb
            cas_nel = nel

        ##AO 2 MO Transformation: orb_basis or scf
        if orb_basis == 'scf':
            print("\nUsing Canonical Hartree Fock orbitals...\n")
            C = cp.deepcopy(mf.mo_coeff)
            print("C shape")
            print(C.shape)

        elif orb_basis == 'lowdin':
            assert(cas == False)
            S = mol.intor('int1e_ovlp_sph')
            print("Using lowdin orthogonalized orbitals")

            C = lowdin(S)
            #end

        elif orb_basis == 'boys':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :cas_nstart]
            cl_a = lo.Boys(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, cas_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'boys2':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :loc_nstart]
            cl_a = lo.Boys(mol, mf.mo_coeff[:, loc_nstart:loc_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, loc_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'PM':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :cas_nstart]
            cl_a = lo.PM(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, cas_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'PM2':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :loc_nstart]
            cl_a = lo.PM(mol, mf.mo_coeff[:, loc_nstart:loc_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, loc_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'ER':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :cas_nstart]
            cl_a = lo.PM(mol, mf.mo_coeff[:, cas_nstart:cas_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, cas_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'ER2':
            pyscf.lib.num_threads(1)  #with degenerate states and multiple processors there can be issues
            cl_c = mf.mo_coeff[:, :loc_nstart]
            cl_a = lo.ER(mol, mf.mo_coeff[:, loc_nstart:loc_nstop]).kernel(verbose=4)
            cl_v = mf.mo_coeff[:, loc_nstop:]
            C = np.column_stack((cl_c, cl_a, cl_v))

        elif orb_basis == 'ibmo':
            loc_vstop =  loc_nstop - n_a
            print(loc_vstop)

            mo_occ = mf.mo_coeff[:,mf.mo_occ>0]
            mo_vir = mf.mo_coeff[:,mf.mo_occ==0]
            c_core = mo_occ[:,:loc_nstart]
            iao_occ = lo.iao.iao(mol, mo_occ[:,loc_nstart:])
            iao_vir = lo.iao.iao(mol, mo_vir[:,:loc_vstop])
            c_out  = mo_vir[:,loc_vstop:]

            # Orthogonalize IAO
            iao_occ = lo.vec_lowdin(iao_occ, mf.get_ovlp())
            iao_vir = lo.vec_lowdin(iao_vir, mf.get_ovlp())

            #
            # Method 1, using Knizia's alogrithm to localize IAO orbitals
            #
            '''
            Generate IBOS from orthogonal IAOs
            '''
            ibo_occ = lo.ibo.ibo(mol, mo_occ[:,loc_nstart:], iao_occ)
            ibo_vir = lo.ibo.ibo(mol, mo_vir[:,:loc_vstop], iao_vir)

            C = np.column_stack((c_core,ibo_occ,ibo_vir,c_out))

        else: 
            print("Error:NO orbital basis defined")

        molden.from_mo(mol, 'h8.molden', C)

        if cas == True:
            print(C.shape)
            print(cas_norb)
            print(cas_nel)
            mycas = mcscf.CASSCF(mf, cas_norb, cas_nel)
            h1e_cas, ecore = mycas.get_h1eff(mo_coeff = C)  #core core orbs to form ecore and eff
            h2e_cas = ao2mo.kernel(mol, C[:,cas_nstart:cas_nstop], aosym='s4',compact=False).reshape(4 * ((cas_norb), )) 
            print(h1e_cas)
            print(h1e_cas.shape)
            #return h1e_cas,h2e_cas,ecore,C,mol,mf
            self.h = h1e_cas
            self.g = h2e_cas
            self.ecore = ecore
            self.mf = mf
            self.mol = mol
            self.C = cp.deepcopy(C[:,cas_nstart:cas_nstop])
            J,K = mf.get_jk()
            self.J = self.C.T @ J @ self.C
            self.K = self.C.T @ J @ self.C
            if 0:
                h = C.T.dot(mf.get_hcore()).dot(C)
                g = ao2mo.kernel(mol,C,aosym='s4',compact=False).reshape(4*((n_orb),))
                const,heff = get_eff_for_casci(cas_nstart,cas_nstop,h,g)
                print(heff)
                print("const",const)
                print("ecore",ecore)
                
                idx = range(cas_nstart,cas_nstop)
                h = h[:,idx] 
                h = h[idx,:] 
                g = g[:,:,:,idx] 
                g = g[:,:,idx,:] 
                g = g[:,idx,:,:] 
                g = g[idx,:,:,:] 

                self.ecore = const
                self.h = h + heff
                self.g = g 


        elif cas==False:
            h = C.T.dot(mf.get_hcore()).dot(C)
            g = ao2mo.kernel(mol,C,aosym='s4',compact=False).reshape(4*((n_orb),))
            print(h)
            #return h, g, enu, C,mol,mf
            self.h = h
            self.g = g
            self.ecore = enu
            self.mf = mf
            self.mol = mol
            self.C = C
            J,K = mf.get_jk()
            self.J = self.C.T @ J @ self.C
            self.K = self.C.T @ J @ self.C
    # }}}

def run_fci_pyscf( h, g, nelec, ecore=0,nroots=1):
# {{{
    # FCI
    from pyscf import fci
    #efci, ci = fci.direct_spin1.kernel(h, g, h.shape[0], nelec,ecore=ecore, verbose=5) #DO NOT USE 
    cisolver = fci.direct_spin1.FCI()
    efci, ci = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,nroots =nroots,verbose=100)
    fci_dim = ci.shape[0]*ci.shape[1]
    d1 = cisolver.make_rdm1(ci, h.shape[1], nelec)
    print(d1)
    print(" FCI:        %12.8f Dim:%6d"%(efci,fci_dim))
    print("FCI %10.8f"%(efci))
    #for i in range(ci.shape[0]):
    #    for j in range(ci.shape[0]):
    #        print("%20.14f"%(ci[i,j]*ci[i,j]))
    #exit()
            
    return efci,fci_dim
# }}}

def run_hci_pyscf( h, g, nelec, ecore=0, select_cutoff=5e-4, ci_cutoff=5e-4):
# {{{
    #heat bath ci
    from pyscf import mcscf
    from pyscf.hci import hci
    cisolver = hci.SCI()
    cisolver.select_cutoff = select_cutoff
    cisolver.ci_coeff_cutoff = ci_cutoff
    ehci, civec = cisolver.kernel(h, g, h.shape[1], nelec, ecore=ecore,verbose=4)
    hci_dim = civec[0].shape[0]
    print(" HCI:        %12.8f Dim:%6d"%(ehci,hci_dim))
    print("HCI %10.8f"%(ehci))
    return ehci,hci_dim
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

def reorder_integrals(idx,h,g):
# {{{
    h = h[:,idx] 
    h = h[idx,:] 

    g = g[:,:,:,idx] 
    g = g[:,:,idx,:] 
    g = g[:,idx,:,:] 
    g = g[idx,:,:,:] 
    return h,g
# }}}

def e1_order(h,cut_off):
# {{{
    hnew = np.absolute(h)
    hnew[hnew < cut_off] = 0
    np.fill_diagonal(hnew, 0)
    print(hnew)
    import scipy.sparse
    idx = scipy.sparse.csgraph.reverse_cuthill_mckee(
        scipy.sparse.csr_matrix(hnew))
    print(idx)
    idx = idx 
    hnew = hnew[:, idx]
    hnew = hnew[idx, :]
    print("New order")
    print(hnew)
    return idx
# }}}

def ordering(pmol,cas,cas_nstart,cas_nstop,loc_nstart,loc_nstop,ordering='hcore'):
# {{{
    loc_range = np.array(list(range(loc_nstart-cas_nstart,loc_nstop-cas_nstart)))
    #cas_range = range(cas_nstart,cas_nstop)
    out_range = np.array(list(range(loc_nstop-cas_nstart,cas_nstop-cas_nstart)))
    print(loc_range)
    print(out_range)

    h = cp.deepcopy(pmol.h)
    print(h)
    if ordering == 'hcore':
        print("Bonding Active Space")
        hl = h[:,loc_range]
        hl = hl[loc_range,:]
        print(hl)
        idl = e1_order(hl,cut_off = 1e-2)

        ho = h[:,out_range]
        ho = ho[out_range,:]
        print("Virtual Active Space")
        ido = e1_order(ho,cut_off = 1e-2)

        idl = idl 
        ido = ido + loc_nstop - cas_nstart 

    print(idl)
    print(ido)
    idx = np.append(idl,ido)
    print(idx)
    return idx
    # }}}

def ordering_diatomics(mol,C):
# {{{
    ##DZ basis diatomics reordering with frozen 1s

    orb_type = ['s','pz','dz','px','dxz','py','dyz','dx2-y2','dxy']
    ref = np.zeros(C.shape[1]) 

    ## Find dimension of each space
    dim_orb = []
    for orb in orb_type:
        print("Orb type",orb)
        idx = 0
        for label in mol.ao_labels():
            if orb in label:
                #print(label)
                idx += 1

        ##frozen 1s orbitals
        if orb == 's':
            idx -= 2 
        dim_orb.append(idx)
        print(idx)
    

    new_idx = []
    ## Find orbitals corresponding to each orb space
    for i,orb in enumerate(orb_type):
        print("Orbital type:",orb)
        from pyscf import mo_mapping
        s_pop = mo_mapping.mo_comps(orb, mol, C)
        #print(s_pop)
        ref += s_pop
        cas_list = s_pop.argsort()[-dim_orb[i]:]
        print('cas_list', np.array(cas_list))
        new_idx.extend(cas_list) 
        #print(orb,' population for active space orbitals', s_pop[cas_list])

    ao_labels = mol.ao_labels()
    #idx = mol.search_ao_label(['N.*s'])
    #for i in idx:
    #    print(i, ao_labels[i])
    print(ref)
    print(new_idx)
    for label in mol.ao_labels():
        print(label)

    return new_idx
# }}}

class Psi4Helper(object):
    """
    Psi4 is used to generate 
    integrals etc for the TPSCI program. this is a class which keeps info about pyscf mol info and HF stuff
    """

    def __init__(self):

        self.mf  = None
        self.mol = None

        self.h      = None
        self.g      = None
        #self.na     = 0
        #self.nb     = 0
        self.ecore  = 0
        self.C  = None
        self.mo_energy = None

    def init(self,molecule,charge,spin,basis_set,orb_basis='scf',cas=False,cas_nstart=None,cas_nstop=None,cas_nel=None,loc_nstart=None,loc_nstop=None):
    # {{{
        print(" ---------------------------------------------------------")
        print("                      Using Psi4:")
        print(" ---------------------------------------------------------")
        print("                                                          ")
        psi4.set_memory('1000 MB')

        beg = '\n' + str(charge) + ' ' + str(2*spin+1) + '\n'
        end = 'symmetry c1\n'

        mol = psi4.core.Molecule.create_molecule_from_string(beg + molecule + end)

        bas = psi4.core.BasisSet.build(mol, target=basis)

        mints = psi4.core.MintsHelper(bas)

        psi4.set_options({
            'basis': basis,
            'scf_type': 'pk',
            'mp2_type': 'conv',
            'e_convergence': 1e-10,
            'preconditioner':'gen_davidson',
            'd_convergence': 1e-10
        })
        #'diag_method':'rsp',


        psi4.core.be_quiet()
        #run scf
        energy, wfn = psi4.energy('SCF', return_wfn=True, molecule=mol)

        C = np.asarray(wfn.Ca())
        S = np.array(mints.ao_overlap())
        V = np.array(mints.ao_potential())
        T = np.array(mints.ao_kinetic())
        hcore = T + V
        g = np.array(mints.ao_eri())
        enu = mol.nuclear_repulsion_energy()


        #orbitals and lectrons
        n_orb = wfn.nmo()
        n_a = wfn.nalpha()
        n_b = wfn.nbeta()
        nel = n_a + n_b

        print("Basis set                                      :%12s" %(basis))
        print("Number of Orbitals                             :%10i" %(n_orb))
        print("Number of electrons                            :%10i" %(n_a + n_b))
        print("Nuclear Repulsion                              :%16.10f " %enu)
        print("Electronic SCF energy                          :%16.10f " %(energy-enu))
        print("SCF Energy                                     :%16.10f"%(energy))


        assert(cas==False)
        cas_nstart = 0
        cas_nstop = n_orb
        cas_nel = nel

        ##AO 2 MO Transformation: orb_basis or scf
        if orb_basis == 'scf':
            print("\nUsing Canonical Hartree Fock orbitals...\n")
            C = np.asarray(wfn.Ca())
            print("C shape")
            print(C.shape)

        elif orb_basis == 'lowdin':
            assert(cas == False)
            S = np.array(mints.ao_overlap())
            print("Using lowdin orthogonalized orbitals")

            C = lowdin(S)
            #end

        elif orb_basis == 'boys':
            loc = psi4.core.Localizer.build('BOYS', wfn.basisset(), C)
            loc.localize()
            loc.L.print_out()
            wfn.Ca().copy(psi4.core.Matrix.from_array(loc.L))
            wfn.Cb().copy(psi4.core.Matrix.from_array(loc.L))
            C = np.asarray(wfn.Ca())

        molden.from_mo(mol, 'h8.molden', C)

        if cas == True:
            print(C.shape)
            print(cas_norb)
            print(cas_nel)
            mycas = mcscf.CASSCF(mf, cas_norb, cas_nel)
            h1e_cas, ecore = mycas.get_h1eff(mo_coeff = C)  #core core orbs to form ecore and eff
            h2e_cas = ao2mo.kernel(mol, C[:,cas_nstart:cas_nstop], aosym='s4',compact=False).reshape(4 * ((cas_norb), )) 
            print(h1e_cas)
            print(h1e_cas.shape)
            #return h1e_cas,h2e_cas,ecore,C,mol,mf
            self.h = h1e_cas
            self.g = h2e_cas
            self.ecore = ecore
            self.mf = mf
            self.mol = mol
            self.C = C
            self.mo_energy = mf.mo_energ[cas_nstart:cas_nstop]

        elif cas==False:
            V = np.array(mints.ao_potential())
            T = np.array(mints.ao_kinetic())
            hcore = T + V
            h = C.T.dot(hcore).dot(C)
            g = np.array(mints.ao_eri())
            g = np.einsum("pqrs,pl->lqrs",g,C)
            g = np.einsum("lqrs,qm->lmrs",g,C)
            g = np.einsum("lmrs,rn->lmns",g,C)
            g = np.einsum("lmns,so->lmno",g,C)
            print(h)
            #return h, g, enu, C,mol,mf
            self.h = h
            self.g = g
            self.ecore = enu
            self.mf = wfn
            self.mol = mol
            self.C = C
            self.mo_energy = mf.mo_energ[cas_nstart:cas_nstop]
    # }}}
