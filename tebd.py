import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as la
from scon import scon

###############################################################################
#INTERFACE
###############################################################################
class SpinWaveTEBD:
    """Represent a spin wave using the TEBD technique. This class is designed
       for direct interface with user code. The constructor accepts the
       following arguments:
        chain -> A list of integers specifying the initial state of the spin
                 chain. This is assumed to be a product state such that each
                 qubit is parallel to an axis of the computational basis. The
                 integers in 'chain' specify that axis. For example
                 chain = [0, 1, 1, 0, 0] would specify the state
                 |psi> = |01100>.
        chi   -> The bond dimension of the simulation.
        delta -> Length of a timestep. This is fixed.
        hdim  -> The dimension of the Hilbert space. Only '2' supported for now,
                 which is the default argument.

       After construction the simulation is performed using method 'evolve'.
       Evolve takes two arguments: the timestep T to be advanced to, and the
       link l whose eigenvalues are to be returned.

       Method 'printnetwork' outputs detailed information about the current
       state of the spin chain. 'printnetwork(l)' outputs the tensors
       for the l'th qbit only.
    """
    def __init__(self, chain, chi, delta, utype="identity", hdim=2):
        if hdim != 2:
            raise NotImplementedError("Higher-dim Hilbert space unimplemented.")
        self.thist = 0.
        self.step = 0
        self.chi = chi
        self.n = len(chain)
        self.delta = delta
        self.hdim = hdim
        self.network = TensorNetwork(chain, chi, hdim=hdim)
        if utype == "pauli":
            self.ops = PauliSpinChainOperators(delta, 1, 1)
        # if utype == "identity":
            # self.ops = SpinChainEyes(len(chain), delta)
        else:
            raise NotImplementedError("Only identity currently supported.")
        self.__initprint(utype, chain)

    def __initprint(self, utype, chain):
        chainstr = ""
        for c in chain:
          chainstr+=str(c)
        print "*************************************************************"
        print "Initializing TEBD spin wave simulation."
        print "Bond dimension: ", self.chi
        print "Hilbert space dimension: ", self.hdim
        print "Time-evolution operator type: ", utype
        print "Number of qbits: ", self.n
        print "Initial state: |"+chainstr+">"
        print "Timestep size: ", self.delta
        print "Initial time: ", self.thist
        print "*************************************************************"

    def evolve(self, T, trackqbit, verbose=False):
        if trackqbit > self.n-1:
            raise ValueError("You asked to track a qbit whose index was"+
                             " greater than the length of the chain.")
        print "Beginning evolution..."
        print "Initial time: ", self.thist
        print "Final time: ", T
        print "*************************************************************"
        output = []
        times = []
        while (self.thist + self.delta) < T:
            outstr = "Step: " + str(self.step) + ", Time: " + str(self.thist)
            outstr += "/" + str(T) 
            print outstr
            self.__advance(verbose)
            outevs = list(self.network.evs(trackqbit))
            output.append([self.thist,] + outevs)
            # times.append(self.thist)
            # output.append(outevs)

        #tarr = np.array(times)
        outarr = np.array(output)
        print "Evolution finished!"
        return outarr

    def __advance(self, verbose):
        """Advance the on one timestep. 
        """
        for l in range(0, self.n):
            if verbose:
                print "qbit", l, ":"
            theseops = self.ops(l)
            for op in theseops:
                self.network.transform(l, op, verbose)
            if verbose:
                print "\tApplied the operators successfully."
        self.thist += self.delta
        self.step += 1

    def printnetwork(self, l=None):
        if l is None:
            return str(self.network)
        else:
            return self.network.lthnodestring(l)
###############################################################################

class TensorNetwork:
    """Represent the tensor network. TensorNetwork(l) returns the l'th
       coefficient (a reference to the l'th tensor and vector, as 
       appropriate). 
    """

    def __init__(self, chain, chi, hdim=2):
        """Constructs initial decomposition of a 1D product state.
           'chain' is a numpy array or list representing the initial ket in 
           the computational basis. The Hilbert space is assumed to be a product
           of Hilbert spaces with dimension hdim. chi is the maximum bond 
           dimension which the simulation will allow.
           e.g. chain=[1, 1, 0, 0, 0], chi=5, hdim=2 generates
           |11000>, where only 0 and 1 are allowed, with bond dim 5.
        """
        self.chi = chi
        self.hdim = hdim
        self.n = len(chain)
        self.coefs = [] 
        for ket, l in zip(chain, range(0, self.n)):
            if not (0 <= ket <= hdim):
                raise ValueError("Invalid chain for hdim = " + str(hdim))
            self.coefs.append(self.__makecoef(ket, l, self.n))
     
    def evs(self, l):
        """Return the eigenvalues p_a = |lam^l_a|^2 for the l'th link
        """
        return np.square(np.absolute(self.coefs[l].lam))

    def __makecoef(self, ket, l, n):
        """Generate the l'th coefficient (gamma and lambda) in the initial
           chain. The first, last, and middle values differ qualitatively,
           which we deal with using inheritance.
        """
        if (0 == l): 
            return FirstTensorCoef(ket, self.chi, self.hdim)
        if ((n-1)==l): 
            return LastTensorCoef(ket, self.chi, self.hdim)
        return TensorCoef(ket, self.chi, self.hdim)

    def __getitem__(self, l):
        return self.coefs[i]

    def __repr__(self):
        return str(self)
    
    def __str__(self):
        thestring = "*******PRINTING TENSOR NETWORK*******\n"
        thestring += "chi = " + str(self.chi)
        thestring += "\nhdim = " + str(self.hdim)
        thestring += "\nn = " + str(self.n)
        thestring += "\n\n**CHAIN:**"
        for coef, l in zip(self.coefs, range(0, self.n)):
            thestring += "\n\nl= " + str(l) + ": " 
            thestring += "\n****************************"
            thestring += str(coef)
            thestring += "\n****************************"
        thestring += "\n*******DONE PRINTING*******\n"
        return thestring

    def lthnodestring(self, l):
        return str(self.coefs[l])

    def __single_transform(self, l, U, verbose=False):
        atup = (U, self.coefs[l].gamma)
        U_idx = [-1, 1]
        if (len(self.coefs[l].gamma.shape)==2):
            G_idx = [1, -2]
        else:
            G_idx = [1, -2, -3]
        idx = (U_idx, G_idx)
        newgamma = scon(atup, idx)
        self.coefs[l].setgamma(newgamma) 
        if verbose:
            print "\t\tUpdated gamma^"+str(l)+"."

    def __newgammaD(self, theta, l):
        """ Apply Eqns. 22-24 in Vidal 2003 to update gamma^D 
            (gamma of the next qbit).
        """
        rhoDK = self.__rhoDK(theta, self.coefs[l-1].lam)    
        #diagonalize
        idx = self.chi * self.hdim
        rhoDKflat = rhoDK.reshape([idx, idx]) 
        evals, evecs = la.eigh(rhoDKflat) #note rho is a density matrix and thus
                                          #hermitian
        evals = evals[:self.chi]
        evecs = evecs[:,:self.chi]
        return evecs

    def __newlambdagammaC(self, theta, l):
        """ Apply Eqns. 25-27 in Vidal 2003 to update lambda^C and gamma^C
            (lambda and gamma of this qbit).
        """
        gamma_ket = self.coefs[l+1].lam
        gamma_bra = np.conjugate(gamma_ket)
        Gamma_star = np.conjugate(self.coefs[l+1].gamma)
        inputs = [Gamma_star, theta, gamma_bra, gamma_ket]
        Gamma_star_idx = [1, -3, -2]
        theta_idx = [-1, 1, -4, -5]
        gamma_bra_idx = [-6]
        gamma_ket_idx = [-7]
        idx = [Gamma_star_idx, theta_idx, gamma_bra_idx, gamma_ket_idx]
        contract_me = scon(inputs, idx)
        svd_me = np.einsum('agibggg', contract_me)
        evals, evecs = la.eigh(svd_me)
        return evals, evecs
        
    def __double_transform(self, l, V, verbose=False):
        """ Appply the operations from Eqn. 22-27 in Vidal 2003 in order to 
            update gamma^C, lambda^C, and gamma^D, where C refers to the present
            qbit and D to the one to its immediate right.
        """
        #Compute theta and update gamma^D.
        theta = self.__theta_ij(V, self.coefs[l], self.coefs[l+1])
        newgammaD = self.__newgammaD(theta, l)
        newgammaD.shape = self.coefs[l+1].gamma.shape
        self.coefs[l+1].setgamma(newgammaD)
        if verbose:
            print "\t\tUpdated gamma^"+str(l+1)+"."

        #Updata lambda^C, gamma^C.
        newlambdaC, newgammaC = self.__newlambdagammaC(theta, l)
        self.coefs[l].setlambda(newlambdaC[0, :])
        if verbose:
            print "\t\tUpdated lambda^"+str(l)+"."
        newgammaC.shape = self.coefs[l].gamma.shape
        self.coefs[l].setgamma(newgammaC)
        if verbose:
            print "\t\tUpdated gamma^"+str(l)+"."



    def __theta_ij(self, V, coef_l, coef_r):
        """ Compute theta^ij_ac 
          =sum(b) sum(k,l){ V^ij_kl G^Ck_ab L_b G^Dl_bc}
          (Eq. 22 of Vidal 2003)
        """
        Gc = coef_l.gamma
        lam = coef_l.lam
        Gd = coef_r.gamma
        atup = (V, Gc, lam, Gd)
        V_idx = [-1, -2, 1, 2]
        Gc_idx = [1, -3, -5]
        L_idx = [-6]
        Gd_idx = [2, -7, -4]
        idx = (V_idx, Gc_idx, L_idx, Gd_idx)
        out_one = scon(atup, idx)
        out_two = np.einsum('ijklmmm', out_one)
        return out_two
    
    def __rhoDK(self, theta, lam):
        """ Does the diagonalization in Eq.23 of Vidal 2003.
        """
        theta_star = np.conjugate(theta) 
        lam_bra = np.conjugate(lam)
        inputs = (lam_bra, lam, theta, theta_star)
        lam_bra_idx = [-5]
        lam_idx = [-6]
        theta_idx = [2, -1, -7, -2]
        theta_star_idx = [2, -3, -8, -4]
        idx = (lam_bra_idx, lam_idx, theta_idx, theta_star_idx)
        bracket_one = scon(inputs, idx)
        rho = np.einsum('abcdeeee', bracket_one)
        return rho

    def transform(self, l, operator, verbose=False):#single, double=None):
        """Apply an operator to the l'th and, optionally,
           the l+1th link. The former transform is performed with the operator
           fed in through the argument 'single'. The latter is performed with
           the argument 'double'. If the latter is left 'None' only the 
           'single' transform is performed.
        """
        if operator.issingle:
            self.__single_transform(l, operator.data, verbose)
        else:
            if l!=0 and l-self.n > 2:
                self.__double_transform(l, operator.data, verbose)
    

###############################################################################
#SpinChainOperators - operators representing the time-evolution of the spin
#chain.
###############################################################################

class Operator:    
    def __init__(self, data, issingle=True):
        self.issingle=issingle
        self.data = data

class PauliSpinChainOperators:
    def __init__(self, delta, B, J):
        self.pauli00 = paulidouble(0, 0)
        self.pauli11 = paulidouble(1, 1)
        self.pauli22 = paulidouble(2, 2)
        self.pauli33 = paulidouble(3, 3)
        self.even = self.__evenops(delta, B, J)
        self.odd = self.__oddops(delta, B, J)
        self.evensingle = self.__evensingle(delta, B)
        self.oddsingle = self.__oddsingle(delta, B)

    def __oddops(self, delta, B, J):
        leftop = self.__oddsingle(delta, B)[0]
        cosJ = np.cos(J*delta)
        sinJ = 1.j*np.sin(J*delta)
        right = cosJ*self.pauli11 + sinJ*self.pauli11
        right += cosJ*self.pauli22 + sinJ*self.pauli22
        right += cosJ*self.pauli33 + sinJ*self.pauli33
        rightop = Operator(right, issingle=False)
        return [leftop, rightop]

    def __oddsingle(self, delta, B):
        cosB = np.cos(B*delta)
        sinB = 1.j*np.sin(B*delta)
        left = cosB*pauli(0) + sinB*pauli(3)  
        leftop = Operator(left, issingle=True)
        return [leftop]

    def __evenops(self, delta, B, J):
        leftop = self.__evensingle(delta, B)[0]
        cosJ = np.cos(J*delta/2.)
        sinJ = 1.j*np.sin(J*delta/2.)
        right = cosJ*self.pauli11 + sinJ*self.pauli11
        right += cosJ*self.pauli22 + sinJ*self.pauli22
        right += cosJ*self.pauli33 + sinJ*self.pauli33
        rightop = Operator(right, issingle=False)
        return [leftop, rightop, leftop, rightop]

    def __evensingle(self, delta, B):
        cosB = np.cos(B*delta/2.)
        sinB = 1.j*np.sin(B*delta/2.)
        left = cosB*pauli(0) + sinB*pauli(3)  
        leftop = Operator(left, issingle=True)
        return [leftop]

    def __call__(self, l):
        if l%2 == 0:
            return self.even
        else:
            return self.odd
          
            
class SpinChainOperators:
    """Interface class. Method 'single' returns the operator acting on the 
       l'th link. Method 'double' returns the operator acting on the l'th
       and l+1'th links.
    """
    def sGet(self, l):
        if l<0 or l >= self.n :
            raise IndexError("Index " + str(l) + " out of bound " + str(self.n))
        return self.single[l]

    def dGet(self, l):
        #Note > rather than >=.
        if l<0 or l > self.n :
            raise IndexError("Index " + str(l) + " out of bound " + str(self.n))
        return self.double[l]
    
class SpinChainEyes(SpinChainOperators):
    """Identity map with same data structure as SpinChainOps. For testing.
    """
    def __init__(self, n, delta, B=1, J=1):
        self.n = n
        sElem = np.eye(2)
        self.single = [sElem]*n

        dElem = np.zeros((2, 2, 2, 2))
        self.double = [dElem]*(n-1)
        self.double.append(None)
    
class SpinChainPauli(SpinChainOperators):
    """Represent the time-evolution operators of a quantum spin chain.
    """
    def __init__(self, n, delta, B=1, J=1):
        # self.n = n
        # deven = delta/2
        # dodd = delta
        # single_even = pauli(0)*np.cos(B*deven) + 1.0j*pauli(3)*np.sin(B*deven)
        # single_odd = pauli(0)*np.cos(B*dodd) + 1.0j*pauli(3)*np.sin(B*dodd)
        # pauli00 = paulidouble(0, 0)
        # pauli11 = paulidouble(1, 1)
        # pauli22 = paulidouble(2, 2)
        # pauli33 = paulidouble(3, 3)

        #for i in range(0, n-1):

        # self.single = [B*pauli(3)]*n
        # self.double = [None]*(n-1)

        # self.double = [1]*n
        # psum = pauli(0) + pauli(1) + pauli(2)
        # pz = pauli(2)
        # s
      
        # for single, double in zip(self.single, self.double):
            # U = np.zeros((2, 2, 2, 2))
            # if 0 == l % 2: #even
              # scalar = 2.*np.cos(delta/2. * B) + 2.*np.cos(delta/2. * J)
              # left = 2.j * np.sin(delta/2. * B) * pz
              # left += 2.j * np.sin(delta/2. * J) * psum
              # right = 2.j * np.sin(delta/2. * J) * psum
              
            # else: #odd 
              # scalar = np.cos(delta * B) + np.cos(delta * J)
              # left = 1.j * np.sin(delta * B) * pz
              # left += 1.j * np.sin(delta * J) * psum
              # right = 1.j * np.sin(delta * J) * psum
        self.double.append(None)
###############################################################################


###############################################################################
#TensorCoef - represents individual links in the tensor network. Three different
#classes (with identical interfaces) are used; FirstTensorCoef, TensorCoef,
#and LastTensorCoef, representing the first, middle, and last qubits 
#respectively.
class TensorCoef:
    """Represent the 'coefs' (Gamma and Lambda) for a particular value of
       l in a tensor network. This class represents coefficients in the
       middle of the chain: Gamma^i_a1,a2 Lambda_a1, i->{0, hdim-1},
       a->{0, chi-1}.
       The shapes are: gamma^i_jk -> (i, j, k) -> (hdim, chi, chi)
                       lambda^i -> (i) -> chi
                       gamma^i_j -> (i, j) -> (hdim, chi)
    """
    def __init__(self, ket, chi, hdim):
        self.hdim = hdim
        self.chi = chi
        self.gamma = self.makegamma(ket, chi, hdim)
        self.lam = self.makelambda(ket, chi, hdim)

    def __repr__(self):
        return str(self)
    
    def __str__(self):
        thestring = ""
        for i in range(0, self.hdim):
            thestring += "\nGamma[i="+str(i)+"]:\n"
            thestring += str(self.gamma[i, :, :])
        thestring += "\nLambda: \n"
        thestring += str(self.lam[:])
        return thestring

    def setgamma(self, newgamma):
        if newgamma.shape != self.gamma.shape:
            raise ValueError(('Shape of newgamma=%i differs from shape of'
                              'oldgamma=%i.')%newgamma.shape, self.gamma.shape)
        self.gamma = np.copy(newgamma)

    def setlambda(self, newlambda):
        if newlambda.shape != self.lam.shape:
            raise ValueError(('Shape of newlambda=%i differs from shape of'
                              'oldlambda=%i.')%newlambda.shape, self.lam.shape)
        self.lam = np.copy(newlambda)
    
    def makegamma(self, ket, chi, hdim):
        gamma = np.zeros((hdim, chi, chi), dtype=np.complex)
        gamma[ket, 0, 0] = 1.
        return gamma

    def makelambda(self, ket, chi, hdim):
        lam = np.zeros(chi, dtype=np.complex)
        lam[ket] = 1.
        return lam

###############################################################################
class FirstTensorCoef(TensorCoef):
    """Represents the first link in the chain:
      Gamma^i_a1 Lambda_a1, i->{0, hdim-1}, a->{0, chi-1}.
    """
    def __str__(self):
        thestring = ""
        for i in range(0, self.hdim):
            thestring += "\nGamma[i="+str(i)+"]:\n"
            thestring += str(self.gamma[i, :])
        thestring += "\nLambda: \n"
        thestring += str(self.lam[:])
        return thestring

    def makegamma(self, ket, chi, hdim):
        gamma = np.zeros((hdim, chi), dtype=np.complex)
        gamma[ket, 0] = 1.
        return gamma

    def makelambda(self, ket, chi, hdim):
        lam = np.zeros(chi, dtype=np.complex)
        lam[ket] = 1.
        return lam 

###############################################################################
class LastTensorCoef(TensorCoef):
    """The last link in the chain: Gamma^i_a1 (lambda just 1)
    """
    def __str__(self):
        thestring = ""
        for i in range(0, self.hdim):
            thestring += "\nGamma[i="+str(i)+"]:\n"
            thestring += str(self.gamma[i, :])
        thestring += "\nLambda: \n"
        thestring += str(self.lam)
        return thestring

    def makegamma(self, ket, chi, hdim):
        gamma = np.zeros((hdim, chi), dtype=np.complex)
        gamma[ket, 0] = 1.
        return gamma

    def makelambda(self, ket, chi, hdim):
        return 1 

    def setlambda(self, newlambda):
        raise NotImplementedError("No lambda for last coef.")
###############################################################################
    

def pauli(i):
    """ Return the i'th Pauli matrix.
    """
    if (0==i):
        return np.eye(2, dtype=np.complex)
    elif (1==i):
        return np.array([[0,1], [1, 0]], dtype=np.complex)
    elif (2==i):
        return np.array([[0, -1.j], [1.j, 0]], dtype=np.complex)
    elif (3==i):
        return np.array([[1, 0], [0, -1]], dtype=np.complex)
    else:
        raise ValueError("Invalid Pauli matrix index " + str(i))

def paulidouble(i, j, tensor=True):
    pauli_i = pauli(i)
    pauli_j = pauli(j)
    outer = np.zeros((4, 4), dtype=np.complex)
    outer[:2, :2] = pauli_i
    outer[2:, 2:] = pauli_j
    if tensor:
        outer.shape = (2, 2, 2, 2)
    return outer

# def paulitwo_left(i):
    # return np.kron(pauli(i), pauli(0)) 

# def paulitwo_right(i):
    # return np.kron(pauli(0), pauli(i))

# def newrho_DK(Lket, theta_ij):
    # Lbra = np.conjugate(L_before)
    # theta_star = np.conjugate(theta_ij)
    # in_bracket = scon(Lbra, Lket, theta_ij, theta_star,
                # [1], [1], [2, 3, 1, 

if __name__=="__main__":
    n = 30
    chi = n/2 + 2
    chain = [1, 1]
    for i in range(0, n-2):
        chain.append(0)
    spinwave = SpinWaveTEBD(chain, chi, 0.005, hdim=2, utype="pauli") 
    evs = spinwave.evolve(25, 15, verbose=False)
    t = evs[:, 0]
    for i in range(1, evs.shape[1]):
        plt.semilogy(t, evs[:, i], label=str(i))
    plt.xlabel("Time")
    plt.ylabel("Spectrum")
    #plt.ylim((-0.1, 1.1))
    plt.legend(loc="best")
    plt.show()
    #print evs
