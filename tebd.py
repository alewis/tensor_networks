import numpy as np
import matplotlib.pyplot as plt
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
       Evolve takes a single argument T: the time value to advance to.

       Method 'printnetwork' outputs detailed information about the current
       state of the spin chain.
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
        if utype == "identity":
            self.ops = SpinChainEyes(len(chain), delta)
        elif utype == "pauli":
            self.ops = SpinChainPauli(len(chain), delta)
        else:
            raise NotImplementedError("Only identity currently supported.")

    def evolve(self, T):
        while (self.thist < T):
            self.__advance()
            print "Step: " + str(self.n) + ", Time: " + str(self.thist) + "\n"
        print "Evolution finished!"

    def __advance(self):
        """Advance the simulation one timestep. 
        """
        ridx = self.n-1
        for l in range(0, ridx):
            op_single = self.ops.single[l]
            op_double = self.ops.double[l]
            self.network.transform(l, op_single, op_double)
        op_single = self.ops.single[ridx]
        self.network.transform(l, op_single)
        
        self.thist += delta
        ++self.step

    
    def printnetwork(self):
        return str(self.network)
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

    def __theta_ij(self, V, coef_l, coef_r):
        """ Compute theta^ij_ac 
          =sum(b) sum(k,l){ V^ij_kl G^Ck_ab L_b G^Dl_bc}
        """
        atup = (V, coef_l.gamma, coef_l.lam, coef_r.gamma)
        V_idx = [-1, -2, 3, 4]
        Gc_idx = [2, -3, 3]
        L_idx = [3]
        Gd_idx = [1, 3, -4]
        idx = (V_idx, Gc_idx, L_idx, Gd_idx)
        return scon(atup, idx)
    
    def __rho_evs(self, theta, lam):
        theta_star = np.conjugate(theta) 
        lam_bra = np.conjugate(lam)
        inputs = (lam_bra, lam, theta, theta_star)
        lam_bra_idx = [1]
        lam_idx = [1]
        theta_idx = [2, -1, 1, -2]
        theta_star_idx = [2, -3, 1, -4]
        idx = (lam_bra_idx, lam_idx, theta_idx, theta_star_idx)
        return scon(inputs, idx)

    def transform(self, l, single, double=None):
        """Apply an operator to the l'th and, optionally,
           the l+1th link. The former transform is performed with the operator
           fed in through the argument 'single'. The latter is performed with
           the argument 'double'. If the latter is left 'None' only the 
           'single' transform is performed.
        """
        if not 0 <= l < self.n:
            raise IndexError("l out of bounds.")

        if l == (self.n-1):
            __single_transform(l, single)
        else:
            if double is not None:
                __double_transform(l, double) 

            __single_transform(l, single)
    
    def __single_transform(self, l, U):
        coef = self.coefs[l]
        atup = (U, coef.gamma)
        U_idx = [-1, 1]
        G_idx = [1, 2, 3]
        newgamma = scon(atup, (U_idx, G_idx))
        self.coefs[l].setgamma(newgamma) 


    def __double_transform(self, V, l):
        theta = self.__theta_ij(V, self.coefs[l], self.coefs[l+1])
        Jevs = self.__rho_evs(theta, self.coefs[l-1].lam)    

###############################################################################
#SpinChainOperators - operators representing the time-evolution of the spin
#chain.
###############################################################################
class SpinChainOperators:
    """Interface class. Method 'single' returns the operator acting on the 
       l'th link. Method 'double' returns the operator acting on the l'th
       and l+1'th links.
    """
    def single(self, l):
        if l<0 or l >= self.n :
            raise IndexError("Index " + str(l) + " out of bound " + str(self.n))
        return self.single[l]

    def double(self, l):
        if l<0 or l >= self.n :
            raise IndexError("Index " + str(l) + " out of bound " + str(self.n))
        return self.double[l]
    
class SpinChainEyes(SpinChainOperators):
    """Identity map with same data structure as SpinChainOps. For testing.
    """
    def __init__(self, n, delta, B=1, J=1):
        self.n = n
        self.single = [1]*n #the single-link operators
        self.double = [1]*n #the double-link operators
        for single, double in zip(self.single, self.double):
            single = np.eye(2)
            double = np.zeros((2, 2, 2, 2)) 
    
class SpinChainPaulis(SpinChainOperators):
    """Represent the time-evolution operators of a quantum spin chain.
    """
    def __init__(self, n, delta, B=1, J=1):
        self.n = n
        self.single = [1]*n 
        self.double = [1]*n
        psum = pauli(0) + pauli(1) + pauli(2)
        pz = pauli(2)
      
        for single, double in zip(self.single, self.double):
            U = np.zeros((2, 2, 2, 2))
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
        self.gamma = newgamma

    def setlambda(self, newlambda):
        self.lam = newlam
    
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
        return np.eye(2)
    elif (1==i):
        return np.array([[0,1], [1, 0]], dtype=np.complex)
    elif (2==i):
        return np.array([[0, -1.j], [1.j, 0]], dtype=np.complex)
    elif (3==i):
        return np.array([[1, 0], [0, -1]], dtype=np.complex)
    else:
        raise ValueError("Invalid Pauli matrix index " + str(i))

def paulitwo_left(i):
    return np.kron(pauli(i), pauli(0)) 

def paulitwo_right(i):
    return np.kron(pauli(0), pauli(i))

# def newrho_DK(Lket, theta_ij):
    # Lbra = np.conjugate(L_before)
    # theta_star = np.conjugate(theta_ij)
    # in_bracket = scon(Lbra, Lket, theta_ij, theta_star,
                # [1], [1], [2, 3, 1, 

if __name__=="__main__":
    n = 8
    chi = n/2 + 2
    chain = [1, 1]
    for i in range(0, n-2):
        chain.append(0)
    spinwave = TensorNetwork(chain, chi, 2) 
    print spinwave
