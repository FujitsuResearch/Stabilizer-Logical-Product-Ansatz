import os
os.environ["OMP_NUM_THREADS"] = "1"
print(os.cpu_count())



import argparse
psr  = argparse.ArgumentParser()
psr.add_argument('-d', '--d', required=True)
psr.add_argument('-ntrain', '--ntrain', required=True)
psr.add_argument('-ntest', '--ntest', required=True)
psr.add_argument('-nepoch', '--nepoch', required=True)
psr.add_argument('-nsamp', '--nsamp', required=True)
psr.add_argument('-b', '--batchsize', required=True)
psr.add_argument('-ns', '--ns', required=True)
psr.add_argument('-e', '--eta_adam', required=True)
psr.add_argument('-s', '--seed', required=True)
args = psr.parse_args()
depth = int(args.d)
ntrain = int(args.ntrain)
ntest = int(args.ntest)
num_epoch = int(args.nepoch)
num_samp = int(args.nsamp)
batchsize = int(args.batchsize)
num_stat = int(args.ns)
eta_adam = float(args.eta_adam)
seed = int(args.seed)
seed2 = 3140000 + seed



import numpy as np
from copy import deepcopy
import random as rand
from qulacs import QuantumState,QuantumCircuit, Observable, PauliOperator,ParametricQuantumCircuit,DensityMatrix, GeneralQuantumOperator
from qulacs.gate import H,X,Y,Z,RX,RY,RZ,CNOT,CZ,TOFFOLI,SWAP,merge,DenseMatrix,add, Probabilistic, P0, P1,to_matrix_gate,PauliRotation,Pauli, RandomUnitary
from qulacs.state import inner_product, partial_trace, tensor_product



# parameters
nqubits = 4 # number of qubits
eff_nqubits = nqubits - 2 # number of logical qubits


# observable
obsX = GeneralQuantumOperator(nqubits)
obsX.add_operator( PauliOperator("X 0 X 1", 1.0) )


# optimizer
class Adam:
    def __init__(self, eta=eta_adam, beta1=0.9, beta2=0.999):
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)


        self.iter += 1
        eta_t  = self.eta * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            params[i] -= eta_t * self.m[i] / np.sqrt(self.v[i] + 1e-8)

            
# parameter update
def my_update(self, params):
    for i in range(len(params)):
        self.set_parameter(i, params[i])
ParametricQuantumCircuit.my_update = my_update   
            
            
# target random unitary
def make_unitary():
    # Hamiltonian with eigenvalues -1.5, -0.5, +0.5, +1.5, where each eigenspace is characterized by ZZZZ=\pm 1 and XXXX=\pm 1
    # Diagonalize the Hamiltonian
    Ham = GeneralQuantumOperator(nqubits)
    Ham.add_operator( PauliOperator("Z 0 Z 1 Z 2 Z 3", 1.0) )
    Ham.add_operator( PauliOperator("X 0 X 1 X 2 X 3", 0.5) )

    Hmat_csr = Ham.get_matrix()
    Hmat = Hmat_csr.toarray()
    vals,vecs = np.linalg.eigh(Hmat)

    # Consider a Haar random unitary for each eigenspace
    u1 = RandomUnitary(range(eff_nqubits),seed=0)
    u2 = RandomUnitary(range(eff_nqubits),seed=1)
    u3 = RandomUnitary(range(eff_nqubits),seed=2)
    u4 = RandomUnitary(range(eff_nqubits),seed=3)
    u1_mat = u1.get_matrix()
    u2_mat = u2.get_matrix()
    u3_mat = u3.get_matrix()
    u4_mat = u4.get_matrix()

    umat = np.zeros((2**nqubits,2**nqubits),dtype=np.complex128)
    s = 0
    for i,a in enumerate(range(s,s+2**eff_nqubits)):
        for j,b in enumerate(range(s,s+2**eff_nqubits)):
            umat[a,b] = u1_mat[i,j]

    s += 2**eff_nqubits
    for i,a in enumerate(range(s,s+2**eff_nqubits)):
        for j,b in enumerate(range(s,s+2**eff_nqubits)):
            umat[a,b] = u2_mat[i,j]

    s += 2**eff_nqubits
    for i,a in enumerate(range(s,s+2**eff_nqubits)):
        for j,b in enumerate(range(s,s+2**eff_nqubits)):
            umat[a,b] = u3_mat[i,j]

    s += 2**eff_nqubits
    for i,a in enumerate(range(s,s+2**eff_nqubits)):
        for j,b in enumerate(range(s,s+2**eff_nqubits)):
            umat[a,b] = u4_mat[i,j]


    # Return the original basis
    umat_comp = np.dot(np.dot(vecs, umat), vecs.T.conj())
    target_gate = DenseMatrix(range(nqubits), umat_comp)

    return DenseMatrix(range(nqubits), umat_comp)

target_gate = make_unitary()



# data
# An input state is a Haar product state
def make_product_harr(nqubits,seed):
    qstate = QuantumState(1)
    qstate.set_Haar_random_state(seed)

    for i in range(1,nqubits):
        qstate2 = QuantumState(1)
        qstate2.set_Haar_random_state(seed+i)
        qstate = tensor_product(qstate, qstate2)

    return qstate

#rand.seed(0)
def make_data():
    rand.seed(0)
    training_data = []
    test_data = []

    for i in range(ntrain):
        qstate = make_product_harr(nqubits,100*i)
        vec = qstate.get_vector()

        # Apply the target unitary
        target_gate.update_quantum_state(qstate)
        exX = obsX.get_expectation_value(qstate)

        # training data
        training_data.append((vec, np.real(exX)))
    print("Training data generated!")

    for i in range(ntest):
        qstate = make_product_harr(nqubits,100000*(i+1))
        vec = qstate.get_vector()

        # Apply the target unitary
        target_gate.update_quantum_state(qstate)
        exX = obsX.get_expectation_value(qstate)

        # test data
        test_data.append((vec, np.real(exX)))

    print("test data generated!")
    return training_data,test_data

training_data,test_data = make_data()




# functions
def random_initialize(params):
    for i in range(len(params)):
        params[i] = 2 * np.pi * rand.random()

def convert(string):
    target = []
    pauli = []
    for j,s in enumerate(string):
        if s=="X":
            target.append(j)
            pauli.append(1)
        elif s=="Y":
            target.append(j)
            pauli.append(2)
        elif s=="Z":
            target.append(j)
            pauli.append(3)
            
    return target, pauli



# circuit
num_params = depth*12
params = np.zeros(num_params)
random_initialize(params)
circ = ParametricQuantumCircuit(nqubits)
generator_list = ["XXII", "IIYY",
                  "IZZI", "XIIX",
                  "YYII", "IIZZ",
                  "IXXI", "YIIY",
                  "ZZII", "IIXX",
                  "IYYI", "ZIIZ"]
for i in range(depth):
    for g in generator_list:
        target,pauli = convert(g)
        circ.add_parametric_multi_Pauli_rotation_gate(target, pauli, params[i])

        
        
# Efficient sampling
def sampling_binomial(obs,qstate,num_samp):
    ex = np.real( obs.get_expectation_value(qstate) )
    p = (ex + 1.0)/2.0
    p_sample = np.random.binomial(num_samp, p, 1)[0]/num_samp
    result = 2*p_sample - 1.0
    
    return result



# gradient
def cal_grad(data, params, num_samp):
    
    grad = np.zeros(num_params)
    gradX = np.zeros(num_params)
    qstate = QuantumState(nqubits)
    
    # Gradient of <XX>
    for i in range(num_params):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += np.pi/2
        params_minus[i] -= np.pi/2
        
        qstate.load(data[0])
        circ.my_update(params_plus)
        circ.update_quantum_state(qstate)
        r_plus = sampling_binomial(obsX,qstate,num_samp)
        
        qstate.load(data[0])
        circ.my_update(params_minus)
        circ.update_quantum_state(qstate)
        r_minus = sampling_binomial(obsX,qstate,num_samp)
        
        gradX[i] = (r_plus - r_minus)/2.0
        
    # <XX>
    qstate.load(data[0])
    circ.my_update(params)
    circ.update_quantum_state(qstate)
    r = sampling_binomial(obsX,qstate,num_samp)  
    
    return -2*(data[1] - r)*gradX


def cal_grad_minibatch(batch, data, params, num_samp):
    grad = np.zeros(num_params)
    circ.my_update(params)
    
    for i in batch:
        grad += cal_grad(data[i], params, num_samp)

    return grad/len(batch)



# cost function
def cal_cost(data_list,params):
    cost = 0.0
    circ.my_update(params)
    qstate = QuantumState(nqubits)
    
    for i in range(len(data_list)):
        qstate.load(data_list[i][0])
        circ.update_quantum_state(qstate)  
        exX = obsX.get_expectation_value(qstate)
        
        cost += (data_list[i][1] - exX)**2
            
    return np.real(cost)/len(data_list)



# optimization
def QCNN_optimize_SGD(num_epoch, batchsize, params, num_samp):    

    #
    cost_list = []
    test_list = []
    Adam_opt = Adam()
     
    # Optimization
    for i in range(num_epoch):
                   
        # cost function
        cost_list.append(cal_cost(training_data, params))
        test_list.append(cal_cost(test_data, params))

        # choose mini-batch
        if ntrain%batchsize != 0:
            print("Error")
        randbatch = rand.sample(range(ntrain),k=ntrain)
        
        p = 0
        # minibatch optimize
        while p<ntrain:
            b = randbatch[p:p+batchsize]
            grad = cal_grad_minibatch(b,training_data,params, num_samp)
            Adam_opt.update(params,grad)
            p += batchsize

    return cost_list, test_list



# main
rand.seed(seed2)
foldername = f"output_sc_ns={num_samp}/"

for _ in range(num_stat):
    random_initialize(params)
    cost_list, test_list = QCNN_optimize_SGD(num_epoch, batchsize, params, num_samp)
    np.save(foldername + f'sc_cost_seed={seed-1}.npy',cost_list)
    np.save(foldername + f'sc_test_seed={seed-1}.npy',test_list)

