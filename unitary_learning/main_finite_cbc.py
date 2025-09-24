import os
os.environ["OMP_NUM_THREADS"] = "1" 
print(os.cpu_count())



import argparse
psr  = argparse.ArgumentParser()
psr.add_argument('-ntrain', '--ntrain', required=True)
psr.add_argument('-ntest', '--ntest', required=True)
psr.add_argument('-nepoch', '--nepoch', required=True)
psr.add_argument('-nsamp', '--nsamp', required=True)
psr.add_argument('-b', '--batchsize', required=True)
psr.add_argument('-ns', '--ns', required=True)
psr.add_argument('-e', '--eta_adam', required=True)
psr.add_argument('-s', '--seed', required=True)
args = psr.parse_args()
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


""" parameters """
nqubits = 4 # number of qubits
eff_nqubits = nqubits - 2 # number of logical qubits

""" observable """
obsX = GeneralQuantumOperator(nqubits)
obsX.add_operator( PauliOperator("X 0 X 1", 1.0) )

""" optimizer """
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
            
""" efficient sampling """
def sampling_binomial(obs,qstate,num_samp):
    ex = np.real( obs.get_expectation_value(qstate) )
    p = (ex + 1.0)/2.0
    p_sample = np.random.binomial(num_samp, p, 1)[0]/num_samp
    result = 2*p_sample - 1.0
    
    return result

            

""" target random unitary """
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



""" data """
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




##############################
""" Convert a Pauli string (e.g., "X 0 Z 1 X 2 X 3 I 4 I 5 I 6 I 7 I 8") to its check matrix representation as a row vector. """
def Check_Matrix(pauli_str):
    str = ''.join([char for char in pauli_str if not char.isdigit() and not char.isspace()])
    conversion_Xdict = {
        'I': 0,
        'X': 1,
        'Y': 1,
        'Z': 0
    }
    conversion_Zdict = {
        'I': 0,
        'X': 0,
        'Y': 1,
        'Z': 1,
    }
    vector1 = []
    vector2 = []
    for char in str:
        vector1.append(conversion_Xdict[char])
        vector2.append(conversion_Zdict[char])
    return np.concatenate((vector1, vector2))


""" Lambda matrix: g=0 for [A,B]=0 and g=1 for [A,B]≠0, where g=int(Check_Matrix[A]@Lam_Matrix(n)@Check_Matrix(B).T)%2 """
def Lam_Matrix(n):
    # Create the n x n identity matrix
    I_n = np.eye(n)

    # Create the 2n x 2n block matrix
    top_row = np.hstack((np.zeros((n, n)), I_n))
    bottom_row = np.hstack((I_n, np.zeros((n, n))))

    matrix = np.vstack((top_row, bottom_row))

    return matrix




""" Stabilizers, logical operators, and quantum circuit """
num_params = 96
num_blocks = 24
num_stabs = 4

llist = [
"X 0 X 1 I 2 I 3",
"I 0 I 1 Y 2 Y 3",
"I 0 Z 1 Z 2 I 3",
"X 0 I 1 I 2 X 3",
"Y 0 Y 1 I 2 I 3",
"I 0 I 1 Z 2 Z 3",
"I 0 X 1 X 2 I 3",
"Y 0 I 1 I 2 Y 3",
"Z 0 Z 1 I 2 I 3",
"I 0 I 1 X 2 X 3",
"I 0 Y 1 Y 2 I 3",
"Z 0 I 1 I 2 Z 3",
"X 0 X 1 I 2 I 3",
"I 0 I 1 Y 2 Y 3",
"I 0 Z 1 Z 2 I 3",
"X 0 I 1 I 2 X 3",
"Y 0 Y 1 I 2 I 3",
"I 0 I 1 Z 2 Z 3",
"I 0 X 1 X 2 I 3",
"Y 0 I 1 I 2 Y 3",
"Z 0 Z 1 I 2 I 3",
"I 0 I 1 X 2 X 3",
"I 0 Y 1 Y 2 I 3",
"Z 0 I 1 I 2 Z 3"
]

SX = "X 0 X 1 X 2 X 3"
SZ = "Z 0 Z 1 Z 2 Z 3"
Obs = "X 0 X 1 I 2 I 3"


# generator
def make_blocklist():
    Block_List = []
    for pauli_str in llist:
        block = []

        ans0 = PauliOperator(pauli_str, 1.0)
        ans0.change_coef(1.)
        block.append(ans0)

        ans1 = PauliOperator(pauli_str, 1.0)*PauliOperator(SX, 1.0)
        ans1.change_coef(1.)
        block.append(ans1)

        ans2 = PauliOperator(pauli_str, 1.0)*PauliOperator(SZ, 1.0)
        ans2.change_coef(1.)
        block.append(ans2)

        ans3 = PauliOperator(pauli_str, 1.0)*PauliOperator(SX, 1.0)*PauliOperator(SZ, 1.0)
        ans3.change_coef(1.)
        block.append(ans3)

        Block_List.append(block)
    return Block_List

block_list = make_blocklist()

# Quantum circuit
circ = ParametricQuantumCircuit(4)
for block in block_list:
    for pauli in block:
        circ.add_parametric_multi_Pauli_rotation_gate(pauli.get_index_list(), pauli.get_pauli_id_list(), 0.0)

def circ_params_update(circ, params_block):
    x = 0
    for params in params_block:
        for angle in params:
            circ.set_parameter(x, angle)
            x += 1
    return

    
""" cost funtion """
def cal_cost(dataset, params_block):
    ans = 0
    circ_params_update(circ, params_block)
    for data in dataset:
        qstate = QuantumState(4)
        qstate.load(data[0])
        circ.update_quantum_state(qstate)

        exX = np.real( PauliOperator(Obs,1.0).get_expectation_value(qstate) )
        ans += (exX-data[1])**2

    return ans/len(dataset)
    


""" Quantum circuit for gradient estimation """
# A hardware-efficient implementation involves simultaneously diagonalizing Z*O_j^a. However, for simplicity, we use the Hadamard test with additional ancilla qubits to circumvent the need for simultaneous diagonalization.
def grad_circ(b,params_block): # b is the block index

    # Identity gates
    I_gate_4 = DenseMatrix([0,1,2,3], np.eye(2**4))
    I_gate_5 = DenseMatrix([0,1,2,3,4], np.eye(2**5))

    # check matrices for logical operators
    check_matrix_list = []
    for ans in llist:
        check_matrix_list.append(Check_Matrix(ans))
        
    # If the bth block commutes with the observable, flag_g=0, and if it does not, flag_g=1.
    # All generators in each block have a common g_j because the observable X1X2 and stabilizers are commutative. The (anti-)commutation relation is determined from the logical operator.
    flag_g = int(check_matrix_list[b]@Lam_Matrix(4)@Check_Matrix(Obs).T)%2

        
    """ Quantum circuit """
    gcirc = QuantumCircuit(9) # 4 qubits for the system, 1 qubit for the ancilla, and 4 qubits for Hadamard test

    ### Thr formaer part ###
    for i in range(b+1): 
        block = block_list[i] 
        params = params_block[i] 
        for (pauli,angle) in zip(block,params):
            gcirc.add_gate(PauliRotation(pauli.get_index_list(), pauli.get_pauli_id_list(),angle)) 


    ### The latter part ###
    # Hadamard gate
    gcirc.add_H_gate(4)

    # W'
    W_dash = I_gate_4

    for i in range(b+1,num_blocks):
        block = block_list[i]
        params = params_block[i]
        for (pauli,angle) in zip(block,params):
            temp =  merge(W_dash ,PauliRotation(pauli.get_index_list(), pauli.get_pauli_id_list(),angle))
            W_dash = temp

    if flag_g == 0:
        matrix = W_dash.get_matrix()
        new_matrix = -1j * matrix
        new_gate = DenseMatrix([0,1,2,3], new_matrix)
        W_mat_dash = to_matrix_gate(new_gate)

    elif flag_g == 1:
        matrix = W_dash.get_matrix()
        new_matrix = -1 * matrix
        new_gate = DenseMatrix([0,1,2,3], new_matrix)
        W_mat_dash = to_matrix_gate(new_gate)

    control_index = 4
    control_with_value = 1 
    W_mat_dash.add_control_qubit(control_index, control_with_value)
    gcirc.add_gate(W_mat_dash)

    # W^tilde
    W_til = I_gate_4

    for i in range(b+1,num_blocks):
        flag = int(check_matrix_list[b]@Lam_Matrix(4)@check_matrix_list[i].T)%2 #flag=0:commute, flag=1:not commute

        block = block_list[i]
        params = params_block[i]

        if flag == 0:
            for (pauli,angle) in zip(block,params):
                tmp = merge(W_til ,PauliRotation(pauli.get_index_list(), pauli.get_pauli_id_list(),angle))
                W_til = tmp

        elif flag == 1:
            for (pauli,angle) in zip(block,params):
                tmp = merge(W_til ,PauliRotation(pauli.get_index_list(), pauli.get_pauli_id_list(),-angle))
                W_til = tmp

    W_mat_til = to_matrix_gate(W_til)
    control_index = 4
    control_with_value = 0 
    W_mat_til.add_control_qubit(control_index, control_with_value)
    gcirc.add_gate(W_mat_til)

    # Hadamard gate
    gcirc.add_H_gate(4)

    
    
    """ Hadamard Test """
    gcirc.add_H_gate(5)
    gcirc.add_H_gate(6)
    gcirc.add_H_gate(7)
    gcirc.add_H_gate(8)

    block = block_list[b]

    for k in range(4):
        ans = block[k]*PauliOperator(Obs,1.0) # G_j^a * Obs
        O = PauliOperator("I 0 I 1 I 2 I 3 Z 4", 1.0)*ans
        coef = O.get_coef()

        gate = merge(I_gate_5,Pauli(O.get_index_list(), O.get_pauli_id_list()) )
        O_mat = gate.get_matrix()

        if flag_g == 0:
            mat = O_mat * coef
        elif flag_g == 1:
            mat = O_mat * coef * 1j

        new_mat = DenseMatrix([0,1,2,3,4], mat)
        cc = to_matrix_gate(new_mat)
        cc.add_control_qubit(k+5,1)
        gcirc.add_gate(cc)

    gcirc.add_H_gate(5)
    gcirc.add_H_gate(6)
    gcirc.add_H_gate(7)
    gcirc.add_H_gate(8)

    return gcirc


# gradient measurement
def cal_grad(data, params_block, n_shot):
    grad = np.zeros_like(params_block)

    
    for b in range(num_blocks):        
        # <XX>
        state0 = QuantumState(4)
        state0.load(data[0])
        circ.update_quantum_state(state0)
        circ_params_update(circ, params_block)
        exX = sampling_binomial(obsX, state0, n_shot) 
        
        # Gradient of <XX>
        state1 = QuantumState(4)
        state1.load(data[0])
        state2 = QuantumState(5)
        state = tensor_product(state2, state1)

        gcirc = grad_circ(b, params_block)
        gcirc.update_quantum_state(state)
        samples = state.sampling(n_shot)
        bit_strings = [format(sample, f'0{9}b') for sample in samples]

        rate_0 = 0
        rate_1 = 0
        rate_2 = 0
        rate_3 = 0
        for string in bit_strings:
            rate_0 += (-2*int(string[3])+1)/(n_shot)
            rate_1 += (-2*int(string[2])+1)/(n_shot)
            rate_2 += (-2*int(string[1])+1)/(n_shot)
            rate_3 += (-2*int(string[0])+1)/(n_shot)

        grad[b][0] = 2*(exX-data[1])*rate_0
        grad[b][1] = 2*(exX-data[1])*rate_1
        grad[b][2] = 2*(exX-data[1])*rate_2
        grad[b][3] = 2*(exX-data[1])*rate_3

    return grad



""" Main """
def QCNN_optimize_SGD(num_epoch, batchsize, params_block, num_samp):

    #変数
    cost_list = []
    test_list = []
    Adam_opt = Adam()
    for _ in range(num_epoch):
        cost_list.append(cal_cost(training_data,params_block))
        test_list.append(cal_cost(test_data,params_block))

        randbatch = rand.sample(range(ntrain),k=ntrain)
        p = 0
        while p < ntrain:
            batch = randbatch[p:p+batchsize]
            grad = np.zeros_like(params_block)
            for b in batch:
                grad += cal_grad(training_data[b], params_block, num_samp)/batchsize
            Adam_opt.update(params_block,grad)
            p += batchsize

    return cost_list, test_list



""" Execution """
np.random.seed(seed2)
params_block = 2*np.pi*np.random.rand(len(llist), 4)

foldername = f"output_cbc_ns={num_samp}/"

cost,test = QCNN_optimize_SGD(num_epoch, batchsize, params_block, num_samp)
np.save(foldername + f'cbc_cost_seed={seed-1}.npy',cost)
np.save(foldername + f'cbc_test_seed={seed-1}.npy',test)






