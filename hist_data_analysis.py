import numpy as np
from matplotlib import pyplot as plt
import sklearn.model_selection
import warnings
import seaborn as sns
import random
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def one_hot_encoding(seq) -> np.ndarray:
    """
    :param seq_array: np array of DNA sequences
    :return: np array of one-hot encodings of input DNA sequences
    """
    nuc2id = {'A' : 0, 'C' : 1, 'T' : 2, 'G' : 3}
    onehot_array = np.zeros((10000, 4))
    for seq_idx, nucleotide in enumerate(seq):
      if nucleotide != 'N':
        nuc_idx = nuc2id[nucleotide]
        onehot_array[seq_idx, nuc_idx] = 1

    return onehot_array


# Keys to npzfile of train & eval
train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058',
'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097',
'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

eval_only_cells = set(eval_cells)
eval_only_cells.difference_update(set(train_cells))
eval_only_cells = list(eval_only_cells)

# Load data
train_data = np.load('train.npz')
eval_data = np.load('eval.npz')

gene_sequences = {}
id_to_idx = {}
idx = 0

with open("seq_data.csv",'r+') as text:
  for line in text.readlines():
    (id, _) = line.strip().split(',')
    #gene_sequences[idx] = value
    id_to_idx[int(id)] = idx
    idx += 1

# Combine Train Data to use information from all cells
train_inputs = [] # Input histone mark data
train_dna_inputs = []
train_genes = []
train_cells_list = []
train_genes_set = set()
train_outputs = [] # Correct expression value
for cell in train_cells:
    cell_data = train_data[cell]
    hm_data = cell_data[:,:,0:6]
    for x in hm_data:
      train_genes_set.add(int(x[0,0]))
      train_inputs.append([x[:,1:6]])
      train_genes.append(id_to_idx[int(x[0,0])])
      train_cells_list.append(cell)
    exp_values = cell_data[:,0,6]
    #train_inputs.append(hm_data)
    train_outputs.append(exp_values)

train_inputs = np.concatenate(train_inputs, axis=0)
#train_inputs = np.concatenate(train_dna_inputs, axis=0)
train_outputs = np.concatenate(train_outputs, axis=0)

# Prepare Eval inputs in similar way
eval_inputs = []
eval_names = []
eval_only_genes = set()
eval_genes_list = []
eval_cells_list = []
for cell in eval_cells:
    cell_data = eval_data[cell]
    hm_data = cell_data[:,:,0:6]
    for x in hm_data:
      eval_inputs.append([x[:,1:6]])
    for gene in cell_data:
      eval_only_genes.add(int(gene[0,0]))
      eval_genes_list.append(int(gene[0,0]))
      eval_cells_list.append(cell)
      eval_names.append(cell+ "_" + str(int(gene[0,0])))
eval_inputs = np.concatenate(eval_inputs, axis=0)

eval_only_genes.difference_update(train_genes_set)
print(len(eval_only_genes))
print(eval_only_cells)

train_cell_vectors = np.zeros((50,8000000))
for c in range(50):
  curr_cell_data = []
  for i in range(len(train_cells_list)):
    if train_cells_list[i] == train_cells[c]:
        curr_cell_data.append(train_inputs[i].flatten())
  train_cell_vectors[c] = np.concatenate(curr_cell_data)

print(len(train_genes_set))

cell_vectors = np.zeros((6,8000000))
for c in range(6):
  curr_cell_data = []
  for i in range(len(eval_cells_list)):
    if eval_cells_list[i] == eval_only_cells[c]:
        if eval_genes_list[i] in train_genes_set:
            curr_cell_data.append(eval_inputs[i].flatten())
  cell_vectors[c] = np.concatenate(curr_cell_data)

print("Data done!")

cell_vectors = np.concatenate([cell_vectors, train_cell_vectors])

cell_to_idx = {}
idx = 0
for cell in train_cells:
  cell_to_idx[cell] = idx
  idx+=1
print(idx)
print(len(id_to_idx))

gene_vectors = np.zeros((17447,50*100*5))
for i in range(len(train_cells_list)):
  c_type = cell_to_idx[train_cells_list[i]]
  g_type = train_genes[i]
  gene_vectors[g_type,c_type*500:c_type*500+500] = train_inputs[i].flatten()

for i in range(len(eval_cells_list)):
  if eval_cells_list[i] in cell_to_idx:
    c_type = cell_to_idx[eval_cells_list[i]]
    g_type = id_to_idx[eval_genes_list[i]]
    gene_vectors[g_type,c_type*500:c_type*500+500] = eval_inputs[i].flatten()

print(gene_vectors)


from sklearn.manifold import MDS

mds = MDS(n_components=2, max_iter=30000, n_init=10, eps=1e-10)
pos = mds.fit_transform(gene_vectors[:500])
x = pos[:,0]
y = pos[:,1]
x = np.asarray(x)
y = np.asarray(y)
plt.scatter(x,y)
plt.savefig("genes.png")
plt.clf()


mds = MDS(n_components=2, max_iter=30000, n_init=10, eps=1e-10)
pos = mds.fit_transform(cell_vectors)
x = pos[:,0]
y = pos[:,1]
x = np.asarray(x)
y = np.asarray(y)
plt.scatter(x[:50],y[:50])
plt.scatter(x[50:],y[50:])
plt.savefig("cells.png")
