import numpy as np
from matplotlib import pyplot as plt
import sklearn.model_selection
import warnings
import seaborn as sns
import random
import gc
import scipy.stats as stats
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

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

def get_data():
  # Keys to npzfile of train & eval
  train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058',
  'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

  eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097',
  'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

  # Load data
  train_data = np.load('train.npz')
  eval_data = np.load('eval.npz')

  gene_sequences = {}
  id_to_idx = {}
  idx = 0
  with open("seq_data.csv",'r+') as text:
    for line in text.readlines():
      (id, value) = line.strip().split(',')
      gene_sequences[idx] = value
      id_to_idx[int(id)] = idx
      idx += 1

  # Combine Train Data to use information from all cells
  train_inputs = [] # Input histone mark data
  train_dna_inputs = []
  train_genes = []
  train_outputs = [] # Correct expression value
  train_cells_list = []
  for cell in train_cells:
      cell_data = train_data[cell]
      hm_data = cell_data[:,:,0:6]
      for x in hm_data:
        train_inputs.append([x[:,1:6]])
        train_genes.append(id_to_idx[int(x[0,0])])
        train_cells_list.append(cell)
      exp_values = cell_data[:,0,6]
      train_outputs.append(exp_values)

  train_inputs = np.concatenate(train_inputs, axis=0)
  train_outputs = np.concatenate(train_outputs, axis=0)

  # Prepare Eval inputs in similar way
  eval_inputs = []
  eval_names = []
  eval_genes = []
  for cell in eval_cells:
      cell_data = eval_data[cell]
      hm_data = cell_data[:,:,0:6]
      for x in hm_data:
        eval_inputs.append([x[:,1:6]])
        eval_genes.append(id_to_idx[int(x[0,0])])
      for gene in cell_data:
        eval_names.append(cell+ "_" + str(int(gene[0,0])))

  eval_inputs = np.concatenate(eval_inputs, axis=0)

  seq_lookup = np.zeros((len(id_to_idx),40000))
  for gene in range(len(id_to_idx)):
    seq_lookup[gene] = one_hot_encoding(gene_sequences[gene]).flatten()


  return train_inputs, np.asarray(train_genes), train_outputs, eval_inputs, eval_names, np.asarray(eval_genes), seq_lookup, np.asarray(train_cells_list)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# data is (num_cell_types * num_genes, 100, 5) with 100 bins for each gene and 5 measurements for each bin
train_x, train_genes, train_y, test_x, test_names, test_genes, seq_lookup, train_cells = get_data()

print(train_x.shape)
print(train_genes.shape)
print(seq_lookup.shape)
print(train_y.shape)
print("")
print(test_x.shape)
print(test_genes.shape)

def filterer(arr):
  return gaussian_filter1d(arr, 3)
train_x = np.apply_along_axis(filterer, 1, train_x)

# Code for producing the plots of histone levels in the train data for genes with high/low
# expression that was shown in our presentation
#
# high = []
# low = []
# for idx,exp in enumerate(train_y):
#   if exp > 3:
#     high.append(idx)
#   if exp < 3:
#     low.append(idx)
#
# high_avg = np.mean(train_x[high], axis=0)
# low_avg = np.mean(train_x[low], axis=0)
#
# fig, axs = plt.subplots(5, 2, figsize=(8, 6))
#
# i = 0
# for hist in np.transpose(high_avg, (1, 0)):
#   axs[i,0].plot(range(0,100), hist)
#   i += 1
#
# i = 0
# for hist in np.transpose(low_avg, (1, 0)):
#   axs[i,1].plot(range(0,100), hist)
#   i += 1
#
# axs[0,1].axes.yaxis.set_ticks([])
# axs[1,1].axes.yaxis.set_ticks([])
# axs[2,1].axes.yaxis.set_ticks([])
# axs[3,1].axes.yaxis.set_ticks([])
# axs[4,1].axes.yaxis.set_ticks([])
#
# axs[0,0].axes.yaxis.set_ticks([])
# axs[1,0].axes.yaxis.set_ticks([])
# axs[2,0].axes.yaxis.set_ticks([])
# axs[3,0].axes.yaxis.set_ticks([])
# axs[4,0].axes.yaxis.set_ticks([])
#
# axs[0,1].axes.xaxis.set_ticks([])
# axs[1,1].axes.xaxis.set_ticks([])
# axs[2,1].axes.xaxis.set_ticks([])
# axs[3,1].axes.xaxis.set_ticks([])
#
# axs[0,0].axes.xaxis.set_ticks([])
# axs[1,0].axes.xaxis.set_ticks([])
# axs[2,0].axes.xaxis.set_ticks([])
# axs[3,0].axes.xaxis.set_ticks([])
#
# axs[0,0].set_ylim([0.5,1])
# axs[1,0].set_ylim([0.4,1.1])
# axs[2,0].set_ylim([0.75,1.6])
# axs[3,0].set_ylim([0.5,4])
# axs[4,0].set_ylim([0.4,0.65])
#
# axs[0,1].set_ylim([0.5,1])
# axs[1,1].set_ylim([0.4,1.1])
# axs[2,1].set_ylim([0.75,1.6])
# axs[3,1].set_ylim([0.5,4])
# axs[4,1].set_ylim([0.4,0.65])
#
# axs[0,0].title.set_text('High Expression')
# axs[0,1].title.set_text('Low Expression')
#
# axs[0,0].set_ylabel('H3K27me3') # (down) Downregulation via heterochromatin
# axs[1,0].set_ylabel('H3K36me3') # (?) Indicates gene body, mantains expression?
# axs[2,0].set_ylabel('H3K4me1')  # (up) Enriched at enhancers, primed genes
# axs[3,0].set_ylabel('H3K4me3')  # (up) Positively regulates transcription
# axs[4,0].set_ylabel('H3K9me3')  # (down) Heterochromatin, negatively regulates transcription
#
# plt.tight_layout()
# plt.show()

################################################################################
#                       Section 2: Defining the model                          #
################################################################################

def make_prediction(model, input_data, seq_data):
    """
    param model: a trained model
    param input_data: model inputs
    return: the model's predictions for the provided input data
    """
    return model.predict({"dna":seq_data, "histones":input_data}, batch_size=256)

def filter_weight_motifs():
    """
    A helper function for generating corresponding motifs for each of the
    filters in the first layer of the model. Each block of 100 generated
    sequences can be input to a motif comparison tool like Tomtom to see
    which (if any) motifs it corresponds to.
    """
    arr = np.load("conv1_weights.npy")
    print(arr.shape)
    id2nuc = {0:'A', 1:'C', 2:'T', 3:'G'}
    for i in range(arr.shape[2]):
      filter = 10**(5*arr[:,:,i]) # treat weights as log likelihoods
      sample_seqs = ""
      for _ in range(0,100): # generating 100 samples
        for pos in filter:
          pos = pos * 1/sum(pos) # normalize so probs sum to one
          sample_seqs += id2nuc[np.random.choice(4, p=pos)] # sample from it
        sample_seqs += "\n"
      print(sample_seqs)

def train_model(train_x, train_y, train_genes, train_cells, test_x, test_names, seq_lookup, testing_hyperparamaters=False):

    #get a set of all the genes and cells and put them in a random order
    all_genes = list(set(train_genes))
    all_cells = list(set(train_cells))
    random.shuffle(all_genes)
    random.shuffle(all_cells)

    num_steps = 5 if testing_hyperparamaters else 1
    for validation_step in range(num_steps):
        # Helps avoid memory problems :)
        gc.collect()

        # If we are testing hyperparamaters and want to run cross-validation, 5 cells and
        # 1600 genes are witheld for validation for this round of CV
        if testing_hyperparamaters:
          print("Validation step: ", validation_step + 1)
          validation_genes = set(all_genes[validation_step*3200:validation_step*3200+3200])
          validation_cells = set(all_cells[validation_step*10:validation_step*10+10])
          train_idx = []
          validation_idx = []
          for i in range(len(train_genes)):
            if train_genes[i] in validation_genes:
              validation_idx.append(i)
            elif train_cells[i] in validation_cells:
              validation_idx.append(i)
            else:
              train_idx.append(i)

        # Initialize model layers
        inputs = tf.keras.Input(shape=(100,5), name="histones")
        drop_in = tf.keras.layers.Dropout(0.05)
        conv1 = tf.keras.layers.Conv1D(32, 11, activation='relu', padding='same')
        conv2 = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='same', dilation_rate=3)
        flatten = tf.keras.layers.Flatten()
        drop = tf.keras.layers.Dropout(0.5)
        d1 = tf.keras.layers.Dense(96, activation='relu')

        dna_inputs = tf.keras.Input(shape=(1), name="dna")
        dna_lookup = tf.keras.layers.Embedding(17447, 40000, embeddings_initializer=tf.keras.initializers.Constant(seq_lookup), trainable=False)
        dna_reshape = tf.keras.layers.Reshape((10000,4))
        dna_conv1 = tf.keras.layers.Conv1D(32, 13, activation='relu', padding='valid', strides=5)
        dna_conv2 = tf.keras.layers.Conv1D(32, 11, activation='relu', padding='valid',strides=3)
        dna_pool = tf.keras.layers.MaxPooling1D(pool_size=5, strides=3)
        dna_conv3 = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='valid', dilation_rate=2)
        dna_conv4 = tf.keras.layers.Conv1D(32, 5, activation='relu', padding='valid', dilation_rate=4)
        dna_flatten = tf.keras.layers.Flatten()
        dna_drop = tf.keras.layers.Dropout(0.5)
        dna_d1 = tf.keras.layers.Dense(96, activation='relu')

        concat = tf.keras.layers.Concatenate()
        d2 =  tf.keras.layers.Dense(32)
        d3 = tf.keras.layers.Dense(1)

        intermediate = drop_in(inputs)
        intermediate = conv1(intermediate)
        intermediate = conv2(intermediate)
        intermediate = flatten(intermediate)
        intermediate = drop(intermediate)
        intermediate = d1(intermediate)

        dna_inermediate = dna_lookup(dna_inputs)
        dna_inermediate = dna_reshape(dna_inermediate)
        dna_inermediate = dna_conv1(dna_inermediate)
        dna_inermediate = dna_conv2(dna_inermediate)
        dna_inermediate = dna_pool(dna_inermediate)
        dna_inermediate = dna_conv3(dna_inermediate)
        dna_inermediate = dna_conv4(dna_inermediate)
        dna_inermediate = dna_flatten(dna_inermediate)
        dna_inermediate = dna_drop(dna_inermediate)
        dna_inermediate = dna_d1(dna_inermediate)

        combined = concat([intermediate, dna_inermediate])
        combined = d2(combined)
        outputs = d3(combined)

        model = tf.keras.Model(inputs=[inputs, dna_inputs], outputs=outputs, name="hist_2_expression")
        # Uncomment to print the models specifications and flowchart
        # model.summary()
        #tf.keras.utils.plot_model(model, "model_architechture.png", show_shapes=True)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.MeanSquaredError()
        )

        # Train the model with the validation data if tuning hyperparameters is true
        if testing_hyperparamaters:
          train_history = model.fit(
              {"histones":train_x[train_idx], "dna":train_genes[train_idx]},
              train_y[train_idx],
              batch_size=256,
              epochs=15,
              validation_data=({"histones":train_x[validation_idx], "dna":train_genes[validation_idx]}, train_y[validation_idx])
          )

          # Plot loss curves from training
          xs = np.arange(len(train_history.history['loss']))
          fig, ax = plt.subplots()
          ax.plot(xs, train_history.history['loss'])
          ax.plot(xs, train_history.history['val_loss'])
          ax.set(xlabel='Epoch', ylabel='Loss', title='Training and Validation Loss')
          plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
          plt.savefig("losses.jpg")

        # otherwise if testing_hyperparamaters is false we want to train with all data
        else:
          _ = model.fit(
              {"histones":train_x, "dna":train_genes},
              train_y,
              batch_size=256,
              epochs=15
          )



    #return the trained model
    return(model)

################################################################################
#                       Section 3: Running the model                           #
################################################################################

# Uncomment to run cross validation, used for testing changes to the model
# model = train_model(train_x,train_y, train_genes, train_cells, test_x, test_names, seq_lookup, testing_hyperparamaters = True)

# Here, the weights are saved so that we can later run inerpretation on them
# with the code in filter_weight_motifs()
weights,_ = model.layers[3].get_weights()
print(weights.shape)
np.save("conv1_weights", weights)

# Train avg_size different models and record each of their predictions
# Used for generating predictions to submit to kaggle
all_predictions = []
avg_size = 3
for curr_iter in range(avg_size):
  print("Current initialization number:", curr_iter+1)
  model = train_model(train_x,train_y, train_genes, train_cells, test_x, test_names, seq_lookup)
  test_predictions = make_prediction(model, test_x, test_genes)
  all_predictions.append(test_predictions)

# write the predictions to a file
predictions = sum(all_predictions)/avg_size
print(predictions.shape)
f = open("predictions.csv", "w")
f.write("id,expression\n")
for idx in range(len(predictions)):
  f.write(test_names[idx] + "," + str(predictions[idx,0]) + "\n")
f.close()

# Code to plot the distribution of predicted expression levels for the train data, uncomment to make the plots from our writeup
sns.set(style="darkgrid")
train_predictions = make_prediction(model, train_x, train_genes)
print(train_predictions.shape)
x = train_predictions[:,0]
fig = plt.figure(num=None, figsize=(9, 7),facecolor='w', edgecolor='k')
ax=fig.add_subplot(111)
plt.hist(x, density=True, bins=300,color='blue',linewidth=1)
plt.ylabel('PDF',size=25)
plt.xlabel('Expression Levels',size=25)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.xaxis.grid(False)
fit_alpha, fit_loc, fit_beta=stats.gamma.fit(x)
print(fit_alpha, fit_loc, fit_beta)
fig.savefig('train_hist.png',dpi=500)
plt.show()

# Code to plot the distribution of predicted expression levels for the test data, uncomment to make the plots from our writeup
predictions = make_prediction(model, test_x, test_genes)
print(predictions.shape)
print(len(test_names))
x = predictions[:,0]
fig = plt.figure(num=None, figsize=(9, 7),facecolor='w', edgecolor='k')
ax=fig.add_subplot(111)
plt.hist(x, density=True, bins=300,color='blue',linewidth=1)
plt.ylabel('PDF',size=25)
plt.xlabel('Expression Levels',size=25)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.xaxis.grid(False)
fit_alpha, fit_loc, fit_beta=stats.gamma.fit(x)
print(fit_alpha, fit_loc, fit_beta)
fig.savefig('test_hist.png',dpi=500)
plt.show()

# Code for performing the class based optimization and creating the plots
# that we presented in class
#
# weights = model.get_weights()
#
# def class_loss(y_true, y_pred):
#   return y_pred
#
# model.compile(loss=class_loss)
# model.set_weights(weights)
#
# max_imps = []
# for _ in range(100):
#   inp = tf.Variable(np.random.uniform(low=0.0, high=1.0, size=(1, 100, 5)))
#   for epoch in range(1000):
#     with tf.GradientTape() as tape:
#       pred = model(inp)
#       if pred.numpy()[0,0] > 4:
#         print(pred)
#         max_imps.append(inp.numpy())
#         break
#     grads = tape.gradient(pred, inp)
#     inp = tf.Variable(inp + 0.25 * grads)
#
# max_imps = np.array(max_imps)
# max_imps = np.mean(max_imps, axis=0)
#
# min_imps = []
# for _ in range(100):
#   inp = tf.Variable(np.random.uniform(low=0.0, high=1.0, size=(1, 100, 5)))
#   for epoch in range(1000):
#     with tf.GradientTape() as tape:
#       pred = model(inp)
#       if pred.numpy()[0,0] < -4:
#         min_imps.append(inp.numpy())
#         break
#     grads = tape.gradient(pred, inp)
#     inp = tf.Variable(inp - 0.25 * grads)
#
# min_imps = np.array(min_imps)
# min_imps = np.mean(min_imps, axis=0)
#
# high_avg = max_imps[0]
# low_avg = min_imps[0]
# fig, axs = plt.subplots(5, 2, figsize=(8, 6))
#
# i = 0
# for hist in np.transpose(high_avg, (1, 0)):
#   axs[i,0].plot(range(0,100), hist)
#   i += 1
#
# i = 0
# for hist in np.transpose(low_avg, (1, 0)):
#   axs[i,1].plot(range(0,100), hist)
#   i += 1
#
# axs[0,1].axes.yaxis.set_ticks([])
# axs[1,1].axes.yaxis.set_ticks([])
# axs[2,1].axes.yaxis.set_ticks([])
# axs[3,1].axes.yaxis.set_ticks([])
# axs[4,1].axes.yaxis.set_ticks([])
#
# axs[0,0].axes.yaxis.set_ticks([])
# axs[1,0].axes.yaxis.set_ticks([])
# axs[2,0].axes.yaxis.set_ticks([])
# axs[3,0].axes.yaxis.set_ticks([])
# axs[4,0].axes.yaxis.set_ticks([])
#
# axs[0,1].axes.xaxis.set_ticks([])
# axs[1,1].axes.xaxis.set_ticks([])
# axs[2,1].axes.xaxis.set_ticks([])
# axs[3,1].axes.xaxis.set_ticks([])
#
# axs[0,0].axes.xaxis.set_ticks([])
# axs[1,0].axes.xaxis.set_ticks([])
# axs[2,0].axes.xaxis.set_ticks([])
# axs[3,0].axes.xaxis.set_ticks([])
#
# axs[0,0].set_ylim([-3,3])
# axs[1,0].set_ylim([-4,4])
# axs[2,0].set_ylim([-3,3])
# axs[3,0].set_ylim([-3,3])
# axs[4,0].set_ylim([-4,4])
#
# axs[0,1].set_ylim([-3,3])
# axs[1,1].set_ylim([-4,4])
# axs[2,1].set_ylim([-3,3])
# axs[3,1].set_ylim([-3,3])
# axs[4,1].set_ylim([-4,4])
#
# axs[0,0].title.set_text('High Expression')
# axs[0,1].title.set_text('Low Expression')
#
# axs[0,0].set_ylabel('H3K27me3') # (down) Downregulation via heterochromatin
# axs[1,0].set_ylabel('H3K36me3') # (?) Indicates gene body, mantains expression?
# axs[2,0].set_ylabel('H3K4me1')  # (up) Enriched at enhancers, primed genes
# axs[3,0].set_ylabel('H3K4me3')  # (up) Positively regulates transcription
# axs[4,0].set_ylabel('H3K9me3')  # (down) Heterochromatin, negatively regulates transcription
#
# plt.tight_layout()
# plt.show()
