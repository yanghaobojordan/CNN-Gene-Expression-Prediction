################################################################################
#                       Section 1: Loading the data                            #
################################################################################

import numpy as np
from matplotlib import pyplot as plt
import sklearn.model_selection
import warnings
import seaborn as sns
import random
import gc
import scipy.stats as stats
from matplotlib.patches import Rectangle
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf

def get_data():
  # Keys to npzfile of train & eval
  train_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E105', 'E011', 'E106', 'E082', 'E097', 'E116', 'E098', 'E058',
  'E117', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E127', 'E047', 'E094', 'E007', 'E054', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

  eval_cells = ['E065', 'E004', 'E066', 'E005', 'E012', 'E027', 'E053', 'E013', 'E028', 'E061', 'E109', 'E120', 'E062', 'E037', 'E038', 'E024', 'E071', 'E105', 'E087', 'E011', 'E106', 'E096', 'E082', 'E097',
  'E116', 'E098', 'E058', 'E117', 'E084', 'E059', 'E070', 'E118', 'E085', 'E104', 'E119', 'E006', 'E112', 'E127', 'E047', 'E094', 'E007', 'E054', 'E113', 'E128', 'E095', 'E055', 'E114', 'E100', 'E056', 'E016', 'E122', 'E057', 'E123', 'E079', 'E003', 'E050']

  # Load data
  train_data = np.load('train.npz')
  eval_data = np.load('eval.npz')

  # Combine Train Data to use information from all cells
  train_inputs = [] # Input histone mark data
  train_dna_inputs = []
  train_genes = []
  train_cells_list = []
  train_outputs = [] # Correct expression value
  for cell in train_cells:
      cell_data = train_data[cell]
      hm_data = cell_data[:,:,0:6]
      for x in hm_data:
        train_inputs.append([x[:,1:6]])
        train_genes.append(int(x[0,0]))
        train_cells_list.append(cell)
      exp_values = cell_data[:,0,6]
      train_outputs.append(exp_values)

  train_inputs = np.concatenate(train_inputs, axis=0)
  train_outputs = np.concatenate(train_outputs, axis=0)

  # Prepare Eval inputs in similar way
  eval_inputs = []
  eval_names = []
  for cell in eval_cells:
      cell_data = eval_data[cell]
      hm_data = cell_data[:,:,0:6]
      for x in hm_data:
        eval_inputs.append([x[:,1:6]])
      for gene in cell_data:
        eval_names.append(cell+ "_" + str(int(gene[0,0])))

  eval_inputs = np.concatenate(eval_inputs, axis=0)

  return train_inputs, train_outputs, eval_inputs, eval_names, np.asanyarray(train_genes), np.asarray(train_cells_list)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# data is (num_cell_types * num_genes, 100, 5) with 100 bins for each gene and 5 measurements for each bin
train_x, train_y, test_x, test_names, train_genes, train_cells = get_data()

#plot distribution of true gene expression values
sns.set(style="darkgrid")
x = train_y[:]
fig = plt.figure(num=None, figsize=(9, 7),facecolor='w', edgecolor='k')
ax=fig.add_subplot(111)
plt.hist(x, density=True, bins=300,color='blue',linewidth=1)
plt.ylabel('PDF',size=25)
plt.xlabel('Expression Levels',size=25)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
ax.xaxis.grid(False)
fig.savefig('hist.png',dpi=500)
plt.show()

print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(train_genes.shape)
print(train_cells.shape)

################################################################################
#                       Section 2: Defining the model                          #
################################################################################

def make_prediction(model, input_data):
    """
    param model: a trained model
    param input_data: model inputs
    return: the model's predictions for the provided input data
    """
    return model.predict(input_data, batch_size=64)

def train_model(train_x, train_y, train_genes, train_cells, test_x, test_names, testing_hyperparamaters=False):

    #get a set of all the genes and cells and put them in a random order
    all_genes = list(set(train_genes))
    all_cells = list(set(train_cells))
    random.shuffle(all_genes)
    random.shuffle(all_cells)

    num_steps = 10 if testing_hyperparamaters else 1
    for validation_step in range(num_steps):
        # Helps avoid memory problems :)
        gc.collect()

        # If we are testing hyperparamaters and want to run cross-validation, 5 cells and
        # 1600 genes are witheld for validation for this round of CV
        if testing_hyperparamaters:
          print("Validation step: ", validation_step + 1)
          validation_genes = set(all_genes[validation_step*1600:validation_step*1600+1600])
          validation_cells = set(all_cells[validation_step*5:validation_step*5+5])
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
        conv2 = tf.keras.layers.Conv1D(32, 3, activation='relu', padding='same', dilation_rate=4)
        flatten = tf.keras.layers.Flatten()
        drop = tf.keras.layers.Dropout(0.5)
        d1 = tf.keras.layers.Dense(64, activation='relu')
        d2 = tf.keras.layers.Dense(1)

        # define how those layers are stacked
        intermediate = drop_in(inputs)
        intermediate = conv1(intermediate)
        intermediate = conv2(intermediate)
        intermediate = flatten(intermediate)
        intermediate = drop(intermediate)
        intermediate = d1(intermediate)
        outputs = d2(intermediate)

        # Create the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="hist2expression")
        # Uncomment to print the models specifications and flowchart
        #model.summary()
        #tf.keras.utils.plot_model(model, "model_architechture.png", show_shapes=True)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss=tf.keras.losses.MeanSquaredError()
        )

        # Train the model with the validation data if tuning hyperparameters is true
        if testing_hyperparamaters:
          train_history = model.fit(
              train_x[train_idx],
              train_y[train_idx],
              batch_size=128,
              epochs=15,
              validation_data=(train_x[validation_idx], train_y[validation_idx])
          )

          # Plot loss curves from training
          xs = np.arange(len(train_history.history['loss']))
          fig, ax = plt.subplots()
          ax.plot(xs, train_history.history['loss'])
          ax.plot(xs, train_history.history['val_loss'])
          ax.set(xlabel='Epoch', ylabel='Loss', title='Training and Validation Loss')
          plt.legend(['Train Loss', 'Validation Loss'], loc='upper right')
          plt.show()

        # otherwise if testing_hyperparamaters is false we want to train with all data
        else:
          _ = model.fit(
              train_x,
              train_y,
              batch_size=128,
              epochs=15
          )

    #return the trained model
    return(model)

################################################################################
#                       Section 3: Running the model                           #
################################################################################

# Uncomment to run cross validation, used for testing changes to the model
#model = train_model(train_x,train_y,train_genes, train_cells, test_x, test_names, testing_hyperparamaters = True)

# Train avg_size different models and record each of their predictions
# Used for generating predictions to submit to kaggle
all_predictions = []
avg_size = 20
for curr_iter in range(avg_size):
  print("Current initialization number:", curr_iter+1)
  model = train_model(train_x,train_y,train_genes, train_cells, test_x, test_names)
  test_predictions = make_prediction(model, test_x)
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
# train_predictions = make_prediction(model, train_x)
# print(train_predictions.shape)
# x = train_predictions[:,0]
# fig = plt.figure(num=None, figsize=(9, 7),facecolor='w', edgecolor='k')
# ax=fig.add_subplot(111)
# plt.hist(x, density=True, bins=300,color='blue',linewidth=1)  # density=False would make counts
# plt.ylabel('PDF',size=25)
# plt.xlabel('Expression Levels',size=25)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.xaxis.grid(False)
# fit_alpha, fit_loc, fit_beta=stats.gamma.fit(x)
# print(fit_alpha, fit_loc, fit_beta)
# fig.savefig('hist.png',dpi=500)
# plt.show()

# Code to plot the distribution of predicted expression levels for the test data, uncomment to make the plots from our writeup
# predictions = make_prediction(model, test_x)
# print(predictions.shape)
# print(len(test_names))
# x = predictions[:,0]
# fig = plt.figure(num=None, figsize=(9, 7),facecolor='w', edgecolor='k')
# ax=fig.add_subplot(111)
# plt.hist(x, density=True, bins=300,color='blue',linewidth=1)  # density=False would make counts
# plt.ylabel('PDF',size=25)
# plt.xlabel('Expression Levels',size=25)
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# ax.xaxis.grid(False)
# fit_alpha, fit_loc, fit_beta=stats.gamma.fit(x)
# print(fit_alpha, fit_loc, fit_beta)
# fig.savefig('hist.png',dpi=500)
# plt.show()
