"""
Summary
-------
USBR Neural Network Training Module. The goal of this FF-ANN BGD Training Model
is to determine the Weight values for the final FF-ANN.

To migrate to tensorflow v2.0, follow these instructions, accessed 2019-09-20:
https://www.tensorflow.org/beta/guide/migration_guide

To run code on Google Colaboratory with tensorflow v2.0, follow these
instructions, accessed 2019-09-20: https://www.tensorflow.org/beta/tutorials/quickstart/beginner

Notes
-----
1. Regarding os.environ(), see comment by Carmezim here, accessed 4/12/2017:
   https://github.com/tensorflow/tensorflow/issues/7778
   Basicially, the os.environ() function surpresses TensorFlow WARNINGS, which
   state that TensorFlow can run faster on CPU if built from source. Building
   from source will be a consideration if increased runtime is necessary. More
   information also available via the following link:
   https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not-compiled-to-u
2. This model is set up so that the final trained model will accept an input
   feature column vector because that is how traditional Linear Algebra is
   taught. Data sets typically have row vectors (i.e. column features) for ease
   of viewing because the number of data points (rows) greatly outnumbers the
   number of features. Be aware that input data will need to be reshaped when
   it has gone through pre- and post-processing.
3. Successfully able to run this training model with Relu activation function.
   In order to train properly on y = sin(x), the AdamOptimizers needs a small
   learning rate (0.001 according the the Tensorflow website).

"""
# %% Import libraries.
# Import standard libraries.
import os
import datetime as dt
# Import third party libraries.
import numpy as np
import pandas as pd
import tensorflow as tf


# %% Set attributes.
# See Note #1 for explanation.
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
if tf.__version__ != '1.14.0':
    msg = ('This module is tested to work only with tensorflow version 1.14.0'
           ' and will not run on any other version. Please, install tensorflow'
           ' 1.14.0 to use this module.')
    raise ImportError(msg)


# %% Define functions.
def ActFunc(Activation='linear'):
    """
    Summary
    -------
    This function is currently not being used but will be used in future
    development.

    """
    ActivationFunctions = {'TanH': tf.tanh,
                           'ReLu': tf.nn.relu,
                           'linear': lambda x: x}
    return ActivationFunctions[Activation]


def write_summaries():
    """
    For future use.
    """
    pass


def construct_nn():
    """
    For future use.
    """
    pass


def train_nn():
    """
    For future use.
    """
    pass


# def FFANN_SGD(x_Trn, t_Trn, x_Val, t_Val, x_Tst, t_Tst, nHidWeights, vSeed=100,
              # Activation='ReLu', Prompt_Report=True, L_Rate=0.5,
              # Max_Iter=100000, SummaryFreq=10000, ReportFreq=10000):
def FFANN_SGD(x_Trn, t_Trn, x_Val, t_Val, x_Tst, t_Tst, nHidWeights, vSeed=100,
              Prompt_Report=True, L_Rate=0.5, Max_Iter=100000,
              SummaryFreq=10000, ReportFreq=10000):
    time0 = dt.datetime.now()
    # Construct input and target placeholders.
    x = tf.compat.v1.placeholder(tf.float32, name="Inputs")
    t = tf.compat.v1.placeholder(tf.float32, name="Targets")
    # Map placeholders to provided datasets.
    trn_dict = {x: x_Trn, t: t_Trn}
    val_dict = {x: x_Val, t: t_Val}
    tst_dict = {x: x_Tst, t: t_Tst}
    # Initialize variables.
    Max_Iter = Max_Iter + 1
    rows_x_Trn = x_Trn.shape[0] #number of rows for data input
    rows_t_Trn = t_Trn.shape[0] #number of rows for data targets
    if type(nHidWeights) is not list:
        nHidWeights = [nHidWeights]
    nLayers = len(nHidWeights)  #number of hidden layers
    tf.compat.v1.set_random_seed(vSeed)
    C_tanh = tf.constant(1.73205080756888, dtype=tf.float32)
    C_x = tf.constant(0.658478948462, dtype=tf.float32)
    I = {1: x} #input into hidden layer
    rI = {1: rows_x_Trn} #number of rows of input into hidden layer
    W = {} #hidden layer weights & bias
    H = {} #hidden layer output; input into the next layer
    # Construct hidden layers.
    time1 = dt.datetime.now()
    for i, nHW in enumerate(nHidWeights, start=1):
        w_id = 'W{}'.format(i)
        StDev_HW = ((nHW+1)**(-0.5)) / 3
        W_init = tf.random.normal((nHW, rI[i]), 0, StDev_HW, dtype=tf.float32)
        B_init = tf.random.normal((nHW, 1    ), 0, 0.00033 , dtype=tf.float32)
        W[i] = {'W': tf.Variable(W_init, name='W{}'.format(i)),
                'B': tf.Variable(B_init, name='B{}'.format(i))}
        Step1 = tf.matmul(W[i]['W'], I[i])
        Step2 = tf.add(Step1, W[i]['B'])
        #Step3 = tf.multiply(Step2, C_x)
        #Step4 = tf.tanh(Step3)
        H[i] = tf.nn.relu(Step2, name='H{}'.format(i)) #tf.multiply(Step4,C_tanh)
        I[i + 1] = H[i]
        rI[i + 1] = nHW
        # print(I[i].name[:-2])
        # print(W[i]['W'].name[:-2])
    # Construct Output Layer.
    StDev_OW = ((rows_t_Trn+1)**(-0.5)) / 3
    OW_init = tf.random.normal((rows_t_Trn, rI[nLayers + 1]), 0, StDev_OW, dtype=tf.float32)
    OB_init = tf.random.normal((rows_t_Trn, 1              ), 0, 0.00033 , dtype=tf.float32)
    O = {'W': tf.Variable(OW_init, name='WO'),
         'B': tf.Variable(OB_init, name='BO')}
    Step1 = tf.matmul(O['W'], I[nLayers + 1])
    Step2 = tf.add(Step1, O['B'], name='Output')
    #Step3 = tf.multiply(Step2, C_x)
    #Step4 = tf.tanh(Step3)
    y = Step2 #y = tf.multiply(Step4, C_tanh)
    O['y'] = y
    # Construct Error and Accuracy Equations.
    E = tf.reduce_mean(tf.multiply(tf.square(tf.subtract(y, t)), 0.5))
    t_ave = tf.reduce_mean(t)
    Et = tf.reduce_sum(tf.square(tf.subtract(t, t_ave)), name="Model_Variance")
    Es = tf.reduce_sum(tf.square(tf.subtract(y, t)), name="Prediction_Variance")
    Acc = 1 - (Es / Et)
    # Construct optimizer.
    # optimizeE = tf.compat.v1.train.GradientDescentOptimizer(L_Rate).minimize(E)
    optimizeE = tf.compat.v1.train.AdamOptimizer(L_Rate).minimize(E)
    # Initialize training sessions.
    init = tf.compat.v1.global_variables_initializer()
    sess = tf.compat.v1.Session()
    sess.run(init)
    # TODO: Move this to board writing.
    # <JAS 2019-09-20>
    Model_Tag = 'FFANN_SGD'
    Run_Stamp = time0.strftime('%Y-%m-%d-%H%M%S_'+Model_Tag+'_i'+str(int(Max_Iter/1000))+'k')
    tb_dir = os.path.abspath(r'../data/tensorboards')
    run_dir = os.path.join(tb_dir, Run_Stamp)
    trn_dir = os.path.join(run_dir, '0_Training')
    val_dir = os.path.join(run_dir, '1_Validation')
    tst_dir = os.path.join(run_dir, '2_Testing')
    # Write histograms of Hidden Weights, Biases, and Returns.
    for i, w in W.items():
        HL_Name = 'Hidden_Layer_{}'.format(i)
        with tf.name_scope(HL_Name):
            tf.compat.v1.summary.histogram(w['W'].name[:-2], w['W'])
            tf.compat.v1.summary.histogram(w['B'].name[:-2], w['B'])
            tf.compat.v1.summary.histogram(H[i].name[:-2],   H[i])
    # Write histogram of Output Weights, Biases, and Returns.
    with tf.name_scope("Output_Layer"):
        tf.compat.v1.summary.histogram(O['W'].name[:-2], O['W'])
        tf.compat.v1.summary.histogram(O['W'].name[:-2], O['B'])
        tf.compat.v1.summary.histogram(O['y'].name[:-2], O['y'])
    # Construct Error & Accuracy Equations.
    with tf.name_scope("Error_Equation"):
        tf.compat.v1.summary.scalar('Error', E)
    with tf.name_scope("Accuracy_Equation"):
        tf.compat.v1.summary.scalar('Accuracy', Acc)
    # TODO: Move this to board writing.
    # <JAS 2019-09-20>
    # Merge summaries for reporting.
    merged_summary = tf.compat.v1.summary.merge_all()
    # Write summaries to respective directories.
    TrnWriter = tf.compat.v1.summary.FileWriter(trn_dir)
    ValWriter = tf.compat.v1.summary.FileWriter(val_dir)
    TstWriter = tf.compat.v1.summary.FileWriter(tst_dir)
    # Add graphs to TensorBoard session.
    TrnWriter.add_graph(sess.graph)
    ValWriter.add_graph(sess.graph)
    TstWriter.add_graph(sess.graph)
    # ????: Return session object and placeholder dictionaries at this point?
    # <JAS 2019-09-20>
    time2 = dt.datetime.now()
    # Perform training iterations.
    product_header = [['Error', 'Accuracy'],
                      ['Training', 'Validation', 'Testing']]
    lvl_names = ['Indicator', 'Dataset']
    report_header = pd.MultiIndex.from_product(product_header, names=lvl_names)
    df_report = pd.DataFrame(columns=report_header)
    df_report.index.name = 'Iteration'
    for i in range(Max_Iter):
        writ_sum = True if (i % SummaryFreq == 0) else False
        rprt_val = True if (i % ReportFreq == 0) else False
        if writ_sum:
            S_Trn = sess.run(merged_summary, feed_dict=trn_dict)
            S_Val = sess.run(merged_summary, feed_dict=val_dict)
            S_Tst = sess.run(merged_summary, feed_dict=tst_dict)
            TrnWriter.add_summary(S_Trn, i)
            ValWriter.add_summary(S_Val, i)
            TstWriter.add_summary(S_Tst, i)
        if rprt_val:
            E_Trn, A_Trn = sess.run([E, Acc], feed_dict=trn_dict)
            E_Val, A_Val = sess.run([E, Acc], feed_dict=val_dict)
            E_Tst, A_Tst = sess.run([E, Acc], feed_dict=tst_dict)
            record = [E_Trn, E_Val, E_Tst, A_Trn, A_Val, A_Tst]
            df_record = pd.DataFrame([record], columns=report_header, index=[i])
            df_report = pd.concat([df_report, df_record])
            if Prompt_Report == True:
                if i == df_report.index[0]:
                    print(df_report.loc[[i], :].to_string(col_space=12))
                else:
                    print(df_report.loc[[i], :].to_string(col_space=12,
                                                          header=False))
        sess.run(optimizeE, trn_dict)
    time3 = dt.datetime.now()
    # return outputs
    Time_Initialize = time1 - time0
    Time_Construct = time2 - time1
    Time_Training = time3 - time2
    if Prompt_Report:
        msg_rpt = ('\n'
                   'Training Run Time = {}'
                   '\n'
                   'Type the following to activate TensorBoard: tensorboard'
                   ' --logdir {}')
        print(msg_rpt.format(Time_Training, run_dir))
    WeightDict = {}
    for i in range(nLayers):
        WeightDict[i + 1]= {'W': sess.run(W[i + 1]['W'], trn_dict),
                            'B': sess.run(W[i + 1]['B'], trn_dict)}
    WeightDict['O']= {'W': sess.run(O['W'], trn_dict),
                      'B': sess.run(O['B'], trn_dict)}
    y_Trn = sess.run(y, trn_dict)
    y_Val = sess.run(y, val_dict)
    y_Tst = sess.run(y, tst_dict)
    # Construct output dictionary.
    out_dict = {'y_Trn': y_Trn,
                'y_Val': y_Val,
                'y_Tst': y_Tst,
                'ReportSummary': df_report,
                'vSeed': vSeed,
                'L_Rate': L_Rate,
                'WeightDict': WeightDict,
                'Time_Initialize': Time_Initialize,
                'Time_Construct': Time_Construct,
                'Time_Training': Time_Training,
                'Model_Tag': Model_Tag,
                'Run_Stamp': Run_Stamp}
    # Return dictionary of values.
    return out_dict


# %% Execute code.
if __name__ == '__main__':
    msg = ('This module is intended to be imported for use into another'
           ' module. It is not intended to be run as a __main__ file.')
    raise RuntimeError(msg)
