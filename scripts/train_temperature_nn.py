"""
Summary
-------
This trains a neural network weights given monthly inflow, storage, and outflow
as input and average temperature as target data.

"""
# %% Import libraries.
# Import standard libraries.
import os
import sys
import datetime as dt
import operator
# Import third party libraries.
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
# Import custom modules.
import custom_modules
import FFANN as ann
import calsim_toolkit as cs


# %% Define functions.
def plot_results(Output_Dict, t_Trn, t_Val, t_Tst, i_Trn, i_Val, i_Tst, T_max, T_min):
    # Set baseline line.
    BaseX = np.array([T_min, T_max])
    BaseY = np.array([T_min, T_max])
    # Post-process results.
    t_Trn_v = (t_Trn[0] - (-0.99)) / (0.99 - (-0.99)) * (T_max - T_min) + T_min
    t_Val_v = (t_Val[0] - (-0.99)) / (0.99 - (-0.99)) * (T_max - T_min) + T_min
    t_Tst_v = (t_Tst[0] - (-0.99)) / (0.99 - (-0.99)) * (T_max - T_min) + T_min
    y_Trn_v = (Output_Dict['y_Trn'][0] - (-0.99)) / (0.99 - (-0.99)) * (T_max - T_min) + T_min
    y_Val_v = (Output_Dict['y_Val'][0] - (-0.99)) / (0.99 - (-0.99)) * (T_max - T_min) + T_min
    y_Tst_v = (Output_Dict['y_Tst'][0] - (-0.99)) / (0.99 - (-0.99)) * (T_max - T_min) + T_min
    # Create figure.
    fig, ax = plt.subplots(1,2, num='Results')
    # Update scatter plot.
    ax[0].set_title('Model Prediction vs. Target Data')
    ax[0].axhline(0, color='lightgrey')
    ax[0].axvline(0, color='lightgrey')
    ax[0].plot(BaseX, BaseY, 'k--', label='Baseline')
    ax[0].plot(t_Trn_v, y_Trn_v, 'rs', label='Training')
    ax[0].plot(t_Val_v, y_Val_v, 'bo', label='Validation')
    ax[0].plot(t_Tst_v, y_Tst_v, 'gD', label='Testing')
    ax[0].legend()
    ax[0].set_xlim(0.95 * T_min, 1.05 * T_max)
    ax[0].set_ylim(0.95 * T_min, 1.05 * T_max)
    # Update function plot.
    i = list(i_Trn[0]) + list(i_Val[0]) + list(i_Tst[0])
    t = list(t_Trn_v) + list(t_Val_v) + list(t_Tst_v)
    y = list(y_Trn_v) + list(y_Val_v) + list(y_Tst_v)
    data = list(zip(i, t, y))
    sorted_data = sorted(data, key=operator.itemgetter(0))
    arr_data = np.array(sorted_data).transpose()
    ax[1].set_title('Training Data and ANN Model Outputs vs. Inputs')
    ax[1].axhline(0, color='lightgrey')
    ax[1].axvline(0, color='lightgrey')
    ax[1].plot(arr_data[0], arr_data[1], 'r', label='Target Data')
    ax[1].plot(arr_data[0], arr_data[2], 'bo', label='Output Results')
    ax[1].legend()
    ax[1].set_ylim(0.95 * T_min, 1.05 * T_max)
    # Show Plot.
    plt.show()
    # Return success indicator.
    return 0


def main():
    t_0 = dt.datetime.now()
    # Identify all HEC5Q runs.
    HEC5Q_dir = '../__models/HEC5Q/HEC5Q'
    sub_dirs = os.listdir(HEC5Q_dir)
    models = list()
    for sub_dir in sub_dirs:
        if 'CS3' in sub_dir:
            models.append(sub_dir)
    flow_fps = list()
    temp_fps = list()
    for model in models:
        flow_fp = os.path.join(HEC5Q_dir, model, 'SR_HEC5Q_Q0', 'Model',
                               'CALSIMII_HEC5Q.DSS')
        flow_fps.append(flow_fp)
        temp_fp = os.path.join(HEC5Q_dir, model, 'SR_HEC5Q_Q0', 'Model',
                               'SR_WQ_Report.dss')
        temp_fps.append(temp_fp)
    # Acquire daily Storage, Inflow, Outflow, and Temperature data from HEC5Q.
    df_S = cs.read_dss(flow_fps, b='S_SHSTA', e='1DAY', studies=models)
    df_I = cs.read_dss(flow_fps, b='I_SHSTA', e='1DAY', studies=models).cs.wide()
    df_O = cs.read_dss(flow_fps, b='C_KSWCK', e='1DAY', studies=models).cs.wide()
    df_T = cs.read_dss(temp_fps, b='SACR_BLW_CLR_CR', c='TEMP_F',
                       studies=models).cs.wide()
    df_S.fillna(0, inplace=True)
    df_I.fillna(0, inplace=True)
    df_O.fillna(0, inplace=True)
    df_T.fillna(0, inplace=True)
    # Convert df_S from acre-feet to TAF.
    df_S['Value'] = df_S['Value'] / 1000
    df_S['Units'] = 'TAF'
    df_S = df_S.cs.wide()
    # Aggregate data to monthly timestep.
    df_S = df_S.resample('M').last()
    df_I = df_I.resample('M').mean()
    df_O = df_O.resample('M').mean()
    df_T = df_T.resample('M').mean()
    # Process data in preparation for weight training.
    S_min = df_S.min().min()
    S_max = df_S.max().max()
    df_S_n = (df_S - S_min) / (S_max - S_min) * (0.99 - (-0.99)) + (-0.99)
    I_min = df_I.min().min()
    I_max = df_I.max().max()
    df_I_n = (df_I - I_min) / (I_max - I_min) * (0.99 - (-0.99)) + (-0.99)
    O_min = df_O.min().min()
    O_max = df_O.max().max()
    df_O_n = (df_O - O_min) / (O_max - O_min) * (0.99 - (-0.99)) + (-0.99)
    T_min = df_T.min().min()
    T_max = df_T.max().max()
    df_T_n = (df_T - T_min) / (T_max - T_min) * (0.99 - (-0.99)) + (-0.99)
    records = list()
    i = 0
    idx = df_T_n.index[4:]
    for model in models:
        for row in idx:
            S_col = df_S_n[[model]].columns[0]
            I_col = df_I_n[[model]].columns[0]
            O_col = df_O_n[[model]].columns[0]
            T_col = df_T_n[[model]].columns[0]
            record = [i]
            i += 1
            record += df_S_n.loc[:row, S_col].to_list()[:-5:-1]
            record += df_I_n.loc[:row, I_col].to_list()[:-5:-1]
            record += df_O_n.loc[:row, O_col].to_list()[:-5:-1]
            record += df_T_n.loc[:row, T_col].to_list()[-1:]
            records.append(record)
    arr_records = np.array(records)
    # Print read time.
    t_1 = dt.datetime.now()
    print('Read Time: {}'.format(t_1 - t_0))
    t_0 = dt.datetime.now()
    # Randomly split data into Training, Validation, and Testing sets.
    np.random.shuffle(arr_records)
    len_trn = int(0.6 * arr_records.shape[0])
    len_val = int(0.2 * arr_records.shape[0])
    x_Trn = arr_records[:len_trn, 1:-1].transpose()
    t_Trn = arr_records[:len_trn, -1:].transpose()
    i_Trn = arr_records[:len_trn, :1].transpose()
    x_Val = arr_records[len_trn:len_trn + len_val, 1:-1].transpose()
    t_Val = arr_records[len_trn:len_trn + len_val, -1:].transpose()
    i_Val = arr_records[len_trn:len_trn + len_val, :1].transpose()
    x_Tst = arr_records[len_trn + len_val:, 1:-1].transpose()
    t_Tst = arr_records[len_trn + len_val:, -1:].transpose()
    i_Tst = arr_records[len_trn + len_val:, :1].transpose()
    t_1 = dt.datetime.now()
    # Define weight structure.
    nHidWeights = [512, 256, 128, 64]
    # nHidWeights = [64]
    print('Preprocessing Time for {}: {}'.format(nHidWeights, t_1 - t_0))
    # Train weights.
    Output_Dict = ann.FFANN_SGD(x_Trn, t_Trn, x_Val, t_Val, x_Tst, t_Tst,
                                nHidWeights, vSeed=6332, L_Rate=0.0001,
                                Max_Iter=20000, SummaryFreq=1000,
                                ReportFreq=1000)
    # Print returned dictionary.
    print('ann.FFANN_SGD Returned Dictionary of Key & Value Types:')
    for k, v in Output_Dict.items():
        print('{!s:>16}: {!s}'.format(k, type(v)))
    # Post process data to DataFrame.
    y = np.concatenate((Output_Dict['y_Trn'],
                        Output_Dict['y_Val'],
                        Output_Dict['y_Tst']), axis=1)
    arr_data = np.concatenate((arr_records, y.transpose()), axis=1)
    sx_col = ['S_SHSTA(0)', 'S_SHSTA(-1)', 'S_SHSTA(-2)', 'S_SHSTA(-3)']
    ix_col = ['I_SHSTA(0)', 'I_SHSTA(-1)', 'I_SHSTA(-2)', 'I_SHSTA(-3)']
    ox_col = ['C_KSWCK(0)', 'C_KSWCK(-1)', 'C_KSWCK(-2)', 'C_KSWCK(-3)']
    ty_col = ['T_Target(0)', 'T_Output(0)']
    cols = ['Record'] + sx_col + ix_col + ox_col + ty_col
    df = pd.DataFrame(arr_data, columns=cols)
    df.sort_values('Record', inplace=True)
    df.set_index('Record', inplace=True)
    df[sx_col] = (df[sx_col] - (-0.99)) / (0.99 - (-0.99)) * (S_max - S_min) + S_min
    df[ix_col] = (df[ix_col] - (-0.99)) / (0.99 - (-0.99)) * (I_max - I_min) + I_min
    df[ox_col] = (df[ox_col] - (-0.99)) / (0.99 - (-0.99)) * (O_max - O_min) + O_min
    df[ty_col] = (df[ty_col] - (-0.99)) / (0.99 - (-0.99)) * (T_max - T_min) + T_min
    # Output model DataFrames to disk.
    len_idx = idx.size
    for i in range(len(models)):
        df_temp = df.loc[i * len_idx: (i + 1) * len_idx - 1, :].copy()
        df_temp['DateTime'] = idx
        df_temp.set_index('DateTime', inplace=True)
        fp = '../data/training_results/{}.xlsx'.format(models[i])
        df_temp.to_excel(fp)
    # Output weights to disk.
    for k, v in Output_Dict['WeightDict'].items():
        W = v['W']
        B = v['B']
        np.savetxt('../__trained_tnn/W{}.txt'.format(k), W)
        np.savetxt('../__trained_tnn/B{}.txt'.format(k), B)
    # Plot output.
    # if show_plot:
        # plot_results(Output_Dict, t_Trn, t_Val, t_Tst, i_Trn, i_Val, i_Tst,
                     # T_max, T_min)
    plot_results(Output_Dict, t_Trn, t_Val, t_Tst, i_Trn, i_Val, i_Tst,
                 T_max, T_min)
    # Return accuracy.
    A_trn = Output_Dict['ReportSummary'][('Accuracy', 'Training')].iloc[-1]
    A_val = Output_Dict['ReportSummary'][('Accuracy', 'Validation')].iloc[-1]
    A_tst = Output_Dict['ReportSummary'][('Accuracy', 'Testing')].iloc[-1]
    return (A_trn, A_val, A_tst)


# %% Execute script.
if __name__ == '__main__':
    _ = main()
    # # Read data.
    # arr_records, S_min, S_max, I_min, I_max, O_min, O_max, T_min, T_max, idx, models = main_read()
    # # Train data.
    # result_list = list()
    # msg = 'Weights: {}; Accuracy: {}'
    # weights = [[64], [128, 64], [256, 128, 64], [512, 256, 128, 64]]
    # for weight in weights[:-1]:
        # a = main_train(arr_records, S_min, S_max, I_min, I_max, O_min, O_max,
                       # T_min, T_max, idx, models, weight, show_plot=False)
        # result_list.append(msg.format(weight, a))
    # a = main_train(arr_records, S_min, S_max, I_min, I_max, O_min, O_max,
                   # T_min, T_max, idx, models, weights[-1])
    # result_list.append(msg.format(weights[-1], a))
    # with open('results.txt', 'w') as f:
        # f.write('\n'.join(result_list))
    # for results in result_list:
        # print(results)
