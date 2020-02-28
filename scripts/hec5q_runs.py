"""
Summary
-------
Python script to generate baseline datasets for temperature regime.

"""
# %% Import standard libraries.
# Import standard libraries.
import os
import sys
import shutil
import multiprocessing as mp
# Import custom libraries.
CustDirs = [os.path.abspath('../../HEC5Q')]
# The following conditional statement is required when re-running a kernal.
import data_transfer_cs3_2_hec5Q as hecdt
import HEC5Q


# %% Define functions.
def main():
    # Identify all model runs.
    working_subdir = os.listdir('../__models/CalSim3')
    models = list()
    for sd in working_subdir:
        models.append(sd)
    # Copy SV and DV files to HEC5Q folder.
    list_data = list()
    for m in models:
        srce_dir = os.path.join('../__models/CalSim3', m)
        fpSV = os.path.join(srce_dir, 'common/DSS/CS3L2015SVClean_USBR.dss')
        fpDV = os.path.join(srce_dir, 'CONV/DSS/CS3ROC_COS.dss')
        targ_dir = os.path.join('../__models/HEC5Q/HEC5Q/data', m)
        if not os.path.exists(targ_dir):
            os.mkdir(targ_dir)
        _ = hecdt.transfer_data(fpSV, fpDV, OutDir=targ_dir)
        SV_out = os.path.abspath(os.path.join(targ_dir, '2020D09ESV_3.dss'))
        DV_out = os.path.abspath(os.path.join(targ_dir, '2020D09EDV_3.dss'))
        list_data.append((SV_out, DV_out, m))
    # Return data list.
    return list_data


# %% Execute script.
if __name__ == '__main__':
    # Get data list.
    list_data = main()
    # Break data list into chunks.
    chunk = 5
    data_packet = [list_data[i:i + chunk]
                   for i in range(0, len(list_data), chunk)]
    # Run HEC5Q studies.
    for packet in data_packet:
        processes = [mp.Process(target=HEC5Q.HEC5Q_protocol,
                                args=(data[0], data[1]),
                                kwargs={'study': data[2],
                                        'watershed': 'sr',
                                        'climate': 'Q0'})
                     for data in packet]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
    # Print message to console.
    print('End of multiprocessing.')
