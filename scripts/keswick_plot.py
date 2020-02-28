"""
Summary
-------
The script helps to visualize the Keswick releases in each model run.

"""
# %% Import libraries.
# Import standard libraries.
import os
import sys
# Import third party libraries.
import pandas as pd
# Import custom modules.
import custom_modules
import calsim_toolkit as cs


# %% Define functions.
def main():
    # Identify all model runs.
    working_subdir = os.listdir('../__models/CalSim3')
    models = list()
    for sd in working_subdir:
        models.append(sd)
    # Generate list of DSS output files.
    list_dss = list()
    for m in models:
        srce_dir = os.path.join('../__models/CalSim3', m)
        fpDV = os.path.join(srce_dir, 'CONV/DSS/CS3ROC_COS.dss')
        list_dss.append(fpDV)
    # Query DSS files.
    df = cs.read_dss(list_dss, b='C_KSWCK', end_date='2015-09-30',
                     studies=models).cs.wide()
    diff_list = list()
    for m in models[1:]:
        diff_list.append((df[m] - df[models[0]]).copy())
    diff = pd.concat(diff_list, keys=models[1:], names=['Study'], axis=1)
    _ = diff.cs.plot()
    # Return success indicator.
    return 0


# %% Execute script.
if __name__ == '__main__':
    main()