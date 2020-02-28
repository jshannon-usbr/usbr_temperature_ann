"""
Summary
-------
Python script to generate baseline datasets for water regime.

"""
# %% Import standard libraries.
# Import standard libraries.
import os
import sys
import shutil
import itertools
# Import custom modules.
import custom_modules
from calsim_toolkit.apps import run_CalSim


# %% Define functions.
def main(study):
    # Define weight priority and flow goal.
    # weights = [50, 100, 200, 1000]
    # flows = [25, 50, 75, 100, 125, 150, 175, 200, 1000, 2000]
    weights = [50]
    flows = [25, 50]
    var_combo = list(itertools.product(weights, flows))
    # Copy baseline study.
    study_name = os.path.split(study)[1]
    base_dir = '../__models/CalSim3/{}'.format(study_name)
    shutil.copytree(study, base_dir)
    # Initialize launch file list.
    list_lf = list()
    # Make copies of base study, modify code, and update launch file list.
    dir_copies = list()
    dir_fp = '../__models/CalSim3/' + study_name + '-w{}_f{}'
    new_code1 = '''! Set additional flow for temperature regulation.
define C_KSWCK_TEMP {std kind 'FLOW' UNITS 'CFS'}
! Define additional flow cap.'''
    new_code2 = '''! Save target cap.
define C_KSWCK_TEMP_TargDV {std kind 'target' units 'cfs'}
goal set_C_KSWCK_TEMP_TargDV {C_KSWCK_TEMP_TargDV = C_KSWCK_TEMP_Targ}
! Constrain temperature release to cap.
goal C_KSWCK_TEMP_Cap {C_KSWCK_TEMP < C_KSWCK_TEMP_Targ}
! Modified Keswick Releases below.'''
    for w, f in var_combo:
        # Copy base study.
        alt_dir = os.path.abspath(dir_fp.format(w, f))
        shutil.copytree(base_dir, alt_dir)
        # Modify ReOpsVarDefine.wresl, Weight-table.wresl, and CalSim3.launch.
        fp1 = 'common/ReOperations/ReOpsVarDefine.wresl'
        with open(os.path.join(alt_dir, fp1)) as fp:
            content1 = fp.read().split('\n')
        for i, line in enumerate(content1):
            if line.startswith('goal ShastaWhl'):
                mod_line = line.split('}')[0]
                mod_line += ' + C_KSWCK_TEMP}'
                content1[i] = mod_line
                for line in new_code2.split('\n')[::-1]:
                    content1.insert(i, line)
                content1.insert(i, 'define C_KSWCK_TEMP_Targ {value ' + str(f) + r'.}')
                for line in new_code1.split('\n')[::-1]:
                    content1.insert(i, line)
                break
        content1 = '\n'.join(content1)
        with open(os.path.join(alt_dir, fp1), 'w') as fp:
            fp.write(content1)
        fp2 = 'CONV/Run/System/SystemTables_ALL/Weight-table.wresl'
        with open(os.path.join(alt_dir, fp2)) as fp:
            content2 = fp.read().split('\n')
        for i, line, in enumerate(content2):
            if line.startswith('[C_KSWCK_MIF'):
                content2.insert(i + 1, '[C_KSWCK_TEMP,  ' + str(w) + '],')
                content2.insert(i + 1, '! Add weight to temperature release.')
                break
        content2 = '\n'.join(content2)
        with open(os.path.join(alt_dir, fp2), 'w') as fp:
            fp.write(content2)
        # Add launch file to the list.
        list_lf.append(os.path.join(alt_dir, 'CalSim3.launch'))
    # Run studies in parallel.
    _ = run_CalSim.main(list_lf, solver='cbc', run_parallel=True)
    # Return success indicator.
    return 0


# %% Execute script.
if __name__ == '__main__':
    study = '../../../CS3ROC_COS'
    main(study)
