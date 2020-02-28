r"""
Summary
-------
This Python module executes the Simulation Protocols outlined by David
Julian/CH2M HILL, dated 2015-05-26. The following Temperature Model Simulation
Protocols, with a common directory of ./_Tools/HEC5Q_Toolkit, were delivered to
Reclamation on 2016-01-13:
    - American_River/HEC5Q_AR_Temp_Model_Protocol_052615.docx
    - Stanislaus_River/HEC5Q_StanR_Temp_Model_Protocol_052615.docx
    - Trinity_Sacramento_Rivers/HEC5Q_SR_Temp_Model_Protocol_052615.docx

"""
# %% Import libraries.
# Import standard libraries.
import os
import sys
import shutil
import subprocess as sb
# Import third party libraries.
import pandas as pd
import numpy as np
# Import custom libraries.
CustDir = os.path.abspath(r'..\usbr_py3dss')
# The following conditional statement is required when re-running a kernal.
if CustDir not in sys.path:
    sys.path.insert(1, CustDir)
import dss3_functions_reference as dss

# %% Establish functions.
def AR_targets(model_dir):
    r"""
    Summary
    -------
    Function to generate temperature targets for American River HEC5Q model.

    """
    # Read target tables from AmerR_Temp_Sel_Tool_rev15_APP_FINAL_3-16-15.xlsm.
    table_path = (r'Pre_Processor'
                  + r'\AmerR_Temp_Sel_Tool_rev15_APP_FINAL_3-16-15.xlsm')
    table_path = os.path.join(model_dir, table_path)
    col_names = ['Storage Plus Inflow (TAF)'] + list(range(1, 13))
    sheetname = 'Input Schedules Selected'
    F_targ = pd.read_excel(table_path, sheet_name=sheetname, header=None,
                              names=col_names, index_col=0, usecols='D:P',
                              skiprows=list(range(9)), nrows=23)
    W_targ = pd.read_excel(table_path, sheet_name=sheetname, header=None,
                                names=col_names, index_col=0, usecols='D:P',
                                skiprows=list(range(37)), nrows=23)
    F_targ.columns.set_names('Calendar Month Number', inplace=True)
    W_targ.columns.set_names('Calendar Month Number', inplace=True)
    # Read I300, I8, and S8 from CalSimII SV & DV file.
    # ???: Spreadsheet also queries C301, but it looks like it is not used; why
    #      is C301 needed?
    # <JAS 2019-04-16>
    cdate = '31Oct1921'
    ctime = '2400'
    nvalsi = 984
    DateTime = pd.date_range(start='1921-10-31', end='2003-09-30', freq='M')
    Folsom = pd.DataFrame(index=DateTime)
    SV = [r'/CALSIM/I300/FLOW-INFLOW//1MON/2020D09E/',
          r'/CALSIM/I8/FLOW-INFLOW//1MON/2020D09E/']
    DV = [r'/CALSIM/S8/STORAGE//1MON/2020D09E/']
    fpSV = os.path.join(model_dir, r'Pre_Processor\2020D09ESV.dss')
    ifltab_SV = dss.open_dss(fpSV)[0]
    dss_rtn = dss.read_regtsd(ifltab_SV, SV[0], cdate, ctime, nvalsi)
    Folsom['I300'] = dss_rtn[1]
    dss_rtn = dss.read_regtsd(ifltab_SV, SV[1], cdate, ctime, nvalsi)
    Folsom['I8'] = dss_rtn[1]
    dss.close_dss(ifltab_SV)
    fpDV = os.path.join(model_dir, r'Pre_Processor\2020D09EDV.dss')
    ifltab_DV = dss.open_dss(fpDV)[0]
    dss_rtn = dss.read_regtsd(ifltab_DV, DV[0], cdate, ctime, nvalsi)
    Folsom['S8'] = dss_rtn[1]
    dss.close_dss(ifltab_DV)
    # Sum S8 End of May Storage and June through September inflow (I8 & I300).
    cfs2taf_I300 = lambda x: x['I300'] * 86400 * x.name.day / 43560 / 1000
    Folsom['I300'] = Folsom.apply(cfs2taf_I300, axis=1)
    cfs2taf_I8 = lambda x: x['I8'] * 86400 * x.name.day / 43560 / 1000
    Folsom['I8'] = Folsom.apply(cfs2taf_I8, axis=1)
    Folsom = Folsom.iloc[::-1]
    Folsom['Storage'] = (Folsom['S8']
                         + Folsom['I300'].shift(1).rolling(4).sum()
                         + Folsom['I8'].shift(1).rolling(4).sum())
    Folsom = Folsom.iloc[::-1]
    Folsom = Folsom.loc[Folsom.index.month == 5, :]
    # Re-index monthly series with each month equal to its year's May value.
    Folsom = Folsom.shift(7, freq='M')
    Folsom = Folsom.reindex(pd.date_range(start='1921-10-31',
                                          end='2003-09-30',
                                          freq='M'),
                            method='bfill')
    # Lookup temperature target based on summed volume.
    W_target = lambda x: W_targ.iloc[W_targ.index.get_loc(x['Storage'],
                                                          method='ffill'),
                                     W_targ.columns.get_loc(x.name.month)]
    Folsom['Watt Target'] = Folsom.apply(W_target, axis=1)
    F_target = lambda x: F_targ.iloc[F_targ.index.get_loc(x['Storage'],
                                                          method='ffill'),
                                     F_targ.columns.get_loc(x.name.month)]
    Folsom['Folsom Target'] = Folsom.apply(F_target, axis=1)
    # Re-index from monthly to daily series.
    Folsom = Folsom.reindex(pd.date_range(start='1921-10-01',
                                          end='2003-09-30',
                                          freq='D'),
                            method='bfill')
    # Store daily series to CALSIMII_HEC5Q.dss.
    fpHEC5Q = os.path.join(model_dir, r'Pre_Processor\CALSIMII_HEC5Q.dss')
    ifltab_HEC5Q = dss.open_dss(fpHEC5Q)[0]
    cpath = [r'/CALSIM_STOR/WATTAVE_PT/TARGET-F//1DAY/2020D09E-1/',
             r'/CALSIM_STOR/FOLSOM_PT/TARGET-F//1DAY/2020D09E-1/']
    cdate = '01Oct1921'
    cunits = 'DEGF'
    ctype = 'Per-aver'
    dss_storWatt = dss.write_regtsd(ifltab_HEC5Q, cpath[0], cdate, ctime,
                                    Folsom['Watt Target'].values, cunits,
                                    ctype)
    dss_storFlsm = dss.write_regtsd(ifltab_HEC5Q, cpath[1], cdate, ctime,
                                    Folsom['Folsom Target'].values, cunits,
                                    ctype)
    dss.close_dss(ifltab_HEC5Q)
    # Return success indicators.
    return (0, dss_storWatt, dss_storFlsm)


def SR_targets(model_dir):
    r"""
    Summary
    -------
    Function to generate temperature targets for Trinity/Sacramento River HEC5Q
    model.

    Notes
    -----
    1. Need to determine how SacR_Temp_Sel_Tool_rev05_FULL_FINAL_3-3-15.xlsm
       fits into this process.

    """
    # Get target table in SacR_Temp_Sel_Tool_rev05_APP_FINAL_3-3-15-16-15.xlsm.
    table_path = (r'Pre_Processor'
                  + r'\SacR_Temp_Sel_Tool_rev05_APP_FINAL_3-3-15.xlsm')
    table_path = os.path.join(model_dir, table_path)
    col_names = ['S4 EO Apr Storage (TAF)'] + list(range(1, 13))
    targ = pd.read_excel(table_path, sheet_name='Input Schedules Selected',
                         header=None, names=col_names, index_col=0,
                         usecols='E:Q', skiprows=list(range(7)), nrows=6)
    targ.columns.set_names('Calendar Month Number', inplace=True)
    # Read S4 from CalSimII DV file.
    # ???: Why are C5 and C109 queried for the excel spreadsheet? They do not
    #      seem to be used.
    # <JAS 2019-04-17>
    cdate = '31Oct1921'
    ctime = '2400'
    nvalsi = 984
    DateTime = pd.date_range(start='1921-10-31', end='2003-09-30', freq='M')
    Shasta = pd.DataFrame(index=DateTime)
    DV = [r'/CALSIM/S_SHSTA/STORAGE//1MON/2020D09E/']
    fpDV = os.path.join(model_dir, r'Pre_Processor\2020D09EDV.dss')
    ifltab_DV = dss.open_dss(fpDV)[0]
    dss_rtn = dss.read_regtsd(ifltab_DV, DV[0], cdate, ctime, nvalsi)
    Shasta['S4'] = dss_rtn[1]
    dss.close_dss(ifltab_DV)
    # Select only April Months.
    Shasta = Shasta.loc[Shasta.index.month == 4, :]
    # Re-index monthly series with each month equal to its year's April value.
    Shasta = Shasta.shift(8, freq='M')
    Shasta = Shasta.reindex(pd.date_range(start='1921-10-31', end='2003-09-30',
                                  freq='M'),
                    method='bfill')
    # Lookup temperature target based on S4 End of April storage level.
    target = lambda x: targ.iloc[targ.index.get_loc(x['S4'], method='ffill'),
                                 targ.columns.get_loc(x.name.month)]
    Shasta['Target'] = Shasta.apply(target, axis=1)
    # Re-index from monthly to daily series.
    Shasta = Shasta.reindex(pd.date_range(start='1922-01-01', end='2003-09-30',
                                          freq='D'),
                            method='bfill')
    # Store daily series to CALSIMII_HEC5Q.dss.
    fpHEC5Q = os.path.join(model_dir, r'Pre_Processor\CALSIMII_HEC5Q.dss')
    ifltab_HEC5Q = dss.open_dss(fpHEC5Q)[0]
    cpath = [r'/CALSIM_STOR/Shasta_PT/TARGET-F//1DAY/2020D09E-1/']
    cdate = '01Jan1922'
    cunits = 'DEGF'
    ctype = 'Per-aver'
    dss_stor = dss.write_regtsd(ifltab_HEC5Q, cpath[0], cdate, ctime,
                                Shasta['Target'].values, cunits, ctype)
    dss.close_dss(ifltab_HEC5Q)
    # Return success indicators.
    return (0, dss_stor)


def HEC5Q_protocol(SV, DV, study='NAA', watershed='SR', climate='Q5',
                   force_delete=False, safe_mode=True):
    r"""
    No documentation as of 2019-04-12.

    Notes
    -----
    1. Select `climate` scenario (i.e. q1, q2, q3, q4, or q5); string is not
       case sensitive.
    2. Select one of the following watersheds (string is not case sensitive):
        - 'AR' = American River
        - 'StanR' = Stanislaus River
        - 'SR' = Trinity & Sacramento Rivers

    """
    # Check inputs.
    if watershed.upper() not in ['AR', 'STANR', 'SR']:
        err_msg = watershed + ' is not a valid value for `watershed`.'
        raise ValueError(err_msg)
    # Stylize `watershed`.
    if watershed.upper() == 'STANR':
        watershed = 'StanR'
        # Check to ensure that `climate` is 'q5', per HEC5Q_Toolkit_102315_v21.
        if climate.lower() != 'q5':
            err_msg = 'Cannot run StanR model with climate other than Q5.'
            raise ValueError(err_msg)
    else:
        watershed = watershed.upper()
    # Change to HEC5Q directory.
    cwd = os.getcwd()
    this_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(this_dir)
    # Select HEC5Q_Toolkit directory.
    tool_map = {'AR': 'American_River',
                'StanR': 'Stanislaus_River',
                'SR': 'Trinity_Sacramento_Rivers'}
    tool_dir = tool_map[watershed]
    # Set model directory.
    model_name = '{}_HEC5Q_{}'.format(watershed, climate.upper())
    study_dir = os.path.abspath(os.path.join(r'./', study))
    model_dir = os.path.join(study_dir, model_name)
    # Check existence of model directory.
    dir_exists = os.path.isdir(model_dir)
    # If model directory exists, ...
    if dir_exists:
        # ...force delete and replace model directory without user input, or...
        if force_delete and not safe_mode:
            shutil.rmtree(model_dir)
            del_msg = 'Deleted content of existing model directory {}.'
            print(del_msg.format(model_dir))
            # NOTE: PermissionError raised if user had `model_dir` open in
            #       Windows File Explorer when this code was executed.
            # <JAS 2019-04-15>
            os.mkdir(model_dir)
        # ...prompt user to allow deletion and replacement of model directory.
        else:
            if force_delete and safe_mode:
                print('`safe_mode` is True.')
            prompt = 'Delete contents of existing model directory {}? Y/[N]: '
            ans = input(prompt.format(model_dir))
            if ans == 'Y':
                shutil.rmtree(model_dir)
                del_msg = 'Deleted content of existing model directory {}.'
                print(del_msg.format(model_dir))
                # NOTE: PermissionError raised if user had `model_dir` open in
                #       Windows File Explorer when this code was executed.
                # <JAS 2019-04-15>
                os.mkdir(model_dir)
            else:
                err_msg = 'Permission Denied: Cannot delete directory.'
                raise ValueError(err_msg)
    # If model directory does not exist, create directory.
    else:
        print('Directory does not exist. Created new directory.')
        if not os.path.isdir(study_dir):
            os.mkdir(study_dir)
        os.mkdir(model_dir)
    # Copy setup_{watershed}_temp_run.bat from .\_Tools\HEC5Q_Toolkit.
    setup_bat = 'setup_{}_temp_run.bat'.format(watershed)
    setup_bat = os.path.join(model_dir, setup_bat)
    setup_src = (r'.\_Tools\HEC5Q_Toolkit\{}'.format(tool_dir)
                 + r'\Common\setup_{}_temp_run.bat'.format(watershed))
    setup_src = os.path.abspath(setup_src)
    shutil.copyfile(setup_src, setup_bat)
    # Run setup_{watershed}_temp_run.bat with input `climate`.
    run_setup = sb.run(setup_bat, cwd=model_dir, input=climate.lower(),
                       encoding='utf-8', stdout=sb.PIPE)
    if run_setup.returncode == 0:
        print(r'Copied model files from .\_Tools\HEC5Q_Toolkit.')
    else:
        err_msg = 'Setup process interrupted.'
        print(err_msg)
        return run_setup.returncode
    # Copy CalSimII SV and DV files to {model_dir}\Pre_Processor.
    input_data = os.path.join(model_dir, 'Pre_Processor')
    shutil.copyfile(SV, os.path.join(input_data, '2020D09ESV.dss'))
    shutil.copyfile(DV, os.path.join(input_data, '2020D09EDV.dss'))
    print('Added CalSim files to Pre_Processor folder.')
    # Run process_calsim_temp_inputs.bat.
    inputs_bat = os.path.join(model_dir, 'process_calsim_temp_inputs.bat')
    print('Processing CalSim temperature inputs...')
    process_input = sb.run(inputs_bat, cwd=model_dir, encoding='utf-8',
                           creationflags=sb.CREATE_NEW_CONSOLE)
    if process_input.returncode == 0:
        print('Processed CalSim temperature inputs.')
    else:
        err_msg = 'Input processing interrupted.'
        print(err_msg)
        return process_input.returncode
    # Store temperature target schedules in input file.
    if watershed == 'AR':
        AR_targets(model_dir)
    elif watershed == 'SR':
        SR_targets(model_dir)
    # Run run_{watershed}_temp_model.bat.
    run_bat = 'run_{}_temp_model.bat'.format(watershed)
    run_bat = os.path.join(model_dir, run_bat)
    print('Running temperature model...')
    run_model = sb.run(run_bat, cwd=model_dir, encoding='utf-8',
                       creationflags=sb.CREATE_NEW_CONSOLE)
    if run_model.returncode == 0:
        print('Temperature model run complete!')
    else:
        err_msg = 'Temperature model process interrupted.'
        print(err_msg)
        return run_model.returncode
    # Switch back to original directory.
    os.chdir(cwd)
    # Print message to console and return success indicator.
    print('HEC5Q Subprocess Complete!')
    return 0


# %% Execute script.
if __name__ == '__main__':
    SV = os.path.abspath('./data/CalSim3-Base/2020D09ESV_3.dss')
    DV = os.path.abspath('./data/CalSim3-Base/2020D09EDV_3.dss')
    HEC5Q_protocol(SV, DV, study='CalSim3-Base', watershed='sr', climate='Q0')
    msg = 'This script is complete!'
    print(msg)
