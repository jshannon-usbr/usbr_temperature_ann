"""

Summary
-------
The goal of this Python module is to transfer a list of CalSim3 SV and DV
variables into DSS files that can be processed by the CalSim3 Version of
HEC5Q (HEC-5Q-CS3).

Future Development
------------------
1. Need to update function `StoreRTS` in module `cs_otfa` for use in this
module.

"""
# %% Import libraries.
# Import standard libraries.
import os
import sys
import argparse
# Import custom modules libraries.
import custom_modules
import calsim_toolkit as cs


# %% Define functions.
def transfer_data(fpSV, fpDV, OutDir=None):
    r"""
    No documentation as of 2019-01-03.

    """
    # %% Create list of CalSim3 variables needed for HEC-5Q-CS3.
    # The list below comes from SR_CS-CS3.dat variables in the 'get' function.
    CS3_Var = [  # Trinity River
               'S_TRNTY', 'I_TRNTY', 'E_TRNTY', 'C_TRNTY', 'Something Crazy', 'E_LWSTN',
               'I_LWSTN', 'D_LWSTN_CCT011',
               # Clear Creek
               'S_WKYTN', 'I_CLR025', 'E_WKYTN', 'C_WKYTN', 'D_WKYTN_SPT003',
               'D_WKYTN_02_PU', 'D_WKYTN_WTPBUK', 'D_WKYTN_WTPCSD',
               # Sacramento River
               'S_SHSTA', 'I_SHSTA', 'E_SHSTA', 'C_SHSTA', 'D_SHSTA_WTPJMS',
               'C_KSWCK', 'C_SAC287', 'C_SAC277', 'C_SAC271', 'C_SAC269',
               'C_SAC259', 'C_SAC257', 'C_SAC254', 'C_SAC247', 'C_SAC229',
               'C_SAC217', 'C_SAC201', 'C_SAC178', 'C_SAC169', 'C_SAC162',
               'C_SAC146', 'C_SAC122', 'SP_SAC193_BTC003', 'SP_SAC188_BTC003',
               'SP_SAC178_BTC003', 'SP_SAC159_BTC003', 'SP_SAC148_BTC003',
               'SP_SAC122_SBP021', 'C_SAC097', 'C_SAC091',
               # Sutter Bypass
               'C_SSL001',
               # Stony Creek
               'S_BLKBT', 'I_BLKBT', 'C_SGRGE', 'E_BLKBT', 'C_BLKBT',
               'C_STN004']
    # %% Set additional process parameters.
    # Establish DSS Output parameters.
    OutSV = '2020D09ESV_3.dss'
    OutDV = '2020D09EDV_3.dss'
    Part_A = 'CALSIM'
    Part_F = '2020D09E'
    # Construct output DSS file paths.
    if OutDir:
        oSVpth = os.path.realpath(os.path.join(OutDir, OutSV))
        oDVpth = os.path.realpath(os.path.join(OutDir, OutDV))
    else:
        oSVpth = os.path.realpath(os.path.join(os.getcwd(), OutSV))
        oDVpth = os.path.realpath(os.path.join(os.getcwd(), OutDV))
    # Read data.
    SV_DF = cs.read_dss(fpSV, b=CS3_Var)
    DV_DF = cs.read_dss(fpDV, b=CS3_Var)
    # Replace Part F.
    cs.transform.split_pathname(SV_DF, inplace=True)
    SV_DF['Part F'] = Part_F
    cs.transform.join_pathname(SV_DF, inplace=True)
    cs.transform.split_pathname(DV_DF, inplace=True)
    DV_DF['Part F'] = Part_F
    cs.transform.join_pathname(DV_DF, inplace=True)
    # Write DSS files to disk.
    SV_DF.cs.to_dss(oSVpth)
    DV_DF.cs.to_dss(oDVpth)
    # Return success indicator.
    return 0


# %% Execute script.
if __name__ == '__main__':
    # TODO: Update reading and writing functions of `cs_otfa` module.
    # <JAS 2019-01-03>
    # Run test function.
    OutDir = '../../HEC5Q/data/CalSim3-Base'
    fpSV = '../../CalSim3-Base/common/DSS/CS3L2015SVClean_wHD.dss'
    fpDV = '../../CalSim3-Base/CONV/DSS/2019-09-22_USBR_DV-Base.dss'
    transfer_data(fpSV, fpDV, OutDir=OutDir)
    # Return error while code is in development.
    dev_msg = ('This script is not ready to accept arguments from the command'
               ' line.')
    print(dev_msg)
    quit()
    # Initialize argument parser.
    parser = argparse.ArgumentParser()
    # Add positional arguments to parser.
    parser.add_argument('fpSV',
                        help=('File path for CalSim3 state variable DSS. Input'
                              + ' absolute path or file name if file is'
                              + ' located in the same directory as this'
                              + ' module.'),
                        type=str)
    parser.add_argument('fpDV',
                        help=('File path for CalSim3 decision variable DSS.'
                              + ' Input absolute path or file name if file'
                              + ' is located in the same directory as this'
                              + ' module.'),
                        type=str)
    # Add optional arguments.
    parser.add_argument('-od',
                        '--OutDir',
                        help=('Directory path for DSS files for HEC-5Q'
                              + ' for CalSim3 model. Default is the current'
                              + ' working directory.'),
                        type=str)
    # Parse arguments.
    args = parser.parse_args()
    transfer_data(args.fpSV, args.fpDV, OutDir=args.OutDir)
