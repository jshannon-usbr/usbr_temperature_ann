"""
Summary
-------
Artificial Neural Network to determine Clear Creek temperature given Shasta
storage, flow, and outflow.

"""
# %% Import libraries.
# Import third party libraries.
import numpy as np


# %% Define functions.
def relu(x):
    return np.where(x > 0, x, 0)


def temp_ann(S_SHSTA_0, S_SHSTA_1, S_SHSTA_2, S_SHSTA_3, I_SHSTA_0, I_SHSTA_1,
             I_SHSTA_2, I_SHSTA_3, C_KSWCK_0, C_KSWCK_1, C_KSWCK_2, C_KSWCK_3):
    """
    Notes
    -----
    Where t = 0, provide the current time step. Where t = 1, provide the
    1-month prior time step value. Repeat this pattern for all other time
    steps.

    Parameters
    ----------
    S_SHSTA_t : units TAF
        Volume of Shasta Reservoir in CalSim3.
    I_SHSTA_t : units CFS
        Flow of Shasta Inflow in CalSim3.
    C_KSWCK_t : units CFS
        Flow of Keswick Release in CalSim3.

    Returns
    -------
    y : units Degrees Fahrenheit
        Channel temperature of the Sacramento River Below Clear Creek.

    """
    # Construct input array.
    x = np.array([S_SHSTA_0, S_SHSTA_1, S_SHSTA_2, S_SHSTA_3,
                  I_SHSTA_0, I_SHSTA_1, I_SHSTA_2, I_SHSTA_3,
                  C_KSWCK_0, C_KSWCK_1, C_KSWCK_2, C_KSWCK_3])
    # Pass through hidden layer 1.
    W1 = np.loadtxt('W1.txt')
    B1 = np.loadtxt('B1.txt')
    h1 = relu(np.dot(W1, x) + B1)
    # Pass through hidden layer 2.
    W2 = np.loadtxt('W2.txt')
    B2 = np.loadtxt('B2.txt')
    h2 = relu(np.dot(W2, h1) + B2)
    # Pass through hidden layer 3.
    W3 = np.loadtxt('W3.txt')
    B3 = np.loadtxt('B3.txt')
    h3 = relu(np.dot(W3, h2) + B3)
    # Pass through hidden layer 4.
    W4 = np.loadtxt('W4.txt')
    B4 = np.loadtxt('B4.txt')
    h4 = relu(np.dot(W4, h3) + B4)
    # Pass through output layer.
    WO = np.loadtxt('WO.txt')
    BO = np.loadtxt('BO.txt')
    y = relu(np.dot(WO, h4) + BO)
    print(y)
    # Return result.
    return y


# %% Execute script.
if __name__ == '__main__':
    S_SHSTA_0, S_SHSTA_1, S_SHSTA_2, S_SHSTA_3 = [4500, 3500, 2500, 3000]

    y = temp_ann(S_SHSTA_0, S_SHSTA_1, S_SHSTA_2, S_SHSTA_3,
                 I_SHSTA_0, I_SHSTA_1, I_SHSTA_2, I_SHSTA_3,
                 C_KSWCK_0, C_KSWCK_1, C_KSWCK_2, C_KSWCK_3)
    print(y)
