import numpy as np
from scipy.interpolate import interp1d

# Known values of N and corresponding a
N_values = [65536, 262144, 1048576, 4194304, 16777216]
a_values = [8.097, 5.283, 4.115, 4.520, 4.032]  # Example values for t=256

# Interpolate a as a function of N
a_interpolator = interp1d(N_values, a_values, kind='linear', fill_value="extrapolate")

def shareTransfer(N, t):
    """
    Calculate shareTransfer based on N and t.
    """
    # Get interpolated value of a for given N
    a = a_interpolator(N)
    # Compute shareTransfer
    return a / t

# Example usage
N = 1048576
t = 64
print(f"shareTransfer for N={N}, t={t}: {shareTransfer(N, t)}")