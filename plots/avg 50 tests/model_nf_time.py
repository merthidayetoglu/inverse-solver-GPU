import numpy as np
from sklearn.linear_model import LinearRegression

# Data
N = np.array([16384, 65536, 262144, 1048576, 4194304, 16777216, 16384, 65536, 262144, 1048576, 4194304, 16777216, 65536, 262144, 1048576, 4194304, 16777216])
t = np.array([16, 16, 16, 16, 16, 16, 64, 64, 64, 64, 64, 64, 256, 256, 256, 256, 256])
nearFieldTime = np.array([0.00021, 0.000668, 0.00221, 0.008548, 0.033181, 0.131954, 0.00058, 0.002478, 0.009443, 0.033031, 0.131819, 0.526557, 0.009944, 0.034378, 0.132116, 0.531288, 2.134629])
farFieldTime = np.array([0.000784, 0.002946, 0.011536, 0.0468, 0.185277, 0.73452, 0.000369, 0.001484, 0.00646, 0.022897, 0.093742, 0.366677, 0.001389, 0.004552, 0.017097, 0.070127, 0.274772])

# Transform data
log_N = np.log(N)
log_t = np.log(t)
log_nearFieldTime = np.log(nearFieldTime)
log_farFieldTime = np.log(farFieldTime)

# Reshape for sklearn
X = np.column_stack((log_N, log_t))  # Independent variables (log(N) and log(t))
y_near = log_nearFieldTime  # Dependent variable for nearFieldTime
y_far = log_farFieldTime    # Dependent variable for farFieldTime

# Fit multiple linear regression for farFieldTime
model_far = LinearRegression()
model_far.fit(X, y_far)
a_far, b_far = model_far.coef_
intercept_far = model_far.intercept_

# Fit multiple linear regression for nearFieldTime
model_near = LinearRegression()
model_near.fit(X, y_near)
a_near, b_near = model_near.coef_
intercept_near = model_near.intercept_

# Results
print("FarFieldTime Model: log(farFieldTime) = {:.4f} * log(N) + {:.4f} * log(t) + {:.4f}".format(a_far, b_far, intercept_far))
print("NearFieldTime Model: log(nearFieldTime) = {:.4f} * log(N) - {:.4f} * log(t) + {:.4f}".format(a_near, -b_near, intercept_near))