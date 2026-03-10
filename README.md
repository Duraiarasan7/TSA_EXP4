# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES

# Date: 
# 212224230071


### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:

```

# Import necessary Modules and Functions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# Load dataset
data=pd.read_csv('/content/Tesla Dataset.csv')

# Declare required variables and set figure size, and visualise the data

N=1000
plt.rcParams['figure.figsize'] = [12, 6] #plt.rcParams is a dictionary-like object in Matplotlib that stores global settings for plots. The "rc" in rcParams stands for runtime configuration. It allows you to customize default styles for figures, fonts, colors, sizes, and more.

X=data['Open']
plt.plot(X)
plt.title('Original Data')
plt.show()
plt.subplot(2, 1, 1)
plot_acf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data ACF')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=len(X)/4, ax=plt.gca())
plt.title('Original Data PACF')
plt.tight_layout()
plt.show()

# Fitting the ARMA(1,1) model and deriving parameters

arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']

# Simulate ARMA(1,1) Process

ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(1,1)

plot_acf(ARMA_1)
plt.show()
plot_pacf(ARMA_1)
plt.show()

# Fitting the ARMA(1,1) model and deriving parameters

arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']


# Simulate ARMA(2,2) Process

ar2 = np.array([1, -phi1_arma22, -phi2_arma22])  
ma2 = np.array([1, theta1_arma22, theta2_arma22])  
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()

# Plot ACF and PACF for ARMA(2,2)

plot_acf(ARMA_2)
plt.show()
plot_pacf(ARMA_2)
plt.show()



```
### OUTPUT:


 # ORIGINAL DATA

<img width="1353" height="665" alt="image" src="https://github.com/user-attachments/assets/20aef934-6c30-4041-91e0-e9031532912d" />

# Partial Autocorrelation

<img width="1385" height="330" alt="image" src="https://github.com/user-attachments/assets/f6638bd8-9af1-46ab-8140-c5cb7ae98609" />

# Autocorrelation

<img width="1390" height="330" alt="image" src="https://github.com/user-attachments/assets/c8d393ad-3a74-4f3b-af80-1f6de94836e0" />


 # SIMULATED ARMA(1,1) PROCESS:

<img width="1343" height="661" alt="image" src="https://github.com/user-attachments/assets/b81ff3e9-a9da-4205-b622-9fea4cb8edc0" />

# Partial Autocorrelation:

 <img width="1265" height="665" alt="image" src="https://github.com/user-attachments/assets/b586019a-81fc-4b76-983e-576bb038fb81" />

# Autocorrelation:

<img width="1328" height="655" alt="image" src="https://github.com/user-attachments/assets/cc807a9c-7e21-4a03-acdd-4c61374fecd5" />


# SIMULATED ARMA(2,2) PROCESS:

<img width="1354" height="661" alt="image" src="https://github.com/user-attachments/assets/61e6c112-9826-4d32-860c-b613de0b81c2" />


# Partial Autocorrelation

<img width="1363" height="671" alt="image" src="https://github.com/user-attachments/assets/b55998ac-81f5-4c4e-9464-ad9ca961e740" />



# Autocorrelation

<img width="1286" height="659" alt="image" src="https://github.com/user-attachments/assets/43b4a31f-ceff-4e16-bcec-064b1a05e38d" />




### RESULT:

Thus, a python program is created to fir ARMA Model successfully.
