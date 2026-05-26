# %%

import numpy as np
import matplotlib.pyplot as plt

# %%

# path_imu = "./DATA_BOOT_0055_TIME_20260120T211500_imu.npy"
path_imu = "/home/jeanr/Downloads/Test_data_2026_02_04/DATA_BOOT_0173_TIME_20260204T021500_imu.npy"

data_imu = np.load(path_imu, allow_pickle=True)

# %%

plt.figure()
plt.plot([entry.millis_reading for entry in data_imu], marker="*", linestyle=None)
plt.show()

# %%
