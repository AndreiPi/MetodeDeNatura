import mne
import numpy as np
import scipy.io

train_file = "/home/augt/Public/MIN/MetodeDeNatura/GA Versions/data/raw/parsed_P01T.mat"
mat = scipy.io.loadmat(train_file)
raw_data = np.array(mat['RawEEGData'])
sfreq = mat['sampRate']

data = np.array(raw_data[0])
data = data[:, :200]
print(data.shape)

# Definition of channel types and names.
ch_types = ['grad' for i in range(12)]
ch_names = ['F3', 'FC3', 'C3', 'CP3', 'P3', 'FCz', 'CPz', 'F4', 'FC4', 'C4', 'CP4', 'P4']

info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
raw = mne.io.RawArray(data, info)

# Scaling of the figure.
# For actual EEG/MEG data different scaling factors should be used.
scalings = {'mag': 2, 'grad': 2}

raw.plot(n_channels=12, scalings=scalings, title='Data from arrays',
         show=True, block=True)

# It is also possible to auto-compute scalings
scalings = 'auto'  # Could also pass a dictionary with some value == 'auto'
raw.plot(n_channels=12, scalings=scalings, title='Auto-scaled Data from arrays',
         show=True, block=True)

print(raw.info)

