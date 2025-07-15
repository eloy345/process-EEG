import mne

raw = mne.io.read_raw_fif("EEG_crop/PRE/KAIZEN/6/EEG/Samuel/Session01/RecordingS01R000_EDF/aacc_basal.fif", preload=False)
print(raw.ch_names)