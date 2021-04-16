import numpy as np
import pandas as pd
import os

path = "../../output/RNN_weather/RNN_h2_ep15_bs64_maxsamples20000/20210320-135158/observations_pierre/"
txt_file = os.path.join(path, "donnees_pierre.txt")

df = pd.read_csv(txt_file, sep=';')

print(df)

states = df.values[:,:2]
states = states[np.newaxis, np.newaxis, :, :]
observations = df.values[:,2:]
observations = observations[np.newaxis, np.newaxis, :, :]

np.save(os.path.join(path, "states.npy"), states)
np.save(os.path.join(path, "observations.npy"), observations)