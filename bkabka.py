import os

import numpy as np
from matplotlib import pyplot as plt

output_dir = "C:/Users/vanholk/sparsifier/output_sketches/camel/test/"

config_path = os.path.join(output_dir, 'config.npy')

config_file = np.load(config_path, allow_pickle=True).item()

loss_eval = config_file.get('loss_eval', [])

plt.figure()
plt.plot(loss_eval, label='Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss During Training')
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(output_dir, "filename"))
plt.show()