import os
import sys

from experiment import ToolsExperiment
from config import configs

import numpy as np
np.random.seed(1)
import random
random.seed(1)


log_dir = sys.argv[1]
config_name = sys.argv[2]
trial = sys.argv[3]

if not os.path.exists(log_dir):
    os.mkdir(log_dir)


xp = ToolsExperiment(config=configs[config_name], log_dir=log_dir)
    
xp.trial = trial

xp.start_trial()