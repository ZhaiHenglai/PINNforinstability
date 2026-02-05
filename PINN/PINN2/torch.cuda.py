import os
import glob
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.cuda.reset_peak_memory_stats()
# ... 做完一次 loss.backward() 之后：
print("peak GB =", torch.cuda.max_memory_allocated() / 1024**3)
