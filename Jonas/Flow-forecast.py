import pandas as pd
import numpy as np
from pathlib import Path
import torch.nn as nn

data = pd.read_csv(Path('total_data.csv'))
data = data[["total_kWhDelivered", "carsCharging", "carsIdle"]]

transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
src = torch.rand((10, 32, 512))
tgt = torch.rand((20, 32, 512))
out = transformer_model(src, tgt)

