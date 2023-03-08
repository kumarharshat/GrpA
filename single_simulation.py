import torch; torch.set_default_dtype(torch.float64)
import yaml
from main import run_simulation

# Comment this when runnign the tuning file
with open("trainingconfig.yaml", "r") as stream:
    try:
        tconfig = yaml.safe_load(stream)
        # print(yaml.safe_load(stream))
    except yaml.YAMLError as exc:
        print(exc)

run_simulation(tconfig, verbose = True)