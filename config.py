import torch

N_SUBJECTS = 678  # number of subjects
N_ROI = 35  # number of ROIs
N_FEATURE = int(N_ROI * (N_ROI - 1) / 2)  # number of features
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = "saved_trained/"

# parameters
fed_rounds = 10
epochs_per_round = 10
batch_size = 32
res = False
aggr = "FedAvg"
n_folds = 4
