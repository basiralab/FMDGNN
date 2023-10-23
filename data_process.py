import random
import numpy as np
import os
from torch.utils.data import Dataset

from utils import *
from config import *
from model import *


class BrainDataset(Dataset):
    def __init__(self, hospital_domain):
        super().__init__()
        self.source_domain = 0
        self.num_domains = 6
        self.hospital_domain = hospital_domain
        self.tgt_features = []
        for domain in range(self.num_domains):
            if domain == 0:
                self.src_feature = self.hospital_domain[domain]
            else:
                tgt_feature = self.hospital_domain[domain]
                self.tgt_features.append(tgt_feature)

    def __len__(self):
        return self.hospital_domain[0].shape[0]  # 192

    def __getitem__(self, index):
        # 一共192个subject，每个subject595个feature
        src = self.src_feature[index]
        tgt1 = self.tgt_features[0][index] * 100
        # tgt1 = self.tgt_features[0][index]
        tgt2 = self.tgt_features[1][index]
        tgt3 = self.tgt_features[2][index]
        tgt4 = self.tgt_features[3][index]
        # tgt5 = self.tgt_features[4][index]
        tgt5 = self.tgt_features[4][index] * 10
        return src, tgt1, tgt2, tgt3, tgt4, tgt5


def generate_random_views(n, seed=42):
    if seed is not None:
        random.seed(seed)

    sequences = []

    for _ in range(n):
        # Randomly choose the size of the sequence
        size = random.randint(1, 5)

        # Randomly sample integers without replacement
        seq = sorted(random.sample(range(1, 6), size))

        sequences.append(seq)

    return sequences


def get_hospital(n):
    hospital_list = []
    views = generate_random_views(n, seed=100)
    encoder = GCNencoder(nfeat=595, nhid=32)
    decoders = [GCNdecoder(nfeat=595, nhid=32) for _ in range(5)]

    for view in views:
        hospital_list.append(Hospital(encoder=encoder, decoders=decoders, views=view).to(device))

    return hospital_list


def get_dataset(file_dir, sizes, seed=42):
    # change to simulated data later
    file_num = []
    for i in range(1, 1374):
        if os.path.exists(os.path.join(file_dir, f"data{i}.mat")):
            file_num.append(i)

    np.random.seed(seed)
    shuffled_indices = np.random.permutation(N_SUBJECTS)
    groups = []
    start = 0
    for size in sizes:
        groups.append(shuffled_indices[start:start + size])
        start += size

    idx_list = []
    for group_id in range(len(groups)):
        idx_list.append([file_num[i] for i in groups[group_id]])

    hospital_domains_list = []
    for idx in range(len(idx_list)):
        hospital_domains_list.append(get_source_target_domain(file_dir, 6, hospital_idx=idx_list[idx]))

    dataset_list = []
    for domain in hospital_domains_list:
        dataset_list.append(BrainDataset(domain))

    return idx_list, hospital_domains_list, dataset_list


# simulated data
def get_simulated_hos_domains(n_domains, size):
    domain_list = []
    for domain in range(n_domains):
        hos_domain = torch.randn(size, 595).to(device)
        domain_list.append(hos_domain)
    return domain_list


def get_simulated_dataset():
    hospital_1_domains = get_simulated_hos_domains(n_domains=6, size=192)
    hospital_2_domains = get_simulated_hos_domains(n_domains=6, size=192)
    hospital_3_domains = get_simulated_hos_domains(n_domains=6, size=192)
    test_domains = get_simulated_hos_domains(n_domains=6, size=102)

    dataset_1 = BrainDataset(hospital_1_domains)
    dataset_2 = BrainDataset(hospital_2_domains)
    dataset_3 = BrainDataset(hospital_3_domains)
    test_dataset = BrainDataset(test_domains)

    datasets = [dataset_1, dataset_2, dataset_3, test_dataset]
    domains = [hospital_1_domains, hospital_2_domains, hospital_3_domains, test_domains]

    return domains, datasets
