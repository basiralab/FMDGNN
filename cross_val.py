import numpy as np
import torch
from torch.utils.data import DataLoader

from utils import *
from data_process import *
from train import *
from evaluate import *


def k_fold_split(hospital_idx, n_split):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(hospital_idx))
    split_indices = np.array_split(shuffled_indices, n_split)

    return split_indices


def train_test_k_fold(hospital_idx, n_folds):
    split_indices = k_fold_split(hospital_idx, n_folds)
    folds = []
    for k in range(n_folds):
        test_indices = split_indices[k]
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds


def get_cross_datasets(ids, n_folds):
    cross_datasets = []
    for hospital in range(len(ids)):
        hospital_datasets = []
        for i, (train_indices, test_indices) in enumerate(train_test_k_fold(ids[hospital], n_folds)):
            train_ids = [ids[hospital][i] for i in train_indices]
            test_ids = [ids[hospital][i] for i in test_indices]
            train_domains = get_source_target_domain("data LH/data LH", 6, train_ids)
            test_domains = get_source_target_domain("data LH/data LH", 6, test_ids)
            train_dataset = BrainDataset(train_domains)
            test_dataset = BrainDataset(test_domains)

            hospital_datasets.append([train_dataset, test_dataset])
        cross_datasets.append(hospital_datasets)
    return cross_datasets


def get_simulated_cross_datasets(n_folds):
    cross_datasets = []
    for hospital in range(4):
        hospital_datasets = []
        for i in range(n_folds):
            train_domains = get_simulated_hos_domains(n_domains=6, size=48*3)
            test_domains = get_simulated_hos_domains(n_domains=6, size=48)
            train_dataset = BrainDataset(train_domains)
            test_dataset = BrainDataset(test_domains)
            # losses = train()
            # test_losses[hospital].append(losses)

            hospital_datasets.append([train_dataset, test_dataset])
        cross_datasets.append(hospital_datasets)
    return cross_datasets


def baseline_cross_validation(cross_datasets, global_test_dataset, n_folds, res):
    learning_rate = 0.001
    num_epochs = 30

    dataset_1 = cross_datasets[0]  # [Fold0, F1, F2, F3]
    dataset_2 = cross_datasets[1]
    dataset_3 = cross_datasets[2]

    best_test_loss_1 = 0.
    best_test_loss_2 = 0.
    best_test_loss_3 = 0.
    best_model_1 = None
    best_model_2 = None
    best_model_3 = None

    fold_test_loss_1 = []
    fold_test_loss_2 = []
    fold_test_loss_3 = []

    fold_global_test_loss_1 = []
    fold_global_test_loss_2 = []
    fold_global_test_loss_3 = []

    fold_train_loss_log = []

    for fold in range(n_folds):
        if not res:
            encoder = GCNencoder(nfeat=595, nhid=32)
            decoders = [GCNdecoder(nhid=32, nfeat=595) for _ in range(5)]
        else:
            encoder = ResGCNencoder(nfeat=595, nhid=32)
            decoders = [ResGCNdecoder(nhid=32, nfeat=595) for _ in range(5)]

        model_1 = Hospital(encoder=encoder, decoders=decoders, views=[1, 3]).to(device)
        model_2 = Hospital(encoder=encoder, decoders=decoders, views=[1, 2, 5]).to(device)
        model_3 = Hospital(encoder=encoder, decoders=decoders, views=[2, 3, 4, 5]).to(device)

        # train data
        local_data_1 = dataset_1[fold][0]
        local_data_2 = dataset_2[fold][0]
        local_data_3 = dataset_3[fold][0]

        train_loader_1 = DataLoader(local_data_1, batch_size=batch_size, shuffle=True)
        train_loader_2 = DataLoader(local_data_2, batch_size=batch_size, shuffle=True)
        train_loader_3 = DataLoader(local_data_3, batch_size=batch_size, shuffle=True)

        # test data
        local_test_data_1 = dataset_1[fold][1]
        local_test_data_2 = dataset_2[fold][1]
        local_test_data_3 = dataset_3[fold][1]

        test_loader_1 = DataLoader(local_test_data_1)
        test_loader_2 = DataLoader(local_test_data_2)
        test_loader_3 = DataLoader(local_test_data_3)

        # Train
        train_loss_log_1 = train_baseline(model_1, train_loader_1, lr=learning_rate,
                                          num_epochs=num_epochs, device=device)
        train_loss_log_2 = train_baseline(model_2, train_loader_2, lr=learning_rate,
                                          num_epochs=num_epochs, device=device)
        train_loss_log_3 = train_baseline(model_3, train_loader_3, lr=learning_rate,
                                          num_epochs=num_epochs, device=device)
        fold_train_loss_log.append([train_loss_log_1, train_loss_log_2, train_loss_log_3])

        # Evaluate each fold
        test_loss_1, _ = evaluate(model_1, test_loader_1, device)
        test_loss_2, _ = evaluate(model_2, test_loader_2, device)
        test_loss_3, _ = evaluate(model_3, test_loader_3, device)
        fold_test_loss_1.append(test_loss_1)
        fold_test_loss_2.append(test_loss_2)
        fold_test_loss_3.append(test_loss_3)


        global_test_loader = DataLoader(global_test_dataset)
        global_test_loss_1, _ = evaluate(model_1, global_test_loader, device)
        global_test_loss_2, _ = evaluate(model_2, global_test_loader, device)
        global_test_loss_3, _ = evaluate(model_3, global_test_loader, device)
        fold_global_test_loss_1.append(global_test_loss_1)
        fold_global_test_loss_2.append(global_test_loss_2)
        fold_global_test_loss_3.append(global_test_loss_3)

        if global_test_loss_1 > best_test_loss_1:
            best_model_1 = model_1
        if global_test_loss_2 > best_test_loss_2:
            best_model_2 = model_2
        if global_test_loss_3 > best_test_loss_3:
            best_model_3 = model_3

    best_models = [best_model_1, best_model_2, best_model_3]
    fold_eval_loss = [fold_test_loss_1, fold_test_loss_2, fold_test_loss_3]
    global_eval_loss = [fold_global_test_loss_1, fold_global_test_loss_2, fold_global_test_loss_3]

    return best_models, fold_train_loss_log, fold_eval_loss, global_eval_loss


def cross_validation(hospital_list, cross_datasets, global_test, n_folds, res, aggr):
    cross_dataset_list = [cross_datasets[i] for i in range(len(cross_datasets))]

    cross_losses = []

    fold_local_eval = []
    fold_global_eval = []

    local_fold = []
    global_fold = []

    best_MAE = 0.
    best_clients = None
    best_server = None

    for fold in range(n_folds):
        local_datasets = []
        local_tests = []
        for i in range(len(hospital_list)):
            local_datasets.append(cross_dataset_list[i][fold][0])
            local_tests.append(cross_dataset_list[i][fold][1])

        train_output = train(hospital_list, local_datasets, local_tests, global_test, res, aggr)
        [clients, server, losses, local_eval, global_eval] = train_output
        cross_losses.append(losses)
        for i in range(len(local_eval)):
            fold_local_eval.append(local_eval[i])
            fold_global_eval.append(global_eval[i])

        local_fold.append(fold_local_eval)
        global_fold.append(fold_global_eval)

        if np.mean(global_eval[0]) > best_MAE:
            best_clients = clients
            best_server = server

    return [best_clients, best_server, cross_losses, local_fold, global_fold]
