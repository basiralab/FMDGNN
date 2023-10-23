import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def show_matrix(m: torch.Tensor, label):
    """
    m: brain grah 35 * 35
    label: graph name as title
    """
    m = m.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(20, 10))

    min_val = round((m.min()), 6)
    max_val = round((m.max()), 6)

    cax = ax.matshow(m, cmap=plt.cm.Spectral)
    cbar = fig.colorbar(cax, ticks=[min_val, float((min_val + max_val) / 2), max_val], pad=0.01)
    cbar.ax.set_yticklabels(['< %.2f' % (min_val), '%.2f' % (float((min_val + max_val) / 2)), '> %.2f' % (max_val)])
    plt.title(label=label)
    plt.show()


def show_results(hospital, test_loader):
    with torch.no_grad():
        temp = 0
        for src, tgt1, tgt2, tgt3, tgt4, tgt5 in test_loader:
            torch.cuda.empty_cache()
            tgts = [tgt1, tgt2, tgt3, tgt4, tgt5]
            hospital_tgts = [tgts[int(view) - 1] for view in hospital.decoders.keys()]
            hospital_views = [int(view) for view in hospital.decoders.keys()]
            edge_idx = torch.Tensor([[i for i in range(src.shape[0])]] * 2).long().to(device)

            hospital.eval()

            outputs = hospital(src, edge_idx)
            show_matrix(antiVectorize(src), label="Source Graph")

            ground_truth = []
            for view, tgt in zip(hospital_views, hospital_tgts):
                if view == 1:
                    tgt /= 100
                if view == 5:
                    tgt /= 10
                print(f"View: {view}")
                graph = antiVectorize(tgt)
                ground_truth.append(graph)
                show_matrix(graph, label=f"Ground Truth Graph View: {view}")

            predicted = []
            for view, output in outputs.items():
                if view == "1":
                    output /= 100
                if view == "5":
                    output /= 10
                print(f"View: {view}")
                graph = antiVectorize(output)
                predicted.append(graph)
                show_matrix(graph, label=f"Predicted Graph View {view}")

            residuals = []
            for i in range(len(predicted)):
                residual = ground_truth[i] - predicted[i]
                residuals.append(residual)
                show_matrix(residual, label=f"Residual_{i}")

            MAEs = []
            for gt, pred in zip(ground_truth, predicted):
                gt = gt.detach().cpu().numpy()
                pred = pred.detach().cpu().numpy()
                MAE = mean_absolute_error(gt, pred)
                print(f"MAE: {MAE}")
                MAEs.append(MAE)

            print(f'Mean MAE is: {np.mean(MAEs)}')
            print(f"total MAE is: {np.sum(MAEs)}")
            break
            # if temp == 1: break
            # temp += 1

