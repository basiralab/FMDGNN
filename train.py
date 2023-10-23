from model import *
from fed import *
from config import *
from evaluate import *


def train(hospital_list, local_datasets, local_tests,
          global_test, res=res, aggr=aggr):
    if not res:
        encoder = GCNencoder(nfeat=595, nhid=32)
        decoders = [GCNdecoder(nhid=32, nfeat=595) for _ in range(5)]
    else:
        encoder = ResGCNencoder(nfeat=595, nhid=32)
        decoders = [ResGCNdecoder(nhid=32, nfeat=595) for _ in range(5)]

    global_model = Hospital(encoder, decoders, views=[1, 2, 3, 4, 5]).to(device)

    optimizer_list = []
    for hospital in hospital_list:
        optimizer_list.append(torch.optim.Adam(hospital.parameters(), lr=0.001, betas=(0.5, 0.999)))

    client_list = []
    for i in range(len(hospital_list)):
        client = Client(hospital_list[i], optimizer_list[i],
                        local_datasets[i], epochs_per_round,
                        batch_size, device, name=f"Hospital {i + 1}")
        client_list.append(client)

    server = Server(client_list, global_model, aggr)
    losses = server.federate(fed_rounds)

    test_loader_list = []
    for i in range(len(hospital_list)):
        test_loader_list.append(DataLoader(local_tests[i]))
    global_test_loader = DataLoader(global_test)

    eval_loss_list = []
    view_loss_list = []

    for i in range(len(client_list)):
        eval_loss, view_loss = evaluate(client_list[i].model, test_loader_list[i], device)
        eval_loss_list.append(eval_loss)
        view_loss_list.append(view_loss)

    global_eval_loss, global_view_loss = evaluate(server.global_model, global_test_loader, device)

    local_eval = [eval_loss_list, view_loss_list]
    global_eval = [global_eval_loss, global_view_loss]

    return [client_list, server, losses, local_eval, global_eval]


def train_baseline(model, train_loader, lr, num_epochs, device):
    torch.cuda.empty_cache()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    train_loss_log = []

    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        for src, tgt1, tgt2, tgt3, tgt4, tgt5 in train_loader:
            torch.cuda.empty_cache()
            tgts = [tgt1, tgt2, tgt3, tgt4, tgt5]
            hospital_tgts = [tgts[int(view) - 1] for view in model.decoders.keys()]
            hospital_views = [int(view) for view in model.decoders.keys()]
            edge_idx = torch.Tensor([[i for i in range(src.shape[0])]] * 2).long().to(device)

            model.train()
            optimizer.zero_grad()
            outputs = model(src, edge_idx)

            loss = sum([F.l1_loss(outputs[str(view)], tgt) for view, tgt in zip(hospital_views, hospital_tgts)])

            train_loss += loss.item()
            loss.backward()
            optimizer.step()

        train_loss_log.append(train_loss / len(train_loader))
        print(f"Epoch: {epoch}  Loss: {train_loss / len(train_loader)}")
    print("Finish Training")

    return train_loss_log