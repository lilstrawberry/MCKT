import pandas as pd
import torch
import torch.nn as nn

from run import run_epoch
from model import MCKT
import glo

mp2path = {
    'static11': {
        'ques_skill_path': 'data/static11/ques_skill.csv',
        'train_path': 'data/static11/train_question.txt',
        'test_path': 'data/static11/test_question.txt',
        'pro_unique_skill_neibor': 'data/static11/static11_unique_skill_Q-Q',
        'skill_max': 106
    }
}

procl2data = {
    'assist09': 1,
    'assist17': 0.001,
    'static11': 0.1,
    'ednet': 0.01
}

actcl2data = {
    'assist09': 1,
    'assist17': 0.001,
    'static11': 0.1,
    'ednet': 0.01
}

statecl2data = {
    'assist09': 0.0001,
    'assist17': 0.1,
    'static11': 0.1,
    'ednet': 0.1
}

dataset = 'static11'

if __name__ == '__main__':

    glo._init()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ques_skill_path = mp2path[dataset]['ques_skill_path']

    train_path = mp2path[dataset]['train_path']
    if "valid_path" in mp2path[dataset]:
        valid_path = mp2path[dataset]['valid_path']
    else:
        valid_path = mp2path[dataset]['test_path']
    test_path = mp2path[dataset]['test_path']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pro_max = 1 + max(pd.read_csv(ques_skill_path).values[:, 0])
    skill_max = mp2path[dataset]['skill_max']
    pos_matrix = torch.load(mp2path[dataset]['pro_unique_skill_neibor']).to(device).to_dense()

    use_epoch = 200
    p = 0.1
    d = 128
    learning_rate = 0.002
    epochs = 70
    batch_size = 80
    min_seq = 3
    max_seq = 200
    grad_clip = 15.0
    patience = 10
    sim_yuzhi = 0.8
    cl_use_batch = 10000

    avg_auc = 0
    avg_acc = 0

    glo.set_value('cl_use_batch', cl_use_batch)
    glo.set_value('sim', sim_yuzhi)
    glo.set_value('pro_cl', procl2data[dataset])
    glo.set_value('inter_cl', actcl2data[dataset])
    glo.set_value('state_cl', statecl2data[dataset])
    glo.set_value('pos_matrix', pos_matrix)

    for now_step in range(5):

        best_acc = 0
        best_auc = 0
        state = {'auc': 0, 'acc': 0, 'loss': 0}

        model = MCKT(pro_max, skill_max, d, p)
        model = model.to(device)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        # , weight_decay=1e-5
        one_p = 0

        for epoch in range(use_epoch):

            one_p += 1

            train_loss, train_acc, train_auc = run_epoch(pro_max, train_path, batch_size,
                                                         True, min_seq, max_seq, model, optimizer, criterion, device,
                                                         grad_clip)
            print(
                f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, train_auc: {train_auc:.4f}')

            valid_loss, valid_acc, valid_auc = run_epoch(pro_max, valid_path, batch_size, False,
                                                      min_seq, max_seq, model, optimizer, criterion, device, grad_clip)

            print(f'epoch: {epoch}, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}, valid_auc: {valid_auc:.4f}')

            if valid_auc > best_auc:
                one_p = 0
                best_auc = valid_auc
                best_acc = valid_acc
                torch.save(model.state_dict(), f"./MCKT_{dataset}_{now_step}_model.pkl")
                state['auc'] = valid_auc
                state['acc'] = valid_acc
                state['loss'] = valid_loss
                torch.save(state, f'./MCKT_{dataset}_{now_step}_state.ckpt')

            if one_p >= patience:
                break

        model.load_state_dict(torch.load(f'./MCKT_{dataset}_{now_step}_model.pkl'))
        model.eval()

        test_loss, test_acc, test_auc = run_epoch(pro_max, test_path, batch_size, False,
                                                     min_seq, max_seq, model, optimizer, criterion, device, grad_clip)

        print(f'*******************************************************************************')
        print(f'test_acc: {test_acc:.4f}, test_auc: {test_auc:.4f}')
        print(f'*******************************************************************************')

        avg_auc += test_acc
        avg_acc += test_auc

    avg_auc = avg_auc / 5
    avg_acc = avg_acc / 5
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
    print(f'*******************************************************************************')
