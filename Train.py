#!/usr/bin/env python3

"""

Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.

"""

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import numpy as np
import argparse
from tqdm import tqdm

from PointerNet import PointerNet
from Data_Generator import TSPDataset
import pickle

# from multiprocessing import freeze_support


def tour_length(cities, indices):
    permutation = torch.gather(cities, 1, indices.unsqueeze(-1).expand_as(cities))
    y = torch.cat((permutation, permutation[:, :1]), dim=1)
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))
    return tour_len.sum(1).detach()


parser = argparse.ArgumentParser(description="Pytorch implementation of Pointer-Net")

# Data
parser.add_argument('--train_size', default=1000000, type=int, help='Training data size')
parser.add_argument('--val_size', default=10000, type=int, help='Validation data size')
parser.add_argument('--test_size', default=10000, type=int, help='Test data size')
parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
# Train
parser.add_argument('--nof_epoch', default=50000, type=int, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
# GPU
parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
# TSP
parser.add_argument('--nof_points', type=int, default=5, help='Number of points in TSP')
# Network
parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
parser.add_argument('--hiddens', type=int, default=512, help='Number of hidden units')
parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')

# train files
parser.add_argument('--train_file', type=str)
parser.add_argument('--val_file', type=str)

# save model directory
parser.add_argument('--save_dir', type=str)

# save parameters
parser.add_argument('--save_loss', type=str)
parser.add_argument('--save_lengths', type=str)
parser.add_argument('--save_loss_val', type=str)
parser.add_argument('--save_lengths_val', type=str)
parser.add_argument('--save_accuracy', type=str)

# freeze_support()
params = parser.parse_args()

if params.gpu and torch.cuda.is_available():
    USE_CUDA = True
    print('Using GPU, %i devices.' % torch.cuda.device_count())
    print(torch.backends.cudnn.version())
    print( torch._C._get_cudnn_enabled())
else:
    USE_CUDA = False

model = PointerNet(params.embedding_size,
                   params.hiddens,
                   params.nof_lstms,
                   params.dropout,
                   params.bidir)

with open(params.train_file, 'rb') as train_data:
    train_ = pickle.load(train_data)

dataset = TSPDataset(params.train_size,
                     params.nof_points, train_)

with open(params.val_file, 'rb') as train_data:
    val_ = pickle.load(train_data)

validation_dataset = TSPDataset(params.val_size,
                                params.nof_points, val_)

dataloader = DataLoader(dataset,
                        batch_size=params.batch_size,
                        shuffle=True,
                        num_workers=4)

validation_dataloader = DataLoader(validation_dataset,
                                   batch_size=params.batch_size,
                                   shuffle=True,
                                   num_workers=4)

if USE_CUDA:
    model.cuda()
    net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

CCE = torch.nn.CrossEntropyLoss()
model_optim = optim.Adam(filter(lambda p: p.requires_grad,
                                model.parameters()),
                         lr=params.lr)

losses = []
lengths = []
validation_lengths = []
validation_losses = []
accuracy = 0
accuracies = []
for epoch in range(params.nof_epoch):
    print(epoch)
    model.train()
    batch_loss = []
    batch_lengths = []
    iterator = tqdm(dataloader, unit='Batch', disable=True)
    batch_accuracy = []
    for i_batch, sample_batched in enumerate(iterator):
        iterator.set_description('Batch %i/%i' % (epoch + 1, params.nof_epoch))

        train_batch = Variable(sample_batched['Points'])
        target_batch = Variable(sample_batched['Solution'])

        if USE_CUDA:
            train_batch = train_batch.cuda()
            target_batch = target_batch.cuda()

        o, p = model(train_batch)
        batch_length_ = torch.mean(tour_length(train_batch, p)).item()
        target_length_ = torch.mean(
            tour_length(train_batch, Variable(sample_batched['Solution']).to('cuda'))).item()
        batch_lengths.append(batch_length_)
        o = o.contiguous().view(-1, o.size()[-1])

        target_batch = target_batch.view(-1)

        loss = CCE(o, target_batch)

        losses.append(loss.item())
        batch_loss.append(loss.item())

        model_optim.zero_grad()
        loss.backward()
        model_optim.step()
        batch_accuracy.append(target_length_ / batch_length_)
        iterator.set_postfix(loss='{}'.format(loss.item()), accuracy='{}'.format(target_length_ / batch_length_))

    if (epoch + 1) % 10 == 0:
        torch.save(model, f'{params.save_dir}/snapshot_epoch_{epoch + 1}_loss_{np.average(batch_loss)}_model.pt')
        # validate
        model.eval()
        with torch.no_grad():
            current_val_len = []
            current_target_len = []
            for ev_batch_idx, ev_batch in enumerate(validation_dataloader):
                validation_batch = Variable(ev_batch['Points'])
                validation_target_batch = Variable(ev_batch['Solution'])
                if USE_CUDA:
                    validation_batch = validation_batch.cuda()
                    validation_target_batch = validation_target_batch.cuda()
                ev_o, ev_p = model(validation_batch)
                ev_o = ev_o.contiguous().view(-1, ev_o.size()[-1])
                validation_target_batch = validation_target_batch.view(-1)
                ev_loss = CCE(ev_o, validation_target_batch)
                validation_losses.append(ev_loss.item())
                curr_length = tour_length(validation_batch, ev_p)
                current_val_len.append(torch.mean(curr_length).item())
                current_target_len.append(
                    torch.mean(tour_length(validation_batch, Variable(ev_batch['Solution']).to('cuda'))).item())
                validation_lengths.append(torch.mean(tour_length(validation_batch, ev_p)).item())
        current_accuracy = np.mean(current_target_len) / np.mean(current_val_len)
        if current_accuracy > accuracy:
            torch.save(model,
                       f'{params.save_dir}/best_snapshot_epoch_{epoch + 1}_loss_{np.average(validation_losses[epoch // 10])}_accuracy_{current_accuracy}_model.pt')
            accuracy = current_accuracy
            iterator.set_postfix(acurracy=current_accuracy)

    iterator.set_postfix(loss=np.average(batch_loss))
    lengths.append(np.mean(batch_lengths))
    accuracies.append(batch_accuracy)
with open(params.save_loss, 'wb') as losses_file:
    pickle.dump(losses, losses_file)

with open(params.save_lengths, 'wb') as lengths_file:
    pickle.dump(lengths, lengths_file)

with open(params.save_loss_val, 'wb') as val_losses_file:
    pickle.dump(validation_losses, val_losses_file)

with open(params.save_lengths_val, 'wb') as val_lengths_file:
    pickle.dump(validation_lengths, val_lengths_file)

with open(params.save_accuracy, 'wb') as accuracies_file:
    pickle.dump(accuracies, accuracies_file)
