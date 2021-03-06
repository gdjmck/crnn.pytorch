from __future__ import print_function
from __future__ import division

import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from utils import plot_preds
import numpy as np
import sys
sys.path.append('/home/chengk/anaconda3/envs/py36/lib/python3.6/site-packages/warpctc_pytorch-0.1-py3.6-linux-x86_64.egg/warpctc_pytorch')
from torch.nn import CTCLoss
import os
import utils
import dataset
import glob
import plot

import models.crnn as crnn

parser = argparse.ArgumentParser()
parser.add_argument('--trainRoot', required=True, help='path to dataset')
parser.add_argument('--valRoot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=100, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
# TODO(meijieru): epoch -> iter
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--pretrained', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--strict', action='store_true', help='load pretrained model in strict mode')
parser.add_argument('--alphabet', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz')
parser.add_argument('--expr_dir', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--n_test_disp', type=int, default=10, help='Number of samples to display when test')
parser.add_argument('--valInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--saveInterval', type=int, default=500, help='Interval to be displayed')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for Critic, not used by adadealta')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--adadelta', action='store_true', help='Whether to use adadelta (default is rmsprop)')
parser.add_argument('--keep_ratio', action='store_true', help='whether to keep ratio for image resize')
parser.add_argument('--manualSeed', type=int, default=1234, help='reproduce experiemnt')
parser.add_argument('--random_sample', action='store_true', help='whether to sample the dataset with random sampler')
parser.add_argument('--test', action='store_true', help='test mode and skip training')
parser.add_argument('--no_need_interpret', action='store_true', help='switch to turn off filename interpretation')
parser.add_argument('--focal', action='store_true', help='switch to turn on focal loss')
parser.add_argument('--summary', type=str, default='./runs/lmdb', help='folder to save tensorboard file')
opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.expr_dir):
    os.makedirs(opt.expr_dir)

random.seed(opt.manualSeed)
np.random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True
focal_alpha = False

val_wrong = None

writer = SummaryWriter(opt.summary)
global_step = 0

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if len(glob.glob(os.path.join(opt.trainRoot, '*.mdb'))):
    train_dataset = dataset.lmdbDataset(root=opt.trainRoot)
    print('Training with lmdb dataset')
else:
    trainRoot = opt.trainRoot
    if ',' in trainRoot:
        trainRoot = trainRoot.split(',')
    train_dataset = dataset.CCPD(trainRoot, requires_interpret=not opt.no_need_interpret)
    print('Training with custom dataset')
    focal_alpha = True
assert train_dataset
if not opt.random_sample:
    sampler = dataset.randomSequentialSampler(train_dataset, opt.batchSize)
else:
    sampler = None
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    sampler=sampler,
    num_workers=int(opt.workers),
    collate_fn=dataset.alignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio=opt.keep_ratio, requires_prob=focal_alpha))
if len(glob.glob(os.path.join(opt.valRoot, '*.mdb'))):
    test_dataset = dataset.lmdbDataset(
        root=opt.valRoot, transform=dataset.resizeNormalize((100, 32)))
    print('Testing with lmdb dataset')
else:
    test_dataset = dataset.CCPD(opt.valRoot, transform=dataset.resizeNormalize((100, 32)))
    print('Testing with custom dataset')

alphabet = utils.generate_alphabet()
print('Alphabet:', alphabet, '\n', len(alphabet))
nclass = len(alphabet) + 1
nc = 1

converter = utils.strLabelConverter(alphabet, ignore_case=False)
criterion = CTCLoss(reduction='none')


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


image = torch.FloatTensor(opt.batchSize, 3, opt.imgH, opt.imgH)
text = torch.IntTensor(opt.batchSize * 5)
length = torch.IntTensor(opt.batchSize)
probs = torch.FloatTensor(opt.batchSize)

crnn = crnn.CRNN(opt.imgH, nc, nclass, opt.nh)
crnn.apply(weights_init)
if opt.cuda:
    crnn.cuda()
    #crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.ngpu))
    image = image.cuda()
    probs = probs.cuda()
    text = text.cuda()
    criterion = criterion.cuda()
if opt.pretrained != '':
    print('loading pretrained model from %s' % opt.pretrained)
    try:
        ckpt = torch.load(opt.pretrained)
        if 'module' in next(iter(ckpt.keys())):
            print('DataParallel model.')
            crnn = nn.DataParallel(crnn)
        crnn.load_state_dict(ckpt)
    except:
        if not opt.strict:
            print('\tStrict load failed, use unstricted loading.')
            model_params = crnn.get_params_name()
            print(model_params)
            for par, val in crnn.named_parameters():
                if par in ckpt.keys() and val.size() != ckpt[par].size():
                    print('par size mismatch:', val.size(), ckpt[par].size())
                    del ckpt[par]
            crnn.load_state_dict(ckpt, strict=False)
        else:
            print('Failed to load model')
            sys.exit(0)
print(crnn)
'''
for name, w in crnn.named_parameters():
    print(name, w.size())
'''


image = Variable(image)
text = Variable(text)
length = Variable(length)
probs = Variable(probs)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.adam:
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr,
                           betas=(opt.beta1, 0.999), amsgrad=True)
elif opt.adadelta:
    optimizer = optim.Adadelta(crnn.parameters())
else:
    optimizer = optim.RMSprop(crnn.parameters(), lr=opt.lr)


def val(net, dataset, criterion, max_iter=100):
    print('Start val')

    for p in crnn.parameters():
        p.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batchSize, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    i = 0
    n_correct = 0
    loss_avg = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for i in range(max_iter):
        try:
            data = val_iter.next()
        except:
            continue
        i += 1
        try:
            cpu_images, cpu_texts = data
        except ValueError:
            cpu_images, cpu_texts, _ = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts)
        global text
        utils.loadData(text, t)
        utils.loadData(length, l)
        text = text.view((batch_size, -1)).cuda()

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost = criterion(F.log_softmax(preds, dim=-1), text, preds_size, length).sum() / batch_size
        loss_avg.add(cost)

        _, preds = preds.max(2)
        #preds = preds.squeeze(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target:
                n_correct += 1
            elif val_wrong is not None:
                val_wrong.append(target+'_'+pred)
        writer.add_scalars('Test', {'loss': cost.item(), 'acc': (np.array(sim_preds) == np.array(cpu_texts)).mean()}, 
                            global_step=global_step*max_iter+i)


    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batchSize)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))


def trainBatch(net, criterion, optimizer):
    data = train_iter.next()
    if focal_alpha:
        cpu_images, cpu_texts, alpha = data
        alpha = torch.FloatTensor(list(alpha))
        utils.loadData(probs, alpha)
        assert not probs.requires_grad
    else:
        cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    assert batch_size > 0
    utils.loadData(image, cpu_images)
    t, l = converter.encode(cpu_texts)
    #print(cpu_texts, 'converts to', t, t.size())
    global text
    utils.loadData(text, t)
    utils.loadData(length, l)
    text = text.view((batch_size, -1))
    text = text.cuda()

    preds = F.log_softmax(crnn(image), dim=-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    _, preds_str = preds.max(2)
    preds_str = preds_str.transpose(1, 0).contiguous().view(-1)
    preds_str = converter.decode(preds_str.data, preds_size.data, raw=False)
    acc = (np.array(preds_str) == np.array(cpu_texts)).mean()

    if display_flag:
        writer.add_figure('Train predictions vs. actuals',
                            plot_preds(cpu_images, preds_str, cpu_texts),
                            global_step=global_step)
        writer.add_figure('Gradient', plot.plot_grad_flow_v2(crnn.named_parameters()),
                            global_step=global_step)
    #print('preds:', preds.size())
    preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
    #print('preds_size:', preds_size, '\tlength:', length)
    #print('preds.size():', preds.size(), 'text.size()', text.size())
    cost = criterion(preds, text, preds_size, length)
    if opt.focal:
        cost = cost * probs
    cost = cost.sum() / batch_size
    writer.add_scalars('training', {'loss': cost.item(), 'acc': acc}, global_step)
    writer.add_scalars('lr', plot.get_lr(optimizer), global_step)
    crnn.zero_grad()
    cost.backward()
    torch.nn.utils.clip_grad_value_(crnn.parameters(), 1)
    optimizer.step()
    return cost

if opt.test:
    val_wrong = []
    val(crnn, test_dataset, criterion, max_iter=len(test_dataset))
    with open('./val_wrong.txt', 'w') as f:
        for item in val_wrong:
            f.write(item+'\n')
    sys.exit(0)

for epoch in range(opt.nepoch):
    train_iter = iter(train_loader)
    i = 0
    while i < len(train_loader):
        global_step += 1
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        if i % opt.displayInterval == 0:
            display_flag = True
        else:
            display_flag = False

        cost = trainBatch(crnn, criterion, optimizer)
        loss_avg.add(cost)
        i += 1

        if i % opt.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' %
                  (epoch, opt.nepoch, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.valInterval == 0:
            val(crnn, test_dataset, criterion)

        # do checkpointing
        if i % opt.saveInterval == 0:
            torch.save(
                crnn.state_dict(), '{0}/netCRNN_{1}_{2}.pth'.format(opt.expr_dir, epoch, i))
