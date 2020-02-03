
#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import six
import sys
import os
import glob
from PIL import Image
import numpy as np
import pickle

with open('./lmdb/ignore_list.pkl', 'rb') as f:
    ignore_list = pickle.load(f)

def character_prob():
    with open('./stat.txt', 'r') as f:
        stat = f.readlines()
    probs = {}
    for line in stat:
        char, cnt, prob = line.split(' ')
        probs[char] = float(prob[:-1])
    return probs
stat = character_prob()


def license_prob(license):
    '''
        calculate the prob of the whole char sequence
        $license: str
    '''
    prob = stat[license[0]]
    for i in range(1, len(license)):
        if license[i] == license[i-1]:
            prob += stat[license[i]]
        else:
            prob += stat[license[i]]
    return prob


class CCPD(Dataset):
    def __init__(self, root, transform=None, target_transform=None, requires_prob=True, requires_interpret=True):
        self.requires_prob = requires_prob
        self.requires_interpret = requires_interpret
        self.data = []
        root = [root] if type(root) is not list else root
        for item in root:
            print(item)
            assert os.path.isdir(item)
            self.data += glob.glob(os.path.join(item, '*'))

        self.transform = transform
        self.target_transform = target_transform

    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
    alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
    ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

    # 假设文件夹内全部都是图片数据
    def __len__(self):
        return len(self.data)

    @classmethod
    def interpret_plate_name(cls, fn):
        chars = fn.rsplit('/', 1)[-1].split('-')[-3].split('_')
        plate = []
        for i, char in enumerate(chars):
            char_idx = int(char)
            if i == 0:
                plate.append(CCPD.provinces[char_idx])
            elif i == 1:
                plate.append(CCPD.alphabets[char_idx])
            else:
                plate.append(CCPD.ads[char_idx])
        return ''.join(plate)

    def __getitem__(self, index):
        assert index < len(self)
        img = Image.open(self.data[index]).convert('L')
        try:
            label = CCPD.interpret_plate_name(self.data[index])
            assert len(label)
        except:
            label = self.data[index].rsplit('/', 1)[-1].split('.')[0]
        if self.requires_prob:
            prob = license_prob(label)
        else:
            prob = 1.0
        alpha = 1/prob

        if self.transform is not None:
            img = self.transform(img)
            
        return (img, label, alpha)
        

class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
                raw_img = img
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = txn.get(label_key.encode()).decode()
            
            '''
            if index % 10 == 0 and index < 100:
                raw_img.save('./%s.jpg'%label)
                print('SAVE a sample dataset image')
            '''

            if self.target_transform is not None:
                label = self.target_transform(label)

            # filter item if label is in ignore_list
            if label in ignore_list:
                return None

        return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.range(0, self.batch_size - 1)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.range(0, tail - 1)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1, requires_prob=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio
        self.requires_prob = requires_prob

    def __call__(self, batch):
        # handle None items in a batch
        batch = [item for item in batch if item is not None]
        if self.requires_prob:
            images, labels, probs = zip(*batch)
        else:
            images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return (images, labels, probs) if self.requires_prob else (images, labels)


if __name__ == '__main__':
    dataset = CCPD('./data')
    print(CCPD.interpret_plate_name('normal_plates/1121120-33_50-210&367_509&680-465&511_210&680_254&536_509&367-0_0_2_13_31_26_25-74-254.jpg'))