import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):

    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x33 stride=(2, 1) for h & w
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x34
        convRelu(6, True)  # 512x1x33

        self.cnn = cnn

        '''
        self.vision3 = nn.Sequential()
        self.vision3.add_module('conv_vision3', nn.Conv2d(nm[-1], nm[-1], (1, 3), 1, (0, 1)))
        self.vision3.add_module('batchnorm_vision3', nn.BatchNorm2d(512))
        self.vision3.add_module('relu_vision3', nn.ReLU(True))
        self.vision5 = nn.Sequential()
        self.vision5.add_module('conv_vision5', nn.Conv2d(nm[-1], nm[-1], (1, 5), 1, (0, 2)))
        self.vision5.add_module('batchnorm_vision5', nn.BatchNorm2d(512))
        self.vision5.add_module('relu_vision5', nn.ReLU(True))

        self.predictor = nn.Sequential()
        self.predictor.add_module('conv_pred', nn.Conv2d(nm[-1], nclass, 1, 1))
        self.predictor.add_module('batchnorm_pred', nn.BatchNorm2d(nclass))
        self.predictor.add_module('relu_pred', nn.ReLU(True))
        '''
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        #print('input.shape:', input.size())
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        #print('conv.shape:', conv.shape)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        output = self.rnn(conv)
        '''

        #print('rnn.shape:', output.size())
        vision1 = conv
        vision3 = self.vision3(conv)
        vision5 = self.vision5(conv)
        feat = vision1 + vision3 + vision5

        output = self.predictor(feat)
        '''

        return output
