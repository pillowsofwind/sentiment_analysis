import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
import numpy as np
import random
import spacy
import time
import argparse
import matplotlib.pyplot as plt

SEED = 5029  # 为复现实验效果，固定random seed
MAX_VOCAB_SIZE = 30000  # 词表大小
BATCH_SIZE = 64  # batch大小
EMBEDDING_SIZE = 100  # 将词embedding为向量的维数
OUTPUT_SIZE = 8
HIDDEN_SIZE = 128  # RNN的隐含层大小
NUM_FILTERS = 100  # CNN通道数
FILTER_SIZE = [2,3,5]  # CNN通道大小
DROPOUT = 0.3  # 弃用神经元比例

TRAIN_PATH = "sinanews.train"
TEST_PATH = "sinanews.test"

tokenize = lambda x: x.split()
TEXT = data.Field(sequential=True, tokenize=tokenize, lower=True)
LABEL = data.Field(sequential=False, use_vocab=False, dtype=torch.float)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

tag_decoding = {0: '感动', 1: '同情', 2: '无聊', 3: '愤怒', 4: '搞笑', 5: '难过', 6: '新奇', 7: '温馨'}


def load_data(path):
    """将材料的label与text进行分离，得到两个list"""
    label_list = []
    text_list = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data = line.strip().split('\t')
            data[1] = data[1].strip().split()
            label = [0 for i in range(8)]
            total = 0
            for i in range(0, 8):
                label[i] = float(data[1][1 + i].split(':')[1])
                total += label[i]
            for i in range(len(label)):
                label[i] /= total
            label_list.append(label)
            text_list.append(data[2].strip().split())
        return label_list, text_list


def build_dataset(label, text, label_field, text_field):
    """构建dataset"""
    examples = []
    fields = [("id", None), ("text", text_field), ("label", label_field)]
    for i, itr in enumerate(label):
        examples.append(data.Example.fromlist([None, text[i], label[i]], fields))
    dataset = data.Dataset(examples, fields)
    return dataset


class WordAVGModel(nn.Module):
    """平均池化模型"""

    def __init__(self, vocab_size, embedding_size, output_size, idx):
        super(WordAVGModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=idx)
        self.linear = nn.Linear(embedding_size, output_size)

    def forward(self, text):
        embedded = self.embed(text)  # 序列长度*batch大小*embedding长度
        embedded = embedded.permute(1, 0, 2)  # batch大小*序列长度*embedding长度

        # 池化
        pooled = F.avg_pool2d(embedded, (embedded.shape[1], 1)).squeeze(1)

        return self.linear(pooled)


class RNNModel(nn.Module):
    """RNN-biLSTM模型"""

    def __init__(self, vocab_size, embedding_size, hidden_size, output_size, dropout, idx):
        super(RNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=idx)
        self.lstm = nn.LSTM(embedding_size, hidden_size, bidirectional=True, num_layers=1)
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embed(text)
        embedded = self.dropout(embedded)  # [sent len, batch size, emb dim]
        output, (hidden, cell) = self.lstm(embedded)

        hidden = torch.cat((hidden[-1], hidden[-2]), dim=1)
        hidden = self.dropout(hidden.squeeze())  # [batch size, hid dim * num directions]

        return self.linear(hidden)


class CNNModel(nn.Module):
    """CNN模型"""

    def __init__(self, vocab_size, idx, embedding_size, output_size, num_filters, filter_size, dropout):
        super(CNNModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, padding_idx=idx)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(f, embedding_size))
            for f in filter_size
        ])
        self.linear = nn.Linear(len(filter_size) * num_filters, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        text = text.permute(1, 0)  # batch大小*序列长度
        embedded = self.embed(text)  # batch大小*序列长度*embedding长度
        embedded = embedded.unsqueeze(1)  # 增加一维：batch大小*1（通道数）*序列长度*embedding长度

        # 卷积层
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]  # 增加一维：batch大小*通道数*序列长度-通道大小+1

        # 池化层
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # batch大小*通道数
        pooled = self.dropout(torch.cat(pooled, dim=1))

        return self.linear(pooled)


def tag_simplify(tag):
    """转化为单标签"""
    temp = [0 for i in range(len(tag))]
    max = float('-inf')
    id = 0
    for i in range(len(tag)):
        if tag[i] > max:
            max = tag[i]
            id = i
    temp[id] = 1
    return temp


def accuracy(pred, truth):
    """计算准确度"""
    acc = 0.
    for i in range(len(pred)):
        temp = accuracy_score(tag_simplify(truth.cpu().detach().numpy()[i]),
                              tag_simplify(pred.cpu().detach().numpy()[i]))
        if temp == 1:
            acc += 1
    return acc / len(pred)


def F_score(pred, truth):
    """计算F——score"""
    f_score = 0.
    for i in range(len(pred)):
        f_score += f1_score(tag_simplify(truth.cpu().detach().numpy()[i]),
                            tag_simplify(pred.cpu().detach().numpy()[i]),
                            average='macro')
    return f_score / len(pred)


def corr(pred, truth):
    """计算相关系数"""
    corr = 0.
    for i in range(len(pred)):
        pred_ = pred[i].cpu().detach().numpy()
        truth_ = truth[i].cpu().detach().numpy()
        c = pearsonr(pred_, truth_)
        c = list(c)
        corr += c[0]
    return corr / len(pred)


def train(model, iter, optimizer, crit):
    """训练"""
    epoch_loss, epoch_acc = 0., 0.
    model.train()

    for batch in iter:
        pred = model(batch.text)
        loss = crit(pred, batch.label)
        acc = accuracy(pred, batch.label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(iter), epoch_acc / len(iter)


def evaluate(model, iter, crit):
    """评估"""
    epoch_loss, epoch_acc, epoch_F_score, epoch_corr = 0., 0., 0., 0.
    model.eval()

    with torch.no_grad():
        for batch in iter:
            pred = model(batch.text).squeeze(1)
            loss = crit(pred, batch.label)
            acc = accuracy(pred, batch.label)
            F_score_ = F_score(pred, batch.label)
            corr_ = corr(pred, batch.label)

            epoch_loss += loss.item()
            epoch_acc += acc
            epoch_F_score += F_score_
            epoch_corr += corr_
    return epoch_loss / len(iter), epoch_acc / len(iter), epoch_F_score / len(iter), epoch_corr / len(iter)


def predict(text, model):
    """预测，返回标签向量和情感"""
    nlp = spacy.load("en")
    text_list = [tok.text for tok in nlp.tokenizer(text)]
    indexed = [TEXT.vocab.stoi[t] for t in text_list]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(1)
    pred = torch.sigmoid(model(tensor)).squeeze()
    max_tag = 0.
    ans = 0
    for i in range(8):
        if pred.data[i] > max_tag:
            max_tag = pred.data[i]
            ans = i
    return pred.data, tag_decoding[ans]


def time_convert(start, end):
    elapsed_time = end - start
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def parse():
    parser = argparse.ArgumentParser(description="基于NLP的文本情感分类")
    parser.add_argument('--model', type=int, default=0,
                        help="请以整型数选择以下模型:\n0:Word Average\n1:RNN-biLSTM\n2:CNN")
    parser.add_argument('--train', help="训练模型", action='store_true')
    parser.add_argument('--test', help="测试模型", action='store_true')
    parser.add_argument('-i', help="试用", action='store_true')
    parser.add_argument('-e', type=int, default=10, help="训练总回合数,默认为10")

    args = parser.parse_args()
    print(args)

    return args


def preprocess():
    """语料预处理"""
    print("加载训练材料...")
    train_data_label, train_data_text = load_data(TRAIN_PATH)
    test_data_label, test_data_text = load_data(TEST_PATH)

    print("从材料中构建dataset...")
    train_data = build_dataset(train_data_label, train_data_text, LABEL, TEXT)
    test_data = build_dataset(test_data_label, test_data_text, LABEL, TEXT)
    train_data, valid_data = train_data.split(random_state=random.seed(SEED))  # 分离出一部分验证集

    print("从dataset构建词表...")
    TEXT.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
    LABEL.build_vocab(train_data)
    print("词表大小为", len(TEXT.vocab))

    return data.BucketIterator.splits((train_data, valid_data, test_data),
                                      batch_size=BATCH_SIZE,
                                      sort_within_batch=True,
                                      sort_key=lambda x: len(x.text), )


def main():
    args = parse()

    if args.model < 0 or args.model > 2:
        print("\n似乎是不存在的模型！")
        return

    train_iter, valid_iter, test_iter = preprocess()

    VOCAB_SIZE = len(TEXT.vocab)
    PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
    UNK_IDX = TEXT.vocab.stoi[TEXT.unk_token]

    if args.model == 0:
        model = WordAVGModel(vocab_size=VOCAB_SIZE,
                             embedding_size=EMBEDDING_SIZE,
                             output_size=OUTPUT_SIZE,
                             idx=PAD_IDX)
        model_path = "WordAvg_model.pth"

    if args.model == 1:
        model = RNNModel(vocab_size=VOCAB_SIZE,
                         embedding_size=EMBEDDING_SIZE,
                         output_size=OUTPUT_SIZE,
                         idx=PAD_IDX,
                         hidden_size=HIDDEN_SIZE,
                         dropout=DROPOUT)
        model_path = "RNN_model.pth"

    if args.model == 2:
        model = CNNModel(vocab_size=VOCAB_SIZE,
                         embedding_size=EMBEDDING_SIZE,
                         output_size=OUTPUT_SIZE,
                         idx=PAD_IDX,
                         num_filters=NUM_FILTERS,
                         filter_size=FILTER_SIZE,
                         dropout=DROPOUT)
        model_path = "CNN_model.pth"

    model.embed.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_SIZE)
    model.embed.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_SIZE)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()

    if args.train:
        """训练主体"""
        start = time.time()
        print("\n开始训练", ['WordAvg', 'RNN-biLSTM', 'CNN'][args.model], "模型...")
        train_loss_list, valid_loss_list = [], []
        train_acc_list, valid_acc_list = [], []
        epoch_list = []

        best_valid_loss = float('inf')
        for epoch in range(args.e):
            train_loss, train_acc = train(model, train_iter, optimizer, criterion)
            valid_loss, valid_acc, valid_F_score, valid_corr = evaluate(model, valid_iter, criterion)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), model_path)

            print("训练Epoch", epoch, "\n测试集损失", train_loss, "测试集准确率: ", train_acc)
            print("验证集损失", valid_loss, "验证集准确率: ", valid_acc, "验证集F-score: ", valid_F_score, "验证集相关系数:",
                  valid_corr)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)
            epoch_list.append(epoch)

        end = time.time()
        mins, secs = time_convert(start, end)
        print("\n模型训练完成，已保存至", model_path)
        print("本模型训练了共", sum(p.numel() for p in model.parameters() if p.requires_grad), "个参数")
        print("本模型训练了共", mins, "分", secs, "秒")

        # 绘制图像
        plt.title(['WordAvg', 'RNN-biLSTM', 'CNN'][args.model])
        plt.xlabel('Number of epoch(es)')
        plt.plot(epoch_list, train_loss_list, "orange")
        plt.plot(epoch_list, valid_loss_list, "blue")
        plt.plot(epoch_list, train_acc_list, "red")
        plt.plot(epoch_list, valid_acc_list, "green")
        plt.legend(["Training loss", "Validation loss", "Training accuracy", "Validation accuracy"])
        plt.show()

    if args.test:
        if args.train is False:
            print("\n尚未对模型训练，使用默认路径的模型")

        """模型测试与评价"""
        print("\n开始测试", ['WordAvg', 'RNN', 'CNN'][args.model], "模型...")
        model.load_state_dict(torch.load(model_path))
        loss, acc, F_score, corr = evaluate(model, test_iter, criterion)

        print("\n模型评价\n准确率:", acc, "F-score:", F_score, "相关系数:", corr)

    if args.i:
        if args.train is False:
            print("\n尚未对模型训练，使用默认路径的模型")

        """接受控制台输入文本进行分析"""
        text = input("\n请输入一句话:")
        tag, pred = predict(text, model)

        print("\n情感分类为:", pred, "\n情感标签为:", tag)


if __name__ == '__main__':
    main()
