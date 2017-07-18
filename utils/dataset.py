import torch
import math
from torch.autograd import Variable
from utils.vocab import Constants


class Dataset(object):
    def __init__(self, dialog_data, batch_size, cuda=True, volatile=False, ret_limit=None):
        self.dialog_data = dialog_data
        if ret_limit:
            self.dialog_data = dialog_data[:ret_limit]

        self.batch_size = batch_size
        self.cuda_flag = cuda
        self.volatile = volatile
        self.num_samples = len(self.dialog_data)
        self.num_batches = math.ceil(len(self.dialog_data) / batch_size)

    def __len__(self):
        return self.num_batches

    @staticmethod
    def batch_identity2(data):
        lengths = [x.size(0) for x in data]

        indices = range(len(data))
        batch = zip(indices, data)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        indices, data = zip(*batch)

        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            out[i].narrow(0, 0, data_length).copy_(data[i])

        indices = torch.LongTensor(indices)
        return out, lengths, indices

    @staticmethod
    def batch_identity(data):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(Constants.PAD)
        for i in range(len(data)):
            out[i].narrow(0, 0, lengths[i]).copy_(data[i])

        return out

    def shuffle(self):
        self.dialog_data = [self.dialog_data[i] for i in torch.randperm(self.num_samples)]

    def wrap(self, b):
        """
        stack tensor list and return var
        :param b:
        :return:
        """
        if b is None:
            return b
        b = torch.stack(b, 0)
        b = b.t().contiguous()
        if self.cuda_flag:
            b = b.cuda()
        b = Variable(b, volatile=self.volatile)
        return b

    def __getitem__(self, index):
        assert index < self.num_batches, "%d > %d" % (index, self.num_batches)

        batch_samples = self.dialog_data[index * self.batch_size:(index + 1) * self.batch_size]

        turns = [len(sample) for sample in batch_samples]
        max_turn = max(turns)

        batch_size = len(batch_samples)
        turn_mask = [[] for _ in range(max_turn)]
        turn_batches = [[] for _ in range(max_turn)]

        for sample in batch_samples:
            j = 0
            # align left
            for sent in sample:
                turn_mask[j].append(1.0)
                turn_batches[j].append(sent)
                j += 1

            while j < max_turn:
                turn_mask[j].append(0.0)
                turn_batches[j].append(torch.LongTensor([Constants.PAD]))
                j += 1

        turn_batch_vars = []
        for turn_batch in turn_batches:
            assert len(turn_batch) == batch_size

            data = self.batch_identity(turn_batch)
            data = self.wrap(data)
            turn_batch_vars += [data]

        turn_mask = Variable(torch.FloatTensor(turn_mask), volatile=self.volatile)
        if self.cuda_flag:
            turn_mask = turn_mask.cuda()

        return turn_batch_vars, turn_mask


def test_dataset():
    dep_data = torch.load("../data/douban/train/train.dep.pt")
    data = dep_data["data"]
    print(len(data))
    train_set = Dataset(data, batch_size=5, cuda=False)
    for i in range(len(train_set)):
        batch_turns, turn_mask = train_set[i]
        print(turn_mask)
        for batch_turn in batch_turns:
            print(batch_turn)
        break
        pass


if __name__ == '__main__':
    test_dataset()
