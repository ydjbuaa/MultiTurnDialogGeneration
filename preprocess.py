import argparse

from utils.tree import *
from utils.vocab import *

parser = argparse.ArgumentParser(description='preprocess.py')

# **Preprocess Options**

parser.add_argument('-config', help="Read options from this file")

parser.add_argument('-train', default="./data/douban/train/train.txt",
                    help="Path to the training source data")

parser.add_argument('-train_tree', default="./data/douban/train/train.parents",
                    help="Path to the training target data")

parser.add_argument('-valid', default="./data/douban/valid/valid.txt",
                    help="Path to the validation source data")

parser.add_argument('-valid_tree', default="./data/douban/valid/valid.parents",
                    help="Path to the validation target data")

parser.add_argument('-save_data', default="./data/douban/",
                    help="Output file for the prepared data")

parser.add_argument('-vocab_size', type=int, default=30000,
                    help="Size of the vocabulary")

parser.add_argument('-vocab_path',
                    help="Path to an existing source vocabulary")

parser.add_argument('-max_length', type=int, default=100,
                    help="Maximum sequence length")

parser.add_argument('-max_turn', type=int, default=10,
                    help="Maximum turns")

parser.add_argument('-shuffle', type=int, default=0,
                    help="Shuffle data")
parser.add_argument('-seed', type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)


def make_vocabulary(filename, size):
    vocab = Vocabulary([Constants.PAD_WORD, Constants.UNK_WORD,
                        Constants.BOS_WORD, Constants.EOS_WORD],
                       lower=opt.lower)

    with open(filename, 'r', encoding='utf-8') as fr:
        for sent in fr:
            for word in sent.split():
                vocab.add(word)

    original_size = vocab.size
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size, original_size))
    return vocab


def init_vocabulary(name, data_file, vocab_file, vocab_size):
    vocab = None
    if vocab_file is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocab_file + '\'...')
        vocab = Vocabulary()
        vocab.load_file(vocab_file)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        vocab = make_vocabulary(data_file, vocab_size)

    print()
    return vocab


def save_vocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.write_file(file)


def make_data(data_file, tree_file, vocab):
    data, syn_indices, sizes = [], [], []

    count, ignored = 0, 0

    print('Processing %s & %s ...' % (data_file, tree_file))
    d_fr = open(data_file, 'r', encoding='utf-8')
    t_fr = open(tree_file, 'r', encoding='utf-8')

    while True:
        sent_seq = d_fr.readline()
        tree_seq = t_fr.readline()

        # normal end of file
        if sent_seq == "" and tree_seq == "":
            break

        # source or target does not have same number of lines
        if sent_seq == "" or tree_seq == "":
            print('WARNING: src and tgt do not have the same # of sentences')
            break
        sent_seq = sent_seq.strip()
        tree_seq = tree_seq.strip()

        # source and/or target are empty
        if sent_seq == "" or tree_seq == "":
            print('WARNING: ignoring an empty line (' + str(count + 1) + ')')
            continue

        sent_seqs = sent_seq.split('\t')
        tree_seqs = tree_seq.split('\t')

        if len(sent_seqs) != len(tree_seqs):
            print('WARNING: sentences do not have the same # of the trees')
            continue

        if len(tree_seqs) > opt.max_turn:
            print("WARNING: truncate the {} samples".format(count))
            sent_seqs = sent_seqs[:opt.max_turn]
            tree_seqs = tree_seqs[:opt.max_turn]

        if len(tree_seqs) < 2:
            print("WARNING: Turn < {} samples".format(count))

        flag = False
        sample = []
        syn_idx = []
        max_length = 0

        for sent_line, tree_line in zip(sent_seqs, tree_seqs):
            words = sent_line.split()
            if len(words) <= 0 or len(words) > opt.max_length:
                flag = True
                break

            if len(words) > max_length:
                max_length = len(words)

            sent = vocab.convert2idx(words,
                                     Constants.UNK_WORD,
                                     Constants.BOS_WORD,
                                     Constants.EOS_WORD)

            tree = read_tree(tree_line)

            ces_indices = tree.child_enriched_structure()
            corr_ces_indices = get_corresponding_order(ces_indices)

            hes_indices = tree.head_enriched_structure()
            corr_hes_indices = get_corresponding_order(hes_indices)

            sample += [sent]
            syn_idx += [[torch.LongTensor(ces_indices),
                         torch.LongTensor(hes_indices),
                         torch.LongTensor(corr_ces_indices),
                         torch.LongTensor(corr_hes_indices)]]

        if flag:
            ignored += 1
            continue

        data += [sample]
        syn_indices += [syn_idx]
        sizes += [len(sample) * max_length]

        count += 1
        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)
        if count == 10000:
            break
    d_fr.close()
    t_fr.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(data))
        data = [data[idx] for idx in perm]
        syn_indices = [syn_indices[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]

    print('... sorting sentences by size')
    _, perm = torch.sort(torch.IntTensor(sizes))
    data = [data[idx] for idx in perm]
    syn_indices = [syn_indices[idx] for idx in perm]
    # sizes = [sizes[idx] for idx in perm]

    sum_sentences = 0
    for sample in data:
        sum_sentences += len(sample)

    print(('Prepared %d samples, %d sentences ' +
           '(%d ignored due to length == 0 or seq len > %d)') %
          (len(data), sum_sentences, ignored, opt.max_length))

    return data, syn_indices


def main():
    vocab = init_vocabulary('train vocab', opt.train, opt.vocab_path, opt.vocab_size)

    train_data, train_syn_indices = make_data(opt.train, opt.train_tree, vocab)
    valid_data, valid_syn_indices = make_data(opt.valid, opt.valid_tree, vocab)

    if opt.vocab_path is None:
        save_vocabulary('train vocab', vocab, opt.save_data + '/vocab/vocab.dict')

    print('Saving train data to \'' + opt.save_data + '/train/train.dep.pt\'...')
    torch.save({"data": train_data, "syn": train_syn_indices}, opt.save_data + '/train/train.dep.pt')

    print('Saving valid data to \'' + opt.save_data + '/valid/valid.dep.pt\'...')
    torch.save({"data": valid_data, "syn": valid_syn_indices}, opt.save_data + '/valid/valid.dep.pt')

    print('Saving vocab data to\'' + opt.save_data + '/vocab/vocab.pt\'...')
    torch.save(vocab, opt.save_data+'/vocab/vocab.pt')


if __name__ == "__main__":
    main()
