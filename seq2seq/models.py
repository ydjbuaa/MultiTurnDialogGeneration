# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.vocab import Constants


class GlobalAttention(nn.Module):
    def __init__(self, dim):
        super(GlobalAttention, self).__init__()
        self.linear_in = nn.Linear(dim, dim, bias=False)
        self.sm = nn.Softmax()
        self.linear_out = nn.Linear(dim * 2, dim, bias=False)
        self.tanh = nn.Tanh()
        self.mask = None

    def apply_mask(self, mask):
        self.mask = mask

    def forward(self, attn_input, context):
        """
        attn_input: batch x dim
        context: batch x sourceL x dim
        """
        target = self.linear_in(attn_input).unsqueeze(2)  # batch x dim x 1

        # Get attention
        attn = torch.bmm(context, target).squeeze(2)  # batch x sourceL
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))

        attn = self.sm(attn)
        attn3 = attn.view(attn.size(0), 1, attn.size(1))  # batch x 1 x sourceL

        weighted_context = torch.bmm(attn3, context).squeeze(1)  # batch x dim
        context_combined = torch.cat((weighted_context, attn_input), 1)

        context_output = self.tanh(self.linear_out(context_combined))

        return context_output, attn


class MaskGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MaskGRU, self).__init__()
        self.gru_cell = nn.GRUCell(input_size, hidden_size)

        self.gru_cell.reset_parameters()

    def forward(self, seq_input, seq_mask, hidden):
        hidden = hidden[0]
        outputs = []
        for t in range(seq_input.size(0)):
            hidden_t = self.gru_cell(seq_input[t], hidden)
            mask_t = seq_mask[t].unsqueeze(1).expand_as(hidden_t)
            hidden = hidden_t * mask_t + hidden * (1.0 - mask_t)

            outputs.append(hidden_t * mask_t)

        outputs = torch.stack(outputs)
        hidden = hidden.unsqueeze(0)
        return outputs, hidden


class MaskBiGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MaskBiGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size // 2
        self.num_directions = 2

        self.gru_c1 = nn.GRUCell(input_size, self.hidden_size)
        self.gru_c2 = nn.GRUCell(input_size, self.hidden_size)

        self.gru_c1.reset_parameters()
        self.gru_c2.reset_parameters()
        pass

    def make_init_hidden(self, seq_input):
        return Variable(seq_input.data.new(self.num_directions, seq_input.size(1),
                                           self.hidden_size).zero_(),
                        requires_grad=False)

    def forward(self, seq_input, seq_mask, hidden=None):
        if hidden is None:
            hidden = self.make_init_hidden(seq_input)

        outputs_1, outputs_2 = [], []
        hidden_1 = hidden[0]
        hidden_2 = hidden[1]

        seq_len = seq_input.size(0)
        for t in range(seq_len):
            hidden_t = self.gru_c1(seq_input[t], hidden_1)
            mask_t = seq_mask[t].unsqueeze(1).expand_as(hidden_1)

            hidden_1 = hidden_t * mask_t + hidden_1 * (1.0 - mask_t)
            outputs_1.append(hidden_t * mask_t)

        for t in range(seq_len - 1, -1, -1):
            hidden_t = self.gru_c2(seq_input[t], hidden_2)
            mask_t = seq_mask[t].unsqueeze(1).expand_as(hidden_t)

            hidden_2 = hidden_t * mask_t + hidden_2 * (1.0 - mask_t)
            outputs_2.append(hidden_t * mask_t)

        hidden = torch.stack([hidden_1, hidden_2], 0)
        outputs = []
        for t in range(seq_len):
            output1 = outputs_1[t]
            output2 = outputs_2[seq_len - 1 - t]
            output = torch.cat([output1, output2], 1)
            outputs.append(output)
        outputs = torch.stack(outputs)
        return outputs, hidden


class DecoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, input_feed, drop_rate=0.):
        super(DecoderGRU, self).__init__()

        self.input_feed = input_feed
        self.hidden_size = hidden_size
        self.input_size = input_size
        if self.input_feed:
            self.input_size += hidden_size

        self.rnn = nn.GRU(self.input_size, hidden_size)
        self.attn = GlobalAttention(hidden_size)
        self.dropout = nn.Dropout(drop_rate)
        pass

    def forward(self, trg_input, hidden, context, init_output):
        """
        :param trg_input: target sequence embeddings trg_seqL * batch * emb_dim
        :param hidden: hidden state
        :param context: encoder outputs as context(batch * seqL * dim)
        :param init_output: init output of decoder
        :return:
        """
        outputs = []
        output = init_output
        attn = None
        for t in range(trg_input.size(0)):
            emb_t = trg_input[t]
            # feet last output as the input
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t.unsqueeze(0), hidden)
            output = output.squeeze(0)
            output, attn = self.attn(output, context)
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn


class Seq2SeqDialogModel(nn.Module):
    def __init__(self, config):
        super(Seq2SeqDialogModel, self).__init__()
        self.utter_hidden_size = config['utter_hidden_size']
        self.embedding = nn.Embedding(config['vocab_size'],
                                      config['word_emb_size'],
                                      padding_idx=Constants.PAD)

        self.encoder = MaskBiGRU(config['word_emb_size'],
                                 config['enc_hidden_size'])

        self.utter_rnn = nn.GRU(config['enc_hidden_size'],
                                config['utter_hidden_size'])

        self.decoder = DecoderGRU(config['word_emb_size'],
                                  config['dec_hidden_size'],
                                  input_feed=config['input_feed'])

        pass

    def fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                .transpose(1, 2).contiguous() \
                .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def make_init_utterance_hidden(self, utter_input):
        batch_size = utter_input.size(1)
        return Variable(utter_input.data.new(1, batch_size,
                                             self.utter_hidden_size).zero_(),
                        requires_grad=False)

    def make_init_decoder_output(self, context):
        batch_size = context.size(0)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def forward(self, src_input, trg_input, utter_hidden=None):
        # get source mask
        src_mask = Variable(src_input.data.ne(Constants.PAD).float())
        # get embeddings of source and target
        src_embs = self.embedding(src_input)
        # print(src_input)
        # print(src_mask)

        enc_outputs, enc_hidden = self.encoder(src_embs, src_mask)
        utter_input = self.fix_enc_hidden(enc_hidden)
        # print(enc_hidden.size(), utter_input.size())

        if utter_hidden is None:
            utter_hidden = self.make_init_utterance_hidden(utter_input)

        _, utter_hidden = self.utter_rnn(utter_input, utter_hidden)
        # print(utter_hidden.size())

        trg_embs = self.embedding(trg_input)
        context = enc_outputs.transpose(0, 1)
        dec_init_output = self.make_init_decoder_output(context)
        dec_outputs, _, _ = self.decoder(trg_embs,
                                         utter_hidden,
                                         context,
                                         dec_init_output)

        return dec_outputs, utter_hidden
