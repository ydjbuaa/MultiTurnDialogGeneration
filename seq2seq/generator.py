# -*- coding:utf-8 -*-
import torch.nn.functional as F
from torch.autograd import Variable
from seq2seq.models import *
from utils.beam import Beam
from utils.dataset import Dataset


class DialogGenerator(object):
    def __init__(self, opt):
        self.opt = opt
        self.tt = torch.cuda if opt.cuda else torch
        self.beam_accum = None

        checkpoint = torch.load(opt.model)

        model_config = checkpoint['config']
        self.vocab = checkpoint['vocab']

        generator = nn.Linear(model_config['dec_hidden_size'], self.vocab.size)
        model = Seq2SeqDialogModel(model_config)

        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])

        if opt.cuda:
            model.cuda()
            generator.cuda()
        else:
            model.cpu()
            generator.cpu()

        model.generator = generator

        self.model = model
        self.model.eval()

    def init_beam_accum(self):
        self.beam_accum = {
            "predicted_ids": [],
            "beam_parent_ids": [],
            "scores": [],
            "log_probs": []}
        pass

    @staticmethod
    def _get_batch_size(batch):
        return batch.size(1)

    def build_data(self, batch_turns):
        # This needs to be the same as preprocess.py.
        dialog_data = []
        print(batch_turns)
        for batch_turn in batch_turns:
            turn_sent = []
            for words in batch_turn:
                sent = self.vocab.convert2idx(words,
                                              Constants.UNK_WORD,
                                              Constants.BOS_WORD,
                                              Constants.EOS_WORD)
                turn_sent += [sent]
            dialog_data += [turn_sent]

        return Dataset(dialog_data, self.opt.batch_size,
                       self.opt.cuda, volatile=True)

    def build_target_tokens(self, pred, src, attn):
        tokens = self.vocab.convert2words(pred, Constants.EOS)
        # tokens = tokens[:-1]  # EOS
        if self.opt.replace_unk:
            for i in range(len(tokens)):
                if tokens[i] == Constants.UNK_WORD:
                    _, max_index = attn[i].max(0)
                    tokens[i] = src[max_index[0]]
        return tokens

    def generate_conversation_batch(self, batch_turns):
        # Batch size is in different location depending on data.

        model = self.model
        beam_size = self.opt.beam_size

        # run encoder on N-1 turn
        utter_hidden = None
        enc_outputs = None
        for i in range(len(batch_turns) - 1):
            # get source mask
            src_turn = batch_turns[i]
            src_mask = Variable(src_turn.data.ne(Constants.PAD).float())
            # get embeddings of source and target
            src_embs = model.embedding(src_turn)
            # print(src_input)
            # print(src_mask)

            enc_outputs, enc_hidden = model.encoder(src_embs, src_mask)
            utter_input = model.fix_enc_hidden(enc_hidden)
            # print(enc_hidden.size(), utter_input.size())

            if utter_hidden is None:
                utter_hidden = model.make_init_utterance_hidden(utter_input)

            _, utter_hidden = model.utter_rnn(utter_input, utter_hidden)

        # trg_init_output = model.make_init_decoder_output(encoder_outputs.transpose(0, 1))

        # Drop the lengths needed for encoder.
        batch_size = enc_outputs.size(1)
        rnn_size = enc_outputs.size(2)

        decoder = self.model.decoder
        attention_layer = decoder.attn
        use_masking = True

        #  This mask is applied to the attention model inside the decoder
        #  so that the attention ignores source padding
        pad_mask = None

        def mask(pad_mask):
            if use_masking:
                attention_layer.apply_mask(pad_mask)

        # (3) run the decoder to generate sentences, using beam search

        # Expand tensors for each beam.
        context = Variable(enc_outputs.data.repeat(1, beam_size, 1))

        dec_states = Variable(utter_hidden.data.repeat(1, beam_size, 1))

        beam = [Beam(beam_size, self.opt.cuda) for _ in range(batch_size)]

        dec_out = model.make_init_decoder_output(context.transpose(0, 1))

        if use_masking:
            pad_mask = batch_turns[-2].data.eq(
                Constants.PAD).t() \
                .unsqueeze(0) \
                .repeat(beam_size, 1, 1)

        batch_idx = list(range(batch_size))
        remaining_sents = batch_size
        for i in range(self.opt.max_sent_length):
            mask(pad_mask)
            # Prepare decoder input.
            trg_input = torch.stack([b.getCurrentState() for b in beam
                                     if not b.done]).t().contiguous().view(1, -1)

            trg_embs = model.embedding(Variable(trg_input, volatile=True))
            dec_out, dec_states, attn = model.decoder(trg_embs,
                                                      dec_states,
                                                      context.transpose(0, 1),
                                                      dec_out)

            # decOut: (beam*batch) x hidden_size
            dec_out = dec_out.squeeze(0)
            out = model.generator(dec_out)
            out = F.log_softmax(out)

            print(trg_input.size(), out.size())

            # batch x beam x numWords
            word_lk = out.view(beam_size, remaining_sents, -1) \
                .transpose(0, 1).contiguous()
            attn = attn.view(beam_size, remaining_sents, -1) \
                .transpose(0, 1).contiguous()

            active = []
            for b in range(batch_size):
                if beam[b].done:
                    continue

                idx = batch_idx[b]
                if not beam[b].advance(word_lk.data[idx], attn.data[idx]):
                    active += [b]

                # layers x beam*sent x dim
                sent_states = dec_states.view(-1, beam_size,
                                              remaining_sents,
                                              dec_states.size(2))[:, :, idx]
                sent_states.data.copy_(
                    sent_states.data.index_select(
                        1, beam[b].getCurrentOrigin()))
                # update output
                # sent_dec_out = beam * hidden_size
                # sent_dec_out = dec_out.view(beam_size, remaining_sents, dec_out.size(1))[:, idx]
                # sent_dec_out.data.copy_(sent_dec_out.data.index_select(0, beam[b].getCurrentOrigin()))
                # print(beam[b].getCurrentOrigin())
                # print(dec_out.view(beam_size, remaining_sents, dec_out.size(1))[:, idx])

            if not active:
                break

            # in this section, the sentences that are still active are
            # compacted so that the decoder is not run on completed sentences
            active_idx = self.tt.LongTensor([batch_idx[k] for k in active])
            batch_idx = {beam: idx for idx, beam in enumerate(active)}

            def update_active(t):
                # select only the remaining active sentences
                view = t.data.view(-1, remaining_sents, rnn_size)
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents
                return Variable(view.index_select(1, active_idx)
                                .view(*new_size), volatile=True)

            dec_states = update_active(dec_states)
            dec_out = update_active(dec_out)
            context = update_active(context)

            if use_masking:
                pad_mask = pad_mask.index_select(1, active_idx)

            remaining_sents = len(active)

        print('over one ...')

        # (4) package everything up
        all_hyp, all_scores, all_attn = [], [], []
        n_best = self.opt.n_best

        for b in range(batch_size):
            scores, ks = beam[b].sortBest()

            all_scores += [scores[:n_best]]
            hyps, attn = zip(*[beam[b].getHyp(k) for k in ks[:n_best]])
            all_hyp += [hyps]

            if use_masking:
                valid_attn = batch_turns[-2].data[:, b].ne(Constants.PAD) \
                    .nonzero().squeeze(1)
                attn = [a.index_select(1, valid_attn) for a in attn]
            all_attn += [attn]

            if self.beam_accum:
                self.beam_accum["beam_parent_ids"].append(
                    [t.tolist()
                     for t in beam[b].prevKs])
                self.beam_accum["scores"].append([
                                                     ["%4f" % s for s in t.tolist()]
                                                     for t in beam[b].allScores][1:])
                self.beam_accum["predicted_ids"].append(
                    [[self.vocab.get_word(idx)
                      for idx in t.tolist()]
                     for t in beam[b].nextYs][1:])

        return all_hyp, all_scores, all_attn

    def generate_conversation(self, turns_data):
        #  (1) convert words to indexes
        batch_turns, turn_mask = self.build_data(turns_data)[0]
        batch_size = len(turns_data)
        print(batch_size)

        #  (2) translate
        pred, pred_score, attn = self.generate_conversation_batch(batch_turns)
        # pred, pred_score, attn, gold_score = list(zip(
        #     *sorted(zip(pred, pred_score, attn, gold_score, indices),
        #             key=lambda x: x[-1])))[:-1]

        #  (3) convert indexes to words
        pred_batch = []
        for b in range(batch_size):
            pred_batch.append(
                [self.build_target_tokens(pred[b][n], turns_data[b][-2], attn[b][n])
                 for n in range(self.opt.n_best)]
            )

        return pred_batch, pred_score, 0
