import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import RobertaModel, RobertaTokenizer
import numpy as np

from relational_chain_reasoning_module.utils import ContrastiveLoss, Attention_layer


class Relational_chain_reasoning_module(torch.nn.Module):
    def __init__(self, relation_dim, dim_l1, dim_l2, lstm_hidden_dim, relation_embeddings, max_sent_len=32):
        # Make sure that the dimensions of the left and right vectors are the same in the similarity calculation
        assert (2 * lstm_hidden_dim) == dim_l2
        super(Relational_chain_reasoning_module, self).__init__()
        self.loss_criterion = ContrastiveLoss()
        self.relation_embed_layer = torch.nn.Embedding.from_pretrained(torch.tensor(relation_embeddings), freeze=True)
        self.BiLSTM = torch.nn.LSTM(relation_dim, lstm_hidden_dim, 1, bidirectional=True, batch_first=True)
        self.attention_layer = Attention_layer(hidden_dim=2 * lstm_hidden_dim, attention_dim=2 * lstm_hidden_dim)

        # pretrained transformer: Roberta-Base
        self.roberta_model = RobertaModel.from_pretrained('roberta-base')
        # The transformer can be fine-tuned when training the QA model
        for param in self.roberta_model.parameters():
            param.requires_grad = True
        self.roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self.fc_bert2dim1 = torch.nn.Linear(768, dim_l1, bias=True)
        torch.nn.init.xavier_normal_(self.fc_bert2dim1.weight.data)
        torch.nn.init.constant_(self.fc_bert2dim1.bias.data, val=0.0)
        self.fc_dim12dim2 = torch.nn.Linear(dim_l1, dim_l2, bias=False)
        torch.nn.init.xavier_normal_(self.fc_dim12dim2.weight.data)
        self.max_sent_len = max_sent_len
        self.cuda()

    def forward(self, question_text, relational_chain_idxs, relation_chain_lengths, max_chain_len, label=0,
                is_test=False):
        """
        Get the similarity between question description and relational_chain
        :param is_test: True: Get similarity during testing, default False: get loss during training
        :param max_chain_len: Maximum length in relation chain
        :param relation_chain_lengths:  The length of each relation chain
        :param question_text: Complete sentences
        :param relational_chain_idxs: A chain of relationships, each of which is represented by a subscript
        :return: Similarity
        """
        encoded_questions = self.roberta_tokenizer(question_text, add_special_tokens=True, max_length=self.max_sent_len,
                                                   padding="max_length", truncation=True, return_attention_mask=True,
                                                   return_tensors="pt")
        roberta_outputs = self.roberta_model(encoded_questions.input_ids.cuda(), attention_mask=encoded_questions.attention_mask.cuda())[0]
        roberta_outputs = roberta_outputs.transpose(1, 0)[0]
        roberta_outputs = self.fc_dim12dim2(torch.nn.functional.relu(self.fc_bert2dim1(roberta_outputs)))

        embedded_chain = self.relation_embed_layer(relational_chain_idxs.unsqueeze(0))
        embedded_chain = embedded_chain.squeeze(0)
        packed_chain = pack_padded_sequence(embedded_chain, relation_chain_lengths.cpu(), batch_first=True)
        packed_outputs, _ = self.BiLSTM(packed_chain)
        # có customize ở đây nữa -> dòng 2, tham số total_length
        if max_chain_len > 8:
            max_chain_len = 8
        chain_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, padding_value=0.0,
                                               total_length=max_chain_len)
        chain_outputs = self.attention_layer(chain_outputs.permute(1, 0, 2), relation_chain_lengths)
        if is_test:
            similarity = self.loss_criterion.get_similarity(roberta_outputs, chain_outputs)
            return similarity
        else:
            euclidean_loss = self.loss_criterion(roberta_outputs, chain_outputs, label=label)
            return euclidean_loss


class Answer_filtering_module(torch.nn.Module):
    def __init__(self, entity_embeddings, embedding_dim, vocab_size, word_dim, hidden_dim, fc_hidden_dim, relation_dim,
                 head_bn_filepath, score_bn_filepath):
        super(Answer_filtering_module, self).__init__()
        self.relation_dim = relation_dim * 2
        self.loss_criterion = torch.nn.BCELoss(reduction='sum')

        # hidden_dim * 2 is the BiLSTM + attention layer output
        self.fc_lstm2hidden = torch.nn.Linear(hidden_dim * 2, fc_hidden_dim, bias=True)
        torch.nn.init.xavier_normal_(self.fc_lstm2hidden.weight.data)
        torch.nn.init.constant_(self.fc_lstm2hidden.bias.data, val=0.0)
        self.fc_hidden2relation = torch.nn.Linear(fc_hidden_dim, self.relation_dim, bias=False)
        torch.nn.init.xavier_normal_(self.fc_hidden2relation.weight.data)
        self.entity_embedding_layer = torch.nn.Embedding.from_pretrained(torch.tensor(entity_embeddings), freeze=True)
        self.word_embedding_layer = torch.nn.Embedding(vocab_size, word_dim)
        self.BiLSTM = torch.nn.LSTM(embedding_dim, hidden_dim, 1, bidirectional=True, batch_first=True)
        self.softmax_layer = torch.nn.LogSoftmax(dim=-1)
        self.attention_layer = Attention_layer(hidden_dim=2 * hidden_dim, attention_dim=4 * hidden_dim)

        self.head_bn = torch.nn.BatchNorm1d(2)
        head_bn_params_dict = np.load(head_bn_filepath, allow_pickle=True).ravel()[0]
        self.head_bn.weight.data = torch.tensor(head_bn_params_dict["weight"])
        self.head_bn.bias.data = torch.tensor(head_bn_params_dict["bias"])
        self.head_bn.running_mean.data = torch.tensor(head_bn_params_dict["running_mean"])
        self.head_bn.running_var.data = torch.tensor(head_bn_params_dict["running_var"])
        # for key in head_bn_params_dict:
        #     self.head_bn[key].data = torch.tensor(head_bn_params_dict[key])

        self.score_bn = torch.nn.BatchNorm1d(2)
        score_bn_params_dict = np.load(score_bn_filepath, allow_pickle=True).ravel()[0]
        self.score_bn.weight.data = torch.tensor(score_bn_params_dict["weight"])
        self.score_bn.bias.data = torch.tensor(score_bn_params_dict["bias"])
        self.score_bn.running_mean.data = torch.tensor(score_bn_params_dict["running_mean"])
        self.score_bn.running_var.data = torch.tensor(score_bn_params_dict["running_var"])
        # for key in score_bn_params_dict:
        #     self.score_bn[key].data = torch.tensor(score_bn_params_dict[key])

    def complex_scorer(self, head, relation):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        head = self.head_bn(head)
        head = head.repeat(1, 1, 2)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]
        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(self.entity_embedding_layer.weight, 2, dim=1)
        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation
        score = torch.stack([re_score, im_score], dim=1)
        score = self.score_bn(score)
        score = score.permute(1, 0, 2)
        re_score = score[0]
        im_score = score[1]
        return torch.sigmoid(torch.mm(re_score, re_tail.repeat(1, 2).transpose(1, 0)) +
                             torch.mm(im_score, im_tail.repeat(1, 2).transpose(1, 0)))

    def forward(self, question, questions_length, head_entity, tail_entity, max_sent_len):
        """
            batch_questions_index,
            batch_questions_length,
            batch_head_entity,
            batch_onehot_answers,
            max_sent_len
        """
        embedded_question = self.word_embedding_layer(question.unsqueeze(0))
        embedded_question = embedded_question.squeeze(0)
        packed_input = pack_padded_sequence(embedded_question, questions_length.cpu(), batch_first=True)

        packed_output, _ = self.BiLSTM(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0, total_length=max_sent_len)
        output = self.attention_layer(output.permute(1, 0, 2), questions_length)
        output = torch.nn.functional.relu(self.fc_lstm2hidden(output))
        relation_embedding = self.fc_hidden2relation(output)
        pred_answers_score = self.complex_scorer(self.entity_embedding_layer(head_entity), relation_embedding)
        loss = self.loss_criterion(pred_answers_score, tail_entity)
        return loss

    def get_ranked_top_k(self, question, questions_length, head_entity, max_sent_len, K=5):
        embedded_question = self.word_embedding_layer(question.unsqueeze(0))
        embedded_question = embedded_question.squeeze(0)
        packed_input = pack_padded_sequence(embedded_question, questions_length.cpu(), batch_first=True)

        packed_output, _ = self.BiLSTM(packed_input)
        output, _ = pad_packed_sequence(packed_output, batch_first=True, padding_value=0.0, total_length=max_sent_len)
        output = self.attention_layer(output.permute(1, 0, 2), questions_length)
        output = torch.nn.functional.relu(self.fc_lstm2hidden(output))
        relation_embedding = self.fc_hidden2relation(output)
        pred_answers_score = self.complex_scorer(self.entity_embedding_layer(head_entity), relation_embedding)
        return torch.topk(pred_answers_score, k=K, dim=-1, largest=True, sorted=True)
