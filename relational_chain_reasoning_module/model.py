import torch
from transformers import RobertaModel, RobertaTokenizer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import ContrastiveLoss, Attention_layer


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
        question_text = '<s> ' + question_text + ' </s>'
        tokenized_question = self.roberta_tokenizer.tokenize(question_text)
        if len(tokenized_question) > self.max_sent_len:
            padded_tokenized_question = tokenized_question[:self.max_sent_len]
        else:
            padded_tokenized_question = tokenized_question + ['<pad>'] * (self.max_sent_len - len(tokenized_question))
        encoded_question = torch.tensor(
            self.roberta_tokenizer.encode(padded_tokenized_question, add_special_tokens=False),
            device=torch.device('cuda'))
        unmask_count = min(self.max_sent_len, len(tokenized_question))
        question_mask = torch.tensor([1] * unmask_count + [0] * (self.max_sent_len - unmask_count), dtype=torch.long,
                                     device=torch.device('cuda'))
        roberta_outputs = self.roberta_model(encoded_question, attention_mask=question_mask)[0]
        roberta_outputs = roberta_outputs.transpose(1, 0)[0]
        roberta_outputs = self.fc_dim12dim2(torch.nn.functional.relu(self.fc_bert2dim1(roberta_outputs)))
        embedded_chain = self.relation_embed_layer(relational_chain_idxs.unsqueeze(0))
        packed_chain = pack_padded_sequence(embedded_chain, relation_chain_lengths, batch_first=True)
        packed_outputs, _ = self.BiLSTM(packed_chain)
        chain_outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True, padding_value=0.0,
                                               total_length=max_chain_len)
        chain_outputs = self.attention_layer(chain_outputs.permute(1, 0, 2), relation_chain_lengths)
        if is_test:
            similarity = self.loss_criterion.get_similarity(roberta_outputs, chain_outputs)
            return similarity
        else:
            euclidean_loss = self.loss_criterion(roberta_outputs, chain_outputs, label=label)
            return euclidean_loss
