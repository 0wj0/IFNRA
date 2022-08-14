import torch
import torch.nn as nn
from transformers import BertModel
import torchvision
import torch.nn.functional as F


class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        
        print(f'num_layers:{num_layers}')

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, inputs, hidden):
        """
        :param inputs: 一个时间步的输入
        :param hidden: 各层的初始隐藏状态
        :return: (最后一层的隐藏状态，各层隐藏状态)
        """
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(inputs, h_0[i])
            inputs = h_1_i
            if i + 1 != self.num_layers:
                inputs = self.dropout(inputs)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return inputs, h_1


class RA_Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer=1, dropout=0.1):
        super().__init__()
        self.max_step_num = 3
        
        self.gru_cell = nn.ModuleList()
        for i in range(0, self.max_step_num):
            self.gru_cell.append(StackedGRU(num_layer, input_size, hidden_size, dropout))
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)
        self.multihead_attn_img = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, batch_first=True)
        
        self.text_img_gru = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, text_feature, text_mask, img_feature, img_mask, h0_s):
        """
        :param text_feature: original text representation,
                             shape should be [batch_size, text_tok_len, embedding_size(e.g. 768)]
        :param text_mask: similar to attention mask used in bert, shape should be [batch_size, text_tok_len]
        :param img_feature: img features extracted by BUA, shape should be [batch_size, feature_num, dim_feature]
        :param img_mask: similar to attention mask used in bert, indicates which img_feature is padding, shape should be [batch_size, feature_num]
        :param h0_s: initial hidden_states of each GRU layer, shape should be [num_layers, batch_size, hidden_size]
        :return: GRU hidden state on each step,
                 shape should be [max_steps, batch_size, hidden_size]
        """
        outputs = []
        
        key_padding_mask_img = torch.where(img_mask==1, False, True)
        ''' attn_output_img.shape = [batch_size, text_tok_len, embedding_size] '''
        attn_output_img, attn_output_weights_img = self.multihead_attn_img(text_feature, img_feature, img_feature,
                                                                           key_padding_mask=key_padding_mask_img)
        text_img_feature = self.text_img_gru(attn_output_img.reshape((-1, text_feature.shape[-1])), text_feature.reshape((-1, text_feature.shape[-1])))
        text_img_feature = text_img_feature.reshape(text_feature.shape)
        key_padding_mask = torch.where(text_mask==1, False, True)
        
        for step in range(0, self.max_step_num):

            last_layer_h = h0_s[-1]
            last_layer_h = last_layer_h.unsqueeze(dim=2)  # last_layer_h.shape = [batch_size, hidden_size, 1]
            
            attn_output, attn_output_weights = self.multihead_attn(last_layer_h.permute(0, 2, 1), text_img_feature, text_img_feature,
                                                                   key_padding_mask=key_padding_mask)
            cur_last_layer_h, h0_s = self.gru_cell[step](attn_output.squeeze(dim=1), h0_s)        # cur_last_layer_h.shape = [batch_size, hidden_size]
        
            outputs.append(cur_last_layer_h)

        outputs = torch.stack(outputs)
        assert len(outputs)==self.max_step_num
        return outputs


class IFNRA(nn.Module):
    def __init__(self, bert_pretrained_pth, gru_layer_num=1):
        super().__init__()
        self.gru_layer_num = gru_layer_num
        self.bert = BertModel.from_pretrained(bert_pretrained_pth)
        self.text_embed_dropout = nn.Dropout(0.1)
        self.embedding_vec_len = self.bert.embeddings.word_embeddings.weight.shape[-1]
        self.decoder = RA_Decoder(self.embedding_vec_len, self.embedding_vec_len, num_layer=gru_layer_num)
        self.fc = nn.Linear(2048, self.embedding_vec_len)
        self.sentiment_classifier = nn.Linear(self.embedding_vec_len, 3)

    def forward(self, inputs):
        input_ids, attention_mask, img_feature, img_mask, aspect_msk = inputs

        outputs_text = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        text = outputs_text.last_hidden_state
        text = self.text_embed_dropout(text)
        
        h0 = torch.matmul(aspect_msk.unsqueeze(dim=1), text).squeeze(dim=1) / torch.sum(aspect_msk, dim=1, keepdim=True)
        h0_s = torch.stack([h0] * self.gru_layer_num)

        img_feature = self.fc(img_feature)

        '''decode_outputs.shape = [max_steps, batch_size, hidden_size]'''
        decode_outputs = self.decoder(text, attention_mask, img_feature, img_mask, h0_s)
        
        ''' sentiment_features.shape = [batch_size, hidden_size] '''
        sentiment_features = decode_outputs[-1]
        sentiment_polarity = self.sentiment_classifier(sentiment_features)
        return sentiment_polarity
