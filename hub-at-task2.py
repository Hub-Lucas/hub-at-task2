import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, MSELoss
from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_roberta import RobertaModel, RobertaClassificationHead


class RobertaClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size*2)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size*2, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaInceptionForSequenceClassification(BertPreTrainedModel):
    config_class = RobertaConfig

    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.tfidf = None
        self.max_length = config.max_length
        self.classifier = RobertaClassificationHead(config)


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            tfidf=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        self.tfidf = tfidf

        roberta_output = outputs[2][-1]

        batch_layer_out_tfidf = torch.zeros_like(roberta_output)
        i = 0
        for l_out, tf in zip(roberta_output, tfidf):
            l_out_tf = torch.mm(tf, l_out)  # (1,hidden_size)
            tf = tf.view(-1, 1)  # (seq_length,1)
            weighted_out = torch.mm(tf, l_out_tf)  # (seq_length, hidden_size)

            batch_layer_out_tfidf[i] = weighted_out
            i += 1

        roberta_weighted_output = torch.as_tensor(batch_layer_out_tfidf)

        out0 = roberta_output
        out1 = roberta_weighted_output

        out = torch.cat([out0, out1], 2)
        logits = self.classifier(out)


        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = [loss, logits]
        else:
            outputs = [logits, ]
        return outputs  # (loss), logits, (hidden_states), (attentions)
