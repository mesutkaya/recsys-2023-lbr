from torch.cuda.amp import autocast
import torch.nn as nn
from transformers import AutoModel

"""
Sentence Pair Classifier that uses BERT, we add a classifier head on top of BERT. 
We can either freeze BERT weights or update them. 
Initial experiments show that updating BERT layer weights is more efficient for accuracy compared with freezing it. 
"""
class SPBERT(nn.Module):
    def __init__(self, tokenizer, bert_model="albert-base-v2", freeze_bert=False):
        super(SPBERT, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)
        self.bert_layer.resize_token_embeddings(len(tokenizer))
        #  Fix the hidden-state size of the encoder outputs (If you want to add other pre-trained models here, search for the encoder output size)
        hidden_size = 768

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        # Classification layer
        self.cls_layer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.1)

  
    @autocast()  # run in mixed precision
    def forward(self, input_ids, attn_masks, token_type_ids):
        '''
        Inputs:
            -input_ids : Tensor  containing token ids
            -attn_masks : Tensor containing attention masks to be used to focus on non-padded values
            -token_type_ids : Tensor containing token type ids to be used to identify sentence1 and sentence2
        '''

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        pooler_output = self.bert_layer(input_ids, attn_masks, token_type_ids, return_dict=False)

        # Feeding to the classifier layer the last layer hidden-state of the [CLS] token further processed by a
        # Linear Layer and a Tanh activation. The Linear layer weights were trained from the sentence order prediction (ALBERT) or next sentence prediction (BERT)
        # objective during pre-training.

        logits = self.cls_layer(self.dropout(pooler_output[1]))
        return logits
