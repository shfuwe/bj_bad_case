from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import torch.nn.functional as F
import config

class fn_cls(nn.Module):
    def __init__(self,device):
        super(fn_cls, self).__init__()
        self.model = AutoModel.from_pretrained("bert-base-uncased")
        # self.model.resize_token_embeddings(len(tokenizer))##############
        self.model.to(device)
        self.dropout = nn.Dropout(0.5)
        self.l1 = nn.Linear(768, config.leishu)

    def forward(self, x, attention_mask=None):
        outputs = self.model(x, attention_mask=attention_mask)
#         print(outputs[0])torch.Size([8, 100, 768])
#         print(outputs[1])torch.Size([8, 768])
#         print(outputs[0][:,0,:])torch.Size([8, 768])
        x = outputs[1]
        # x = self.dropout(x)
        x = self.l1(x)
        return x