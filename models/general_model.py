import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel,AutoTokenizer
class BaseModel(nn.Module):
    def __init__(self, model,num_features,head, outnorm = True,norm_layer=nn.LayerNorm,**kwargs): ##nn.BatchNorm1d
        super().__init__()
        self.model = model
        self.head = head
        if outnorm:
            self.norm_layer = norm_layer(num_features, eps=1e-6)
        else:
            self.norm_layer = nn.Identity()
        
    def forward(self, x):
        x = self.model(x)
        x = self.norm_layer(x)
        x = self.head(x)
        return x



class Img_text_Model(nn.Module):
    def __init__(self, model,mix_model,num_features,head, outnorm = False,norm_layer=nn.LayerNorm,**kwargs): ##nn.BatchNorm1d
        super().__init__()
        self.model = model #nn.Identity()#
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",use_fast=False)
        self.nlp_model = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                                                    output_hidden_states=False,
                                                    return_dict=True)
        # for param in self.model.parameters():
        #     param.requires_grad = False
        # for param in self.nlp_model.parameters():
        #     param.requires_grad = False
        # self.nlp_model.train()
        self.text_norm = norm_layer(768, eps=1e-6)

        self.mix_model = mix_model
        self.head = head
        if outnorm:
            self.norm_layer = norm_layer(num_features, eps=1e-6)
        else:
            self.norm_layer = nn.Identity()
        
    def forward(self, x,x_text):
        x = self.model.forward_features(x)["x_norm_patchtokens"]
        x = self.norm_layer(x)

        # 2) text features
        x_text = self.tokenizer(
            x_text,
            max_length=256,
            padding=True,          
            truncation=True,        
            return_tensors="pt"
        )
        device = next(self.nlp_model.parameters()).device
        x_text = {k: v.to(device) for k, v in x_text.items()}
        text_out = self.nlp_model(**x_text)              # MaskedLMOutput or BaseModelOutput
        txt_feat = text_out.last_hidden_state             # → (B, seq_len, text_hidden_dim)
        txt_feat = self.text_norm(txt_feat)

        x = self.mix_model(x,txt_feat)
        x = self.head(x.mean(dim=1))
        # x = self.head(txt_feat.mean(dim=1))
        return x