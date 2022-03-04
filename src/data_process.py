from torch.utils.data import Dataset
def text2token(text,tokenizer,max_length=100):
    text2id = tokenizer(
        text, max_length=max_length, padding='max_length', truncation=True, return_tensors="pt"
    )
    input_ids=text2id["input_ids"].tolist()
    attention_mask=text2id["attention_mask"].tolist()
    return input_ids,attention_mask
def data2token(data_,tokenizer):
    text=[i for i in data_['title'].values]
    input_ids,attention_mask=text2token(text,tokenizer)
    data_['input_ids']=input_ids
    data_['attention_mask']=attention_mask
    return data_

class SentimentDataset(Dataset):
    def __init__(self,df):
        self.dataset = df
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "title"]
        label = self.dataset.loc[idx, "label"]
        input_ids = self.dataset.loc[idx, "input_ids"]
        attention_mask = self.dataset.loc[idx, "attention_mask"]
        sample = {"text": text, "label": label,"input_ids":input_ids,"attention_mask":attention_mask}
        # print(sample)
        return sample
    
