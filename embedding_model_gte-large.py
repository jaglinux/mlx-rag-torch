import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class Embedding_Model():
    def __init__(self, input_texts):
        self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
        self.model = AutoModel.from_pretrained("thenlper/gte-large")
        self.input_texts = input_texts

    def average_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def create_embeddings(self):
        # Tokenize the input texts
        batch_dict = self.tokenizer(self.input_texts, max_length=512, 
                                    padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
def caclulate_scores(embeddings):
    scores = (embeddings[:1] @ embeddings[1:].T) * 100
    print(scores.tolist())
    return scores.tolist()
 
# Code to test above embedding model   
# input_texts = [
#     "what is the capital of China?",
#     "how to implement quick sort in python?",
#     "Beijing",
#     "sorting algorithms"
# ]
# e = Embedding_Model(input_texts)
# data = e.create_embeddings()
# caclulate_scores(data)