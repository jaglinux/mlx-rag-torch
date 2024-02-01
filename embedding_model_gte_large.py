import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class Embedding_Model():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-large")
        self.model = AutoModel.from_pretrained("thenlper/gte-large")

    def average_pool(self, last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def create_embeddings(self, input_texts):
        # Tokenize the input texts
        mps_device = torch.device("mps")
        batch_dict = self.tokenizer(input_texts, max_length=512, 
                                    padding=True, truncation=True, return_tensors='pt')

        outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # (Optionally) normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
def caclulate_scores_index(source_data_emb, query_data_emb):
    scores = (query_data_emb @ source_data_emb.T) * 100
    print(scores.tolist())
    return torch.argmax(scores).item()
 
# Code to test above embedding model
# query = "what is the capital of China?"
# input_texts = [
#     "how to implement quick sort in python?",
#     "Beijing",
#     "sorting algorithms"
# ]
# e = Embedding_Model()
# source_data_emb = e.create_embeddings(input_texts)
# query_data_emb = e.create_embeddings(query)
# index = caclulate_scores_index(source_data_emb, query_data_emb)
# print(f"question is {query}")
# print(f"index is {index} and Context is {input_texts[index]}")