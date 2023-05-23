import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def vector_search(query, tokenizer, model, index, num_results=10):
    """Tranforms query to vector using a pretrained, sentence-level 
    DistilBERT model and finds similar vectors using FAISS.
    Args:
        query (str): User query that should be more than a sentence long.
        model (sentence_transformers.SentenceTransformer.SentenceTransformer)
        index (`numpy.ndarray`): FAISS index that needs to be deserialized.
        num_results (int): Number of results to return.
    Returns:
        D (:obj:`numpy.array` of `float`): Distance between results and query.
        I (:obj:`numpy.array` of `int`): Paper ID of the results.
    
    """
    query=list(query)
    encoded_input = tokenizer(query,padding=True, truncation=True, return_tensors='pt')

    with torch.no_grad():
        model_output = model(**encoded_input)

    vector = mean_pooling(model_output, encoded_input['attention_mask'])
    vector = F.normalize(vector, p=2, dim=1)


    #vector = model.encode(list(query))
    D, I = index.search(np.array(vector).astype("float32"), k=num_results)
    return D, I

def id2details(df, I, column):
    """Returns the paper titles based on the paper index."""
    return df.select(I[0])[column]
