from transformers import BertTokenizer, BertModel
import torch
import numpy as np

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Load model from HuggingFace Hub
tokenizer = BertTokenizer.from_pretrained('shibing624/text2vec-bge-large-chinese')
model = BertModel.from_pretrained('shibing624/text2vec-bge-large-chinese')
sentences = ['如何更换花呗绑定银行卡']
# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')


# model.eval()
input_ids = encoded_input['input_ids']
token_type_ids = encoded_input['token_type_ids']
attention_mask = encoded_input['attention_mask']

input_ids, attention_mask, token_type_ids = input_ids.numpy(), attention_mask.numpy(), token_type_ids.numpy()
if input_ids.shape[1] > 512:
    input_ids = input_ids[:, :512]
    attention_mask = attention_mask[:, :512]
    token_type_ids = token_type_ids[:, :512]
elif input_ids.shape[1] < 512:
    input_ids = np.pad(input_ids,
                        ((0, 0), (0, 512 - input_ids.shape[1])),
                        mode='constant', constant_values=0)
    attention_mask = np.pad(attention_mask,
                            ((0, 0), (0, 512 - attention_mask.shape[1])),
                            mode='constant', constant_values=0)
    token_type_ids = np.pad(token_type_ids,
                            ((0, 0), (0, 512 - token_type_ids.shape[1])),
                            mode='constant', constant_values=0)
input_ids = torch.tensor(input_ids)
token_type_ids = torch.tensor(token_type_ids)
attention_mask = torch.tensor(attention_mask)

# Compute token embeddings
with torch.no_grad():
    model_output = model(input_ids, attention_mask, token_type_ids)

print(input_ids)
print(token_type_ids)
print(attention_mask)

torch.onnx.export(model, (input_ids,attention_mask,token_type_ids), "text2vec-bge-large-chinese.onnx", input_names=['input_ids', 'attention_mask', 'token_type_ids'],dynamic_axes={'input_ids': {0: 'batch'}, 'attention_mask': {0: 'batch'}, 'token_type_ids': {0: 'batch'}})
# Perform pooling. In this case, mean pooling.
sentence_embeddings = mean_pooling(model_output, attention_mask)
print("Sentence embeddings:")
print(sentence_embeddings)
