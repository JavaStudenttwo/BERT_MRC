import torch
from datetime import datetime, time
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BasicTokenizer, WordpieceTokenizer

# # 加载词典 pre-trained model tokenizer (vocabulary)
# VOCAB = './bert-large-cased-vocab.txt'
# tokenizer = BertTokenizer.from_pretrained(VOCAB)

text1 = "[CLS]预训练语言模型测试[SEP]问答系统信息抽取知识图谱[SEP] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"

text2 = "The coronavirus disease 2019 (COVID-19) outbreak is an ongoing global health emergence, but the pathogenesis remains unclear. We revealed blood cell immune response profiles using 5' mRNA, TCR and BCR V(D)J transcriptome analysis with single-cell resolution. Data from 134,620 PBMCs and 83,387 TCR and 12,601 BCR clones was obtained, and 56 blood cell subtypes and 23 new cell marker genes were identified from 16 participants. The number of specific subtypes of immune cells changed significantly when compared patients with controls. Activation of the interferon-MAPK pathway is the major defense mechanism, but MAPK transcription signaling is inhibited in cured patients. TCR and BCR V(D)J recombination is highly diverse in generating different antibodies against SARS-CoV-2. Therefore, the interferon-MAPK pathway and TCR- and BCR-produced antibodies play important roles in the COVID-19 immune response. Immune deficiency or immune over-response may result"

text3 = "GEO (P=3.60E-02) and Cordoba ( P =8.02E-03) datasets and confirmed by qPCR ( P =0.001). The most significant SNP, rs3741869 ( P =3.2E-05) in OASIS locus 12p11.21,"

# 加载词典 pre-trained model tokenizer (vocabulary)
EN_VOCAB = './bert-large-cased-vocab.txt'

CN_VOCAB = './vocab.txt'


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer.tokenize(text2)
print(tokenized_text)

# tokenizer = BasicTokenizer.from_pretrained(VOCAB)
basicTokenizer = BasicTokenizer()
tokenized_text = basicTokenizer.tokenize(text2)
print(tokenized_text)

# tokenizer = WordpieceTokenizer.from_pretrained(VOCAB)
wordpieceTokenizer = WordpieceTokenizer(CN_VOCAB)
tokenized_text = wordpieceTokenizer.tokenize(text2)
print(tokenized_text)

# Mask a token that we will try to predict back with `BertForMaskedLM`
masked_index = 8
tokenized_text[masked_index] = '[MASK]'
# assert tokenized_text == ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', '[MASK]', 'was', 'a', 'puppet',
#                           '##eer', '[SEP]']

# 将 token 转为 vocabulary 索引
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
# 定义句子 A、B 索引
segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

# 将 inputs 转为 PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])


# 英文bert词表
# ['[CLS]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[SEP]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '信', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[SEP]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', 'henson', 'was', 'a', 'puppet', '##eer', '[SEP]']
# ['[CLS]', '预', '训', '练', '语', '言', '模', '型', '测', '试', '[SEP]', '问', '答', '系', '统', '信', '息', '抽', '取', '知', '识', '图', '谱', '[SEP]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', 'henson', 'was', 'a', 'puppeteer', '[SEP]']
# ['[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', 'a', '[UNK]', '[UNK]']

# 英文uncased
# ['ge', '##o', '(', 'p', '=', '3', '.', '60', '##e', '-', '02', ')', 'and', 'co', '##rd', '##ob', '##a', '(', 'p', '=', '8', '.', '02', '##e', '-', '03', ')', 'data', '##set', '##s', 'and', 'con', '##firm', '##ed', 'by', 'q', '##pc', '##r', '(', 'p', '=', '0', '.', '001', ')', '.', 'the', 'most', 'sign', '##if', '##ica', '##nt', 's', '##np', ',', 'rs', '##37', '##41', '##86', '##9', '(', 'p', '=', '3', '.', '2', '##e', '-', '05', ')', 'in', 'oa', '##sis', 'lo', '##cus', '12', '##p', '##11', '.', '21', ',']

# ['ge', '##o', '(', 'p', '=', '3', '.', '60', '##e', '-', '02', ')', 'and', 'co', '##rd', '##ob', '##a', '(', 'p', '=', '8', '.', '02', '##e', '-', '03', ')', 'data', '##set', '##s', 'and', 'con', '##firm', '##ed', 'by', 'q', '##pc', '##r', '(', 'p', '=', '0', '.', '001', ')', '.', 'the', 'most', 'sign', '##if', '##ica', '##nt', 's', '##np', ',', 'rs', '##37', '##41', '##86', '##9', '(', 'p', '=', '3', '.', '2', '##e', '-', '05', ')', 'in', 'oa', '##sis', 'lo', '##cus', '12', '##p', '##11', '.', '21', ',']



# base-uncased
# ['[CLS]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[SEP]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '信', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[SEP]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', 'henson', 'was', 'a', 'puppet', '##eer', '[SEP]']
# ['[CLS]', '预', '训', '练', '语', '言', '模', '型', '测', '试', '[SEP]', '问', '答', '系', '统', '信', '息', '抽', '取', '知', '识', '图', '谱', '[SEP]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', 'henson', 'was', 'a', 'puppeteer', '[SEP]']
# ['[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', '[UNK]', 'a', '[UNK]', '[UNK]']