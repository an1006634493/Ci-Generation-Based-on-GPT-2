import json
import pdb
import chardet
from tokenizations import tokenization_bert
import os
from tqdm import tqdm

root = './chinese-poetry/ci/'
file_names = ['ci.song.' + str(i*1000) + '.json' for i in range(22)]

file_path = root + file_names[0]

with open(file_path, 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']

train_data = []
train_data_json_file = './data/train_1024.json'
n_ctx = 1024

for file_name in file_names:
    print(file_name)
    file_path = root + file_name

    with open(file_path, 'r', encoding=encoding) as f:
        ci_infors_list = json.load(f)
        for ci_info_dict in ci_infors_list:
            rhythmic = ci_info_dict['rhythmic']
            content = "".join(ci_info_dict['paragraphs'])
            if len(content) > 1000:
                continue
            data = '[MASK]' + rhythmic + '词牌名' + content
            len_data = len(rhythmic) + len(content) + 4
            cls_list = ['[CLS]' for _ in range(n_ctx-len_data)]
            cls_str = "".join(cls_list)
            data = data + cls_str
            train_data.append(data)

train_data_str = json.dumps(train_data)
with open(train_data_json_file, 'w', encoding=encoding) as f:
    f.write(train_data_str)

# pdb.set_trace()

with open(train_data_json_file, 'r') as f:
    check = json.load(f)
    print(len(check))
    print(check[:10])

# def build_files(data_path, tokenized_data_path, ctx, full_tokenizer, min_length):
#     with open(data_path, 'r', encoding='utf8') as f:
#         print('reading lines')
#         lines = json.load(f)
#         lines = [line.replace('\n', ' [SEP] ') for line in lines]  # 用[SEP]表示换行, 段落之间使用SEP表示段落结束
#     all_len = len(lines)
#     if not os.path.exists(tokenized_data_path):
#         os.mkdir(tokenized_data_path)
#     print("all_len: ", all_len)
#     print("ctx: ", ctx)
#     all_examples = []
#     for i in tqdm(range(all_len)):
#         sublines = [lines[i]]
#         # if i == num_pieces - 1:
#         #     sublines.extend(lines[all_len // num_pieces * (i + 1):])  # 把尾部例子添加到最后一个piece
#         sublines = [full_tokenizer.tokenize(line) for line in sublines if
#                     len(line) > min_length]  # 只考虑长度超过min_length的句子
#         sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
#         full_line = []
#         for subline in sublines:
#             # full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # 文章开头添加MASK表示文章开始
#             full_line.extend(subline)
#             # pdb.set_trace()
#             # cls_list = [' [CLS]' for _ in range(ctx - len(full_line))]
#             # cls_str = "".join(cls_list)
#             # full_line.append(full_tokenizer.convert_tokens_to_ids(cls_str))
#             # pdb.set_trace()
#             # while len(full_line) < ctx:
#             #     full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # 文章之间添加CLS表示文章结束
#         line_str = ''
#         for id in full_line:
#             line_str = line_str + str(id) + ' '
#         all_examples.append(line_str)
#     all_examples_str = json.dumps(all_examples)
#     with open(tokenized_data_path + 'tokenized_train_all.txt', 'w') as f:
#         f.write(all_examples_str)
#     print('finish')
#
# raw_data_path = 'data/train.json'
# tokenized_data_path = 'data/tokenized/'
# n_ctx = 1024
# tokenizer_path = 'cache/vocab_small.txt'
# full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=tokenizer_path)
# min_length = 1
#
# print('building files')
# build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, ctx=n_ctx,
#             full_tokenizer=full_tokenizer, min_length=min_length)
# print('files built')