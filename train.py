import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel
from tokenizations.bpe_tokenizer import get_encoder

import pdb

def build_files(data_path, tokenized_data_path, ctx, full_tokenizer, min_length):
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line.replace('\n', ' [SEP] ') for line in lines]
    all_len = len(lines)
    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)
    # print("all_len: ", all_len)
    # print("ctx: ", ctx)
    all_examples = []
    for i in tqdm(range(all_len)):
        # if i == 100:
        #     break
        sublines = [lines[i]]
        sublines = [full_tokenizer.tokenize(line) for line in sublines if
                    len(line) > min_length]
        sublines = [full_tokenizer.convert_tokens_to_ids(line) for line in sublines]
        full_line = []
        for subline in sublines:
            full_line.extend(subline)
        all_examples.append(full_line)
        # line_str = ''
        # for id in full_line:
        #     line_str = line_str + str(id) + ' '
        # all_examples.append(line_str)
    # all_examples_str = json.dumps(all_examples)
    # with open(tokenized_data_path + 'tokenized_train_all.txt', 'w') as f:
    #     f.write(all_examples_str)
    print('finish')
    return all_examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False)
    parser.add_argument('--model_config', default='config/model_config_small.json', type=str, required=False)
    parser.add_argument('--tokenizer_path', default='cache/vocab_small.txt', type=str, required=False)
    parser.add_argument('--raw_data_path', default='data/train.json', type=str, required=False,)
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False)
    parser.add_argument('--raw', action='store_true')
    parser.add_argument('--epochs', default=5, type=int, required=False)
    parser.add_argument('--batch_size', default=8, type=int, required=False)
    parser.add_argument('--lr', default=1.5e-4, type=float, required=False)
    parser.add_argument('--warmup_steps', default=2000, type=int, required=False)
    parser.add_argument('--log_step', default=1, type=int, required=False)
    parser.add_argument('--stride', default=768, type=int, required=False)
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fp16_opt_level', default='O1', type=str, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--num_pieces', default=100, type=int, required=False)
    parser.add_argument('--min_length', default=128, type=int, required=False)
    parser.add_argument('--output_dir', default='model/', type=str, required=False)
    parser.add_argument('--pretrained_model', default='', type=str, required=False)
    parser.add_argument('--writer_dir', default='tensorboard_summary/', type=str, required=False)
    parser.add_argument('--segment', action='store_true')
    parser.add_argument('--bpe_token', action='store_true', help='subword')
    parser.add_argument('--encoder_json', default="tokenizations/encoder.json", type=str, help="encoder.json")
    parser.add_argument('--vocab_bpe', default="tokenizations/vocab.bpe", type=str, help="vocab.bpe")

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    if args.segment:
        from tokenizations import tokenization_bert_word_level as tokenization_bert
    else:
        from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    if args.bpe_token:
        full_tokenizer = get_encoder(args.encoder_json, args.vocab_bpe)
    else:
        full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('using device:', device)
    print('batch size:', args.batch_size)

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    stride = args.stride
    gradient_accumulation = args.gradient_accumulation
    fp16 = args.fp16
    fp16_opt_level = args.fp16_opt_level
    max_grad_norm = args.max_grad_norm
    num_pieces = args.num_pieces
    min_length = args.min_length
    output_dir = args.output_dir
    # pdb.set_trace()

    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if raw:
        print('building files')
        samples = build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, ctx=n_ctx,
                    full_tokenizer=full_tokenizer, min_length=min_length)
        samples = [sample for sample in samples if len(sample)==n_ctx]
        # pdb.set_trace()
        print('files built')
        all_len = len(samples)
        print('all len: ', all_len)

    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        print('From pretrained model: ' + args.pretrained_model)
        # pdb.set_trace()
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
        model.train()
        # pdb.set_trace()
        # for param in model.base_model.parameters():
        #     param.requires_grad = False
        # model.transformer.h[0].parameters # total 0~9
        # pdb.set_trace()

    model.train()
    model.to(device)


    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    print('calculating total steps')
    total_steps = int(all_len * epochs / batch_size)
    print('total steps = {}'.format(total_steps))

    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps,
                                                          t_total=total_steps)
    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DataParallel(model, device_ids=[int(i) for i in args.device.split(',')])
        multi_gpu = True
    print('starting training')
    overall_step = 0
    running_loss = 0

    # samples = []
    # # pdb.set_trace()
    # with open(tokenized_data_path + 'tokenized_train_all.txt', 'r') as f:
    #     line_list = json.load(f)
    # for idx, tokens in enumerate(line_list):
    #     tokens = tokens.split()
    #     tokens = [int(token) for token in tokens]
    #     try:
    #         assert len(tokens) == n_ctx
    #         samples.append(tokens)
    #     except:
    #         print(idx)
    #         continue

    # assert len(samples) == all_len
    save_epoch = 3
    print("save_epoch: ", save_epoch)
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))

        random.shuffle(samples)
        print('processing in epoch ', epoch)
        for step in tqdm(range(len(samples) // batch_size)):  # drop last

            #  prepare data
            batch = samples[step * batch_size: (step + 1) * batch_size]
            batch_inputs = []
            for ids in batch:
                int_ids = [int(x) for x in ids]
                batch_inputs.append(int_ids)
            batch_inputs = torch.tensor(batch_inputs).long().to(device)

            #  forward pass
            outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
            loss, logits = outputs[:2]

            #  get loss
            if multi_gpu:
                loss = loss.mean()
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation

            #  loss backward
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #  optimizer step
            if (overall_step + 1) % gradient_accumulation == 0:
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            if step == (len(samples) // batch_size) - 1:
                log_step = (len(samples) // batch_size) - 1
                print('now time: {}:{}. Step {} of epoch {}, loss {}'.format(
                    datetime.now().hour,
                    datetime.now().minute,
                    step + 1,
                    epoch + 1,
                    running_loss * gradient_accumulation / (log_step / gradient_accumulation)))
                running_loss = 0
            overall_step += 1

        if (epoch + 1) % save_epoch == 0:
            print('saving model for epoch {}'.format(epoch + 1))
            if not os.path.exists(output_dir + 'model_epoch{}'.format(epoch + 1)):
                os.mkdir(output_dir + 'model_epoch{}'.format(epoch + 1))
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(epoch + 1))
            # torch.save(scheduler.state_dict(), output_dir + 'model_epoch{}/scheduler.pt'.format(epoch + 1))
            # torch.save(optimizer.state_dict(), output_dir + 'model_epoch{}/optimizer.pt'.format(epoch + 1))
            print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')
    if not os.path.exists(output_dir + 'final_model'):
        os.mkdir(output_dir + 'final_model')
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(output_dir + 'final_model')
    # torch.save(scheduler.state_dict(), output_dir + 'final_model/scheduler.pt')
    # torch.save(optimizer.state_dict(), output_dir + 'final_model/optimizer.pt')


if __name__ == '__main__':
    main()
