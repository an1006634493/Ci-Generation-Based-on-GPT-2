# Ci Generation Based on GPT-2

## Introduction

- This project is based on [GPT2-Chinese](https://github.com/Morizeyao/GPT2-Chinese).
- To create environment, please use `requiremets.txt`.

## Datas

- All used 'Shi' and 'Ci' data are from [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry).
- You can preprocess the raw data with `preprocessing.py`, 
but we recommend simply use our [preprocessed data](https://drive.google.com/drive/folders/1daj7wzaDJPs9p80EaDXWcqCki-bMHPhr?usp=sharing) 
and place under root directory.

## Pretrained Models

- The 'Sanwen' pretrained model is provided by [hughqiu](https://github.com/hughqiu "hughqiu") 
and can be downloded [here](https) ://drive.google.com/drive/folders/1rJC4niJKMVwixUQkuL9k5teLRnEYTmUf?usp=sharing "gDrive").
- You can pretrain on 'Shi' data by yourself. Here we provide our [pretrained Shi model](https://drive.google.com/drive/folders/1qSVWn-NtzWW_XI8AuQx5TL1hk7UxLhZ_?usp=sharing).

## Finetuned Models

- The finetuned models on 'Ci' data will generated under `model`. Here we provide our [finetuned Ci model](https://drive.google.com/drive/folders/1f2WeXOoO6t7YPzMRPX9D_TsSQI0fXrkT?usp=sharing).

## Training

- After preparing all files described above, you can training the finetune model using:
```
sh train.sh
```
- If you want to pretrain or train on other setting, please defin your own training bash file.

## Generating

- After finetuning, you can generate your own 'Ci' under certain 'Cipai'. With our finetuned model, you can test it by:
```
python generate.py
```
- If you want to test on other models and setting, please check the `args` in `generate.py`.
