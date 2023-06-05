---
license: apache-2.0
---
# LoRA-MPT
A repo to make it so that you can easily fine tune MPT-7B using LoRA. This uses the alpaca dataset but can easily be adapted to use another. 

## Setup

To use as a library in another project/directory simply:
```
pip install -e ./
```

or if you want to build a project within this directly just do a git clone and modify the files in the src folder. 


## Fine Tuning

To fine tune the MPT-7B model on the Alpaca dataset from Stanford using LoRA use the following command:
```
python src/finetune.py --base_model 'mosaicml/mpt-7b-instruct' --data_path 'yahma/alpaca-cleaned' --output_dir './lora-mpt' --lora_target_modules '[Wqkv]'
```

The hyperparameters can be tweaked using the following flags as well:

```
python src/finetune.py \
    --base_model 'mosaicml/mpt-7b-instruct' \
    --data_path 'yahma/alpaca-cleaned' \
    --output_dir './lora-mpt' \
    --batch_size 128 \
    --micro_batch_size 4 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[Wqkv]' \
    --train_on_inputs \
    --group_by_length
    --use_gradient_checkpointing True
```

To speed up training at the expense of GPU memory run with --use_gradient_checkpointing False.

## Inference

A Gradio Interface was also created which can be used to run the inference of the model, once fine-tuned, using:

```
python src/generate.py --load_8bit --base_model 'mosaicml/mpt-7b-instruct' --lora_weights 'lora-mpt'
```


## MosaicML Platform

If you're interested in [training](https://www.mosaicml.com/training) and [deploying](https://www.mosaicml.com/inference) your own MPT or LLMs on the MosaicML Platform, [sign up here](https://forms.mosaicml.com/demo?utm_source=huggingface&utm_medium=referral&utm_campaign=mpt-7b).

Note: I left this in as a thank you to MosaicML for open-sourcing their model.

## Attributions 

I would like to thank the wonderful people down at MosaicML for releasing this model to the public. I believe that the future impacts of AI will be much better if its development is democratized. 

```
@online{MosaicML2023Introducing,
    author    = {MosaicML NLP Team},
    title     = {Introducing MPT-7B: A New Standard for Open-Source, 
    ly Usable LLMs},
    year      = {2023},
    url       = {www.mosaicml.com/blog/mpt-7b},
    note      = {Accessed: 2023-03-28}, % change this date
    urldate   = {2023-03-28} % change this date
}
```

This repo also adapted/built on top of code from Lee Han Chung https://github.com/leehanchung/mpt-lora which was adapted from tloen's repo for training LLaMA on Alpaca using LoRA https://github.com/tloen/alpaca-lora so thank you to them as well. 
