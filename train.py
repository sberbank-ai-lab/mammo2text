#!/usr/bin/env python
# coding: utf-8

import sys, os
# https://github.com/alexyalunin/dotfiles/blob/master/myutils.py
sys.path.append(os.path.expanduser('~')+'/dotfiles')
import myutils

import time, random, pickle, logging, uuid, collections, copy, json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy, sklearn
from functools import partial

import torch
use_cuda, device, n_gpu = myutils.print_torch(torch)

import transformers
myutils.print_packages(transformers)
from transformers import *
from datasets import *

SEED = 42
myutils.seed_everything(SEED, random, os, np, torch)
cache_dir = '~/.cache'

from mammo2text import BreastImgTextDataset, MammoImgTextDataCollator, get_target_text, get_mammo2text_model, \
    generate_predictions, AddParametersToMlflowCallback, check_tokenizer, load_image_model, EncoderWrapper, ignore_tokens_after_eos, save_eval_predictions
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import argparse
import logging
logging.getLogger().setLevel(logging.INFO)
import gc

def train(parameters):
    tokenizer = BertTokenizerFast.from_pretrained(parameters['tokenizer_path'], do_lower_case=False, use_fast=True)
    check_tokenizer(tokenizer)
    print('Vocab size:', len(tokenizer.get_vocab()))

    exam_list_eval = myutils.json_load(parameters['eval_json_path'])
    exam_list_train = myutils.json_load(parameters['train_json_path'])

    if parameters['debug']:
        logging_steps = 10
        eval_steps = 100
        warmup_steps = 0
        eval_size = 4
        max_steps = 2000
        exam_list_eval = exam_list_eval[2:2+eval_size]
        exam_list_train = exam_list_eval
    else:
        logging_steps = 100
        eval_steps = 1000
        warmup_steps = 300
        eval_size = 300
        max_steps = -1

    random_number_generator = np.random.RandomState()

    train_dataset = BreastImgTextDataset(
        exam_list=exam_list_train,
        tokenizer=tokenizer,
        parameters=parameters,
        random_number_generator=random_number_generator,
        swap=parameters['swap']
    )

    eval_parameters = parameters.copy()
    eval_parameters['augmentation'] = False

    eval_dataset = BreastImgTextDataset(
        exam_list=exam_list_eval[:eval_size],
        tokenizer=tokenizer,
        parameters=eval_parameters,
        random_number_generator=random_number_generator,
        swap=False
    )

    # for better readability
    exam_list_eval = sorted(exam_list_eval, key=lambda x: x['birads'])
    test_dataset = BreastImgTextDataset(
        exam_list=exam_list_eval,
        tokenizer=tokenizer,
        parameters=eval_parameters,
        random_number_generator=random_number_generator,
        swap=False
    )

    data_collator = MammoImgTextDataCollator(tokenizer)

    eval_dataloader_for_gen = DataLoader(
        eval_dataset,
        batch_size=parameters['bs'],
        shuffle=False,
        num_workers=4,
        collate_fn=data_collator
    )
    test_dataloader_for_gen = DataLoader(
        test_dataset,
        batch_size=parameters['bs'],
        shuffle=False,
        num_workers=4,
        collate_fn=data_collator
    )


    model = get_mammo2text_model(parameters, tokenizer, device, load_image_model, EncoderWrapper)
    if parameters['checkpoint']:
        model.load_state_dict(torch.load(parameters['checkpoint']))

    rouge = load_metric('rouge', experiment_id=uuid.uuid4())

    def calc_rouge_for_each_birads(rows):
        rouge_output = rouge.compute(
             predictions=rows['pred_str'],
             references=rows['label_str'],
             rouge_types=['rouge2'])['rouge2'].mid.fmeasure
        return {'target_birads': rows['target_birads'].iloc[0], 'rouge': rouge_output}


    # global variables
    trainer = None
    res_df = pd.DataFrame()
    prev_iter = 0
    is_test = False

    def compute_metrics(pred):
        nonlocal trainer, prev_iter, is_test, res_df
        nonlocal model, tokenizer, eval_dataloader_for_gen, test_dataloader_for_gen
        labels_ids = pred.label_ids
        pred_ids = pred.predictions
        pred_ids = ignore_tokens_after_eos(pred_ids, tokenizer)
        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = tokenizer.eos_token_id
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

        # ======== eval from generate ========
        if parameters['eval_from_generate'] or is_test:
            if is_test:
                dataloader_for_gen = test_dataloader_for_gen
                num_beams = 5
            else:
                dataloader_for_gen = eval_dataloader_for_gen
                num_beams = 3
            pred = generate_predictions(model, tokenizer, dataloader_for_gen,
                                        {'num_beams': num_beams})
            if is_test:
                myutils.excel.save(
                    df=pred,
                    path=f"models/{parameters['model_name']}/test_predictions.xlsx",
                    long_columns=['target', 'predicted']
                )
            pred_str = pred['predicted'].values
            label_str = pred['target'].values
        # ======== eval from generate ========
        rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=['rouge2'])['rouge2'].mid

        df = pd.DataFrame({'label_str': label_str, 'pred_str': pred_str})
        df['target_birads'] = df['label_str'].str[:1]
        df['pred_birads'] = df['pred_str'].str[:1]
        iter_num = trainer.state.global_step
        excel_path = f"models/{parameters['model_name']}/eval_dynamics.xlsx"
        if iter_num - prev_iter >= 2000 or iter_num <= 2000:
            prev_iter = iter_num
            save_eval_predictions(res_df, df, iter_num, excel_path)

        birads_f1 = f1_score(
            df['target_birads'].values,
            df['pred_birads'].values,
            average='micro'
        )
        birads_rouge = df.groupby(by='target_birads').apply(calc_rouge_for_each_birads)
        birads_rouge = {f"b{x['target_birads']}_rouge2_f1": x['rouge'] for x in birads_rouge.values}

        res = {
            'rouge2_precision': round(rouge_output.precision, 4),
            'rouge2_recall': round(rouge_output.recall, 4),
            'rouge2_fmeasure': round(rouge_output.fmeasure, 4),
        }
        res.update(birads_rouge)
        res['birads_f1'] = birads_f1
        return res

    training_args = TrainingArguments(
        output_dir=f"models/{parameters['model_name']}",
        per_device_train_batch_size=parameters['bs'],
        per_device_eval_batch_size=parameters['bs'],
        evaluation_strategy='steps',
        do_train=True,
        do_eval=True,
        logging_steps=logging_steps,
        save_steps=3000,
        eval_steps=eval_steps,
        # eval_steps=2,
        overwrite_output_dir=False,
        warmup_steps=warmup_steps,
        save_total_limit=2,
        fp16=True,
        num_train_epochs=parameters['epochs'],
        remove_unused_columns=False,
        dataloader_num_workers=4,
        metric_for_best_model='rouge2_fmeasure',
        greater_is_better=True,
        max_steps=max_steps,
        label_smoothing_factor=parameters['label_smoothing']
    )

    def floating_point_ops(self, inputs):
        return 0

    Trainer.floating_point_ops = floating_point_ops
    Trainer.prediction_step = overwrite.prediction_step_opt


    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    mlflow_add_cb = AddParametersToMlflowCallback(trainer, parameters)
    trainer.callback_handler.add_callback(mlflow_add_cb)

    # mlflow workaround
    class DummyClass:
        def to_dict(self):
            return {}

    model.config.encoder = DummyClass()
    model.config.decoder = DummyClass()

    import mlflow

    trainer.train(parameters['checkpoint'])
    model.save_pretrained(f"models/{parameters['model_name']}")
    # test
    is_test = True
    mlflow_run_id = mlflow_add_cb.run_id
    parameters['mlflow_run_id'] = mlflow_run_id
    mlflow.start_run(mlflow_run_id)
    trainer.evaluate(eval_dataset, metric_key_prefix='test')
    mlflow.end_run()
    myutils.json_dump(parameters, f"models/{parameters['model_name']}/parameters.json")



def main():
    parser = argparse.ArgumentParser(description='Run mammo2text')

    # if decoder_path is not provided the decoder of encoder_decoder_path will be used
    parser.add_argument('--debug', required=False, action="store_true", default=False)
    parser.add_argument('--model_name', required=True)
    parser.add_argument('--pipeline_with_attention', required=False, action="store_true", default=True)
    parser.add_argument('--freeze_image_model', required=False, action="store_true", default=False)
    parser.add_argument('--eval_from_generate', required=False, action="store_true", default=True)
    parser.add_argument('--bs', required=False, default=4, type=int)
    parser.add_argument('--epochs', required=False, default=5, type=int)
    parser.add_argument('--checkpoint', required=False, default=None)
    parser.add_argument('--augmentation', required=False, action="store_true", default=False)
    parser.add_argument('--swap', required=False, action="store_true", default=False)
    parser.add_argument('--use-heatmaps', action="store_true", default=False)
    parser.add_argument('--decoder_length', required=False, default=224, type=int)
    parser.add_argument('--encoder_decoder_path', required=False, default="/ayb/vol3/alexyalunin/summary/models/longformer2rubert_mlm_256")
    parser.add_argument('--encoder_path', required=False, default="/home/yaxen/sbermed/mammo/classification/history/descr2_2years_avg_max_avg_br22_x2_0.4_x1resnet_new_all_enb0_simple_clean_fold0/weight_BR2222222_008_0.858511.pt")
    parser.add_argument('--tokenizer_path', required=False, default="/ayb/vol3/alexyalunin/summary/models/longformer2rubert_mlm_256")
    parser.add_argument('--eval_json_path', required=False, default="data/dataset2/exam_list_eval.json")
    parser.add_argument('--train_json_path', required=False, default="data/dataset2/exam_list_train.json")
    parser.add_argument('--decoder_path', required=False, default=None)
    parser.add_argument('--label_smoothing', required=False, default=0.0, type=float)

    args = parser.parse_args()

    parameters = {
        "debug": args.debug,
        "model_name": args.model_name,
        "pipeline_with_attention": args.pipeline_with_attention,
        "freeze_image_model": args.freeze_image_model,
        "eval_from_generate": args.eval_from_generate,
        "bs": args.bs,
        "epochs": args.epochs,
        "checkpoint": args.checkpoint,
        "augmentation": args.augmentation,
        "swap": args.swap,
        "use_heatmaps": args.use_heatmaps,
        "decoder_length": args.decoder_length,
        "encoder_decoder_path": args.encoder_decoder_path,
        "encoder_path": args.encoder_path,
        "tokenizer_path": args.tokenizer_path,
        "eval_json_path": args.eval_json_path,
        "train_json_path": args.train_json_path,
        "decoder_path": args.decoder_path,
        "label_smoothing": args.label_smoothing,

        "br_th": [2, 2, 2, 2, 2, 2, 2],
        "use_hdf5": False,
        "image_path": '/home/yaxen',
    }

    train(parameters)


if __name__ == "__main__":
    main()
