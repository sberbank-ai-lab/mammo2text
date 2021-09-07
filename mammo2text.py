import os, sys
sys.path.append(os.path.expanduser("~")+"/dotfiles")
import myutils

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_outputs import ModelOutput
from torch.utils.data import Dataset, DataLoader

import transformers
from transformers import *
from datasets import *

from torchvision.transforms import transforms
from PIL import Image

import models
from constants import VIEWS, VIEWANGLES, LABELS, MODELMODES
import re


def preprocess(t):
    t = t.replace('\n', '.')
    t = t.replace('..', '.') 
    ## trim spaces inside
    t = " ".join(t.split())
    ## delete numbers e.g.: Маммограммы (4 проекции) №2717. На
    t = re.sub('(№[0-9]+)', '', t)
    return t


def get_target_text(datum):
    # [SEP] is used as eos token
    text = str(datum['birads']) + '. Заключение: ' + datum['conclusion'] + '. Протокол: ' + datum[
        'protocol'] + '. '
    text = preprocess(text)
    return text


class BreastImgTextDataset(Dataset):
    def __init__(self, 
                 exam_list,
                 tokenizer,
                 parameters, 
                 random_number_generator, 
                 swap=False):
        self.image_extension = ".hdf5" if parameters["use_hdf5"] else "" #".png"
        self.exam_list = exam_list
        self.parameters = parameters
        self.random_number_generator = random_number_generator
        self.swap = swap

        self.tr = {"L-CC": "R-CC", "L-MLO": "R-MLO", "R-CC": "L-CC", "R-MLO": "L-MLO"}
        self.transforms = transforms.Compose([
            transforms.RandomRotation((-5, 5)),
            transforms.RandomResizedCrop((1350, 900), scale=(0.64, 1.00), ratio=(0.5, 2.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.01, 0.1), ratio=(0.5, 2.0)),
        ])
        self.eval_transforms = transforms.Compose([
            transforms.Resize((1350, 900)),
            transforms.ToTensor()
        ])
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.exam_list)

    def __getitem__(self, idx):
        datum = self.exam_list[idx]
        loaded_image_dict = {view: [] for view in VIEWS.LIST}

        for orig_view in VIEWS.LIST:
            if len(datum[orig_view]) > 0:
                view = orig_view
            else:
                view = self.tr[orig_view]
            assert view == orig_view # for 4 view training
            for short_file_path in datum[view]:
                loaded_image = Image.open(os.path.join(self.parameters["image_path"], short_file_path + self.image_extension))
                loaded_image_dict[orig_view].append(loaded_image)

        batch_dict = {view: [] for view in VIEWS.LIST}
        for view in VIEWS.LIST:
            image_index = 0
            if self.parameters["augmentation"]:
                image_index = self.random_number_generator.randint(low=0, high=len(datum[view]))
                img = self.transforms(loaded_image_dict[view][image_index])
            else:
                img = self.eval_transforms(loaded_image_dict[view][image_index])
            batch_dict[view] = img / torch.std(img)
            
        # TEXT
        if 'conclusion' in datum:
            text = get_target_text(datum)
            outputs = self.tokenizer(text, padding=False, truncation=True, max_length=self.parameters['decoder_length'], return_tensors='pt')
            batch_dict["decoder_input_ids"] = outputs.input_ids[0]
            batch_dict["labels"] = batch_dict["decoder_input_ids"].detach().clone()
            batch_dict["decoder_attention_mask"] = outputs.attention_mask[0]
            batch_dict["birads"] = datum['birads']
        
        if self.swap and bool(self.random_number_generator.randint(low=0, high=2)):
            batch_dict[VIEWS.R_CC], batch_dict[VIEWS.L_CC] = batch_dict[VIEWS.L_CC], batch_dict[VIEWS.R_CC]
            batch_dict[VIEWS.R_MLO], batch_dict[VIEWS.L_MLO] = batch_dict[VIEWS.L_MLO], batch_dict[VIEWS.R_MLO]
        return batch_dict



class MammoImgTextDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def _pad_according_to_type(self, features, k, padding_value):
        if isinstance(features, list):
            res = pad_sequence([f[k] for f in features], batch_first=True, padding_value=padding_value)
        elif isinstance(features, dict):
            res = pad_sequence(features[k], batch_first=True, padding_value=padding_value)
        else:
            raise
        return res

    def __call__(self, features):
        batch = {}
        batch['input_ids'] = {}
        for k in VIEWS.LIST:
            view = self._pad_according_to_type(features, k, 0)
            batch['input_ids'][k] = view

        # inference
        if isinstance(features, dict) and 'labels' not in features:
            return batch
        if isinstance(features, list) and 'labels' not in features[0]:
            return batch
        
        ## CrossEntropyLoss ignores index -100
        for k in ['labels']:
            batch[k] = self._pad_according_to_type(features, k, -100)
        for k in ['decoder_input_ids']:
            batch[k] = self._pad_according_to_type(features, k, self.tokenizer.pad_token_id)
        for k in ['decoder_attention_mask']:
            batch[k] = self._pad_according_to_type(features, k, 0)
        return batch

    
# ========= MODEL ========= 

class EncoderWrapper(torch.nn.Module):
    def __init__(self, model, hidden_size, word_emb_size, device):
        super().__init__()
        self.model = model
        self.fc = torch.nn.Linear(hidden_size, word_emb_size)
        self.device = device

    def forward(self, input_ids=None, **kwargs):
        for k in input_ids.keys():
            input_ids[k] = input_ids[k].to(self.device)
        output = self.model(input_ids)
        output = self.fc(output)

        return ModelOutput(
            last_hidden_state=output,
            hidden_states=None,
            attentions=None
        )

    def get_output_embeddings(self):
        return None
    

def check_tokenizer(tokenizer):
    if tokenizer.bos_token is None:
        print(f'Setting tokenizer.bos_token to {tokenizer.cls_token}')
        tokenizer.bos_token = tokenizer.cls_token
    if tokenizer.eos_token is None:
        print(f'Setting tokenizer.eos_token to {tokenizer.sep_token}')
        tokenizer.eos_token = tokenizer.sep_token


def change_config_tokens(config, tokenizer):
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.pad_token_id = tokenizer.pad_token_id
    config.sep_token_id = tokenizer.sep_token_id


def four_view_custom_forward(self, x):
    h = self.all_views_gaussian_noise_layer(x)
    h = self.four_view_net(h)

    hh = torch.stack([h[VIEWS.L_CC], h[VIEWS.R_CC], h[VIEWS.L_MLO], h[VIEWS.R_MLO]], -1)
    n, c, _ = hh.size()
    hh = hh.view(n, c, -1).mean(-1) # (4x1000)
    ## add dimension so that it will be a batch of sequences of length 1
    hh = torch.unsqueeze(hh, 1)
    #     hh = self.fc(hh)
    return hh


def four_view_custom_forward_with_maps(self, x):
    h = self.all_views_gaussian_noise_layer(x)
    h = self.four_view_net(h)

    hh = torch.stack([h[VIEWS.L_CC], h[VIEWS.R_CC], h[VIEWS.L_MLO], h[VIEWS.R_MLO]], -1) #[5,1280,43,29,4]
    bs, channels, height, width, n_pictures = hh.size() # 5, 1280, 43, 29, 4
    hh = hh.view(bs, channels, -1)
    hh = hh.permute(0, 2, 1)
    # hh = self.fc(hh)
    return hh


def forward_feature_maps(self, inputs):
    x = self.extract_features(inputs)
    # Pooling and final linear layer
    # x = self._avg_pooling(x)
    # if self._global_params.include_top:
    #     x = x.flatten(start_dim=1)
    #     x = self._dropout(x)
    #     x = self._fc(x)
    return x


def load_image_model(parameters, device):
    input_channels = 3 if parameters["use_heatmaps"] else 1
    
    if parameters["pipeline_with_attention"]:
        from efficientnet_pytorch import EfficientNet
        EfficientNet.forward = forward_feature_maps
        models.FourViewEfficientNetOnly.forward = four_view_custom_forward_with_maps
        hidden_size = 1280 # number of channels
    else:
        models.FourViewEfficientNetOnly.forward = four_view_custom_forward
        hidden_size = 1000
    model = models.FourViewEfficientNetOnly(input_channels, len(parameters["br_th"]))

    model.load_state_dict(torch.load(parameters["encoder_path"], map_location=device))
    model = model.to(device)
    return model, hidden_size


def get_decoder(decoder_path, debug=False):
    # decoder = BertLMHeadModel.from_pretrained(parameters['decoder_path'])
    decoder_config = AutoConfig.from_pretrained(decoder_path)
    decoder_config.is_decoder = True
    decoder_config.add_cross_attention = True
    if debug:
        decoder_config.hidden_dropout_prob = 0.0
        decoder_config.attention_probs_dropout_prob = 0.0
    decoder = AutoModelForCausalLM.from_pretrained(decoder_path, config=decoder_config)
    return decoder


def get_mammo2text_model(parameters, tokenizer, encoder):
    # longformer2rubert_mlm_256 = EncoderDecoderModel.from_pretrained(parameters['encoder_decoder_path'])
    config = AutoConfig.from_pretrained(parameters['encoder_decoder_path'])
    config.decoder_length = parameters['decoder_length']
    config.length_penalty = 1.0
    config.max_length = parameters['decoder_length']
    # get decoder
    if 'decoder_path' in parameters and parameters['decoder_path'] is not None:
        decoder = get_decoder(parameters['decoder_path'], parameters['debug'])
    else:
        longformer2rubert_mlm_256 = EncoderDecoderModel.from_pretrained(parameters['encoder_decoder_path'])
        decoder = longformer2rubert_mlm_256.decoder
    assert(decoder.get_input_embeddings().num_embeddings == len(tokenizer))

    # build encoder-decoder
    model = EncoderDecoderModel(config=config, encoder=encoder, decoder=decoder)

    change_config_tokens(model.decoder.config, tokenizer)
    change_config_tokens(model.config, tokenizer)

    model.config.decoder_start_token_id = tokenizer.bos_token_id
    return model


# def generate_one_sample(model, tokenizer, data_collator, eval_dataset, device):
#     sample = eval_dataset[0]
#     sample = data_collator([sample])
#     target, predicted, _, _ = generate(sample, model, tokenizer, data_collator, device)
#     sample = {'i': i, 'target': target, 'predicted': predicted}
#     return sample


def save_eval_predictions(res_df, curr_df, iter_num, excel_path):
    """
    saves predictions on given iteration to 
    :param res_df: dataframe for all iterations
    :param curr_df: dataframe for current iteration 
    :param iter_num: 
    :return: 
    """
    print('\nSaving eval predictions to', excel_path)
    if 'label_str' not in res_df:
        res_df['label_str'] = curr_df['label_str']
    res_df[f"prediction_{iter_num}"] = curr_df[['pred_str']]
    myutils.excel.save(res_df, excel_path, list(res_df.columns))

    
def ignore_tokens_after_eos(a, tokenizer):
    m, n = a.shape
    for i in range(m):
        for j in range(n):
            if a[i, j] == tokenizer.eos_token_id:
                a[i, j + 1:] = tokenizer.pad_token_id
    return a

    
def generate(inputs, model, tokenizer, gen_kwargs={}):
    input_ids = inputs['input_ids']
    # print('gen started')
    # with myutils.MyTimeCatcher() as _:
    output_ids = model.generate(input_ids,
                            attention_mask=torch.ones_like(inputs['decoder_input_ids'][:,:1]),
                            decoder_input_ids=inputs['decoder_input_ids'][:,:1],
                            **gen_kwargs)
    # print('gen ended')
    predicted = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    target = None
    if 'labels' in inputs:
        labels_ids = inputs['labels']
        labels_ids[labels_ids == -100] = tokenizer.eos_token_id
        target = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    return target, predicted, input_ids, output_ids


def generate_predictions(model, tokenizer, dataloader_for_gen, gen_kwargs={}):
    device = model.device
    res = []
    for inputs in tqdm(dataloader_for_gen):
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        targets, predicts, _, _ = generate(inputs, model, tokenizer, gen_kwargs)
        for t,p in zip(targets, predicts):
            sample = {'target': t, 'predicted': p}
            res.append(sample)
    return pd.DataFrame(res)



# this cb will be called after MLflowCallback, because it's just added to the list of cbs
class AddParametersToMlflowCallback(TrainerCallback):
    def __init__(self, trainer, parameters, files=None):
        self.trainer = trainer
        self.parameters = parameters
        self.files = files
        self.mlflow_cb = None
        self.run_id = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.mlflow_cb = self.get_mlflow_cb()
        self.run_id = self.mlflow_cb._ml_flow.active_run().info.run_id
        for k,v in self.parameters.items():
            try:
                self.mlflow_cb._ml_flow.log_param('_'+k, v)
            except:
                pass
        if self.files:
            for file in self.files:
                self.mlflow_cb._ml_flow.log_artifact(file)

    def get_mlflow_cb(self):
        from transformers.integrations import MLflowCallback
        mlflow_cb = None
        for cb in self.trainer.callback_handler.callbacks:
            if isinstance(cb, MLflowCallback):
                mlflow_cb = cb
        return mlflow_cb
    
    
def plot_attention(images, words, attention_plot, file_name='viz/attention.png', 
                   cmap='jet', interpolation='spline16', alpha=None, quality_factor=1, 
                   use_add_alpha=True, image_id=0, sum_over_words=False):
    plt.ioff()
    len_words = len(words) 
    len_images = len(images)
    layout_words = len_words+1
    layout_images = len_images
    fig = plt.figure(figsize=(layout_words*1.5*quality_factor, layout_images*3*quality_factor))
    pbar = tqdm(total=len_words*len_images)
    for img_index in range(len_images):
        temp_image = images[img_index]
        i = layout_words*img_index + 1
        ax = fig.add_subplot(layout_images, layout_words, i)
        img = ax.imshow(temp_image, cmap='gray')
        plt.axis('off')
    if sum_over_words:
        attention_plot = attention_plot.sum(0)
        attention_plot = np.expand_dims(attention_plot, 0)
    for word_index in range(len_words):
        word_min = attention_plot[word_index,:,:,:].min()
        word_max = attention_plot[word_index,:,:,:].max()
        for img_index in range(len_images):
            temp_image = images[img_index]
            temp_att = attention_plot[word_index,:,:, img_index]
            i = layout_words*img_index + word_index + 2
            ax = fig.add_subplot(layout_images, layout_words, i)
            plt.axis('off')
            if img_index == 0:
                ax.set_title(words[word_index])
            img = ax.imshow(temp_image, cmap='gray')
            
            p = np.percentile(temp_att, 50)
            temp_att[temp_att < p] = p
            
            _alpha = alpha
            if use_add_alpha:
                _alpha = np.ones(temp_att.shape) * alpha
                scaler = MinMaxScaler((-0.2, 0.1))
                additional_alpha = scaler.fit_transform(temp_att)
                _alpha += additional_alpha

            ax.imshow(temp_att, interpolation=interpolation, cmap=cmap, alpha=_alpha, 
                      extent=img.get_extent(), vmin=word_min, vmax=word_max) #, vmin=word_min, vmax=word_max , vmin=-0.1, vmax=1.1 , vmin=0, vmax=1 # , vmin=temp_att.min(), vmax=temp_att.max()
            pbar.update(1)
    fig.tight_layout()
    _ = plt.savefig(f'viz/{image_id}_att.png', dpi=100)
    pbar.close()
    plt.close(fig)
#     plt.show()