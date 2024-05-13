# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

# import debugpy
# # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
# debugpy.listen(("localhost", 16823))
# print("Waiting for debugger attach")
# debugpy.wait_for_client()

import torch

import transformers

from videollava.constants import *
from torch.utils.data import Dataset
from videollava.train.llava_trainer import LLaVATrainer

from videollava import conversation as conversation_lib
from videollava.model import *
from videollava.mm_utils import tokenizer_image_token, get_model_name_from_path
from videollava.model.builder import load_pretrained_model

from PIL import Image
from videollava.utils import order_pick_k



local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")

    # ===================================================================
    image_tower: Optional[str] = field(default=None)
    video_tower: Optional[str] = field(default=None)
    audio_tower: Optional[str] = field(default=None)
    depth_tower: Optional[str] = field(default=None)
    thermal_tower: Optional[str] = field(default=None)
    # ===================================================================

@dataclass
class DataArguments:
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    # ===================================================================
    data_path: Optional[List[str]] = field(default=None, metadata={"help": "Path to the training data."})
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    audio_folder: Optional[str] = field(default=None)
    depth_folder: Optional[str] = field(default=None)
    thermal_folder: Optional[str] = field(default=None)
    electromagnetic_wave_folder: Optional[str] = field(default=None)
    num_frames: int = 8
    # ===================================================================


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    # ================================================
    tokenizer_model_max_length: Optional[int] = None
    # ================================================

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:

            # ======================================================================================================
            if sentence['value'].startswith(DEFAULT_IMAGE_TOKEN) \
                or sentence['value'].startswith(DEFAULT_VIDEO_TOKEN) \
                or sentence['value'].startswith(DEFAULT_AUDIO_TOKEN) \
                or sentence['value'].startswith(DEFAULT_DEPTH_TOKEN) \
                or sentence['value'].startswith(DEFAULT_THERMAL_TOKEN) \
                or sentence['value'].startswith(DEFAULT_ELECTROMAGNETIC_WAVE_TOKEN):

                if "mmtag" in conversation_lib.default_conversation.version:
                    pass

                IMAGE_TOKEN_NUM = sentence['value'].count(DEFAULT_IMAGE_TOKEN)
                if IMAGE_TOKEN_NUM > MAX_IMAGE_LENGTH:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN * IMAGE_TOKEN_NUM, DEFAULT_IMAGE_TOKEN * MAX_IMAGE_LENGTH).strip()
                VIDEO_TOKEN_NUM = sentence['value'].count(DEFAULT_VIDEO_TOKEN)
                if VIDEO_TOKEN_NUM > MAX_VIDEO_LENGTH:
                    raise ValueError(f"{sentence['value']}")
                    sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN * VIDEO_TOKEN_NUM, DEFAULT_VIDEO_TOKEN * MAX_VIDEO_LENGTH).strip()

            # a <video> is treated as `num_frames * <image>`
            image_replace_token = DEFAULT_IMAGE_TOKEN
            video_replace_token = DEFAULT_IMAGE_TOKEN * data_args.num_frames
            audio_replace_token = DEFAULT_AUDIO_TOKEN
            depth_replace_token = DEFAULT_THERMAL_TOKEN
            thermal_replace_token = DEFAULT_THERMAL_TOKEN
            electromagnetic_wave_replace_token = DEFAULT_ELECTROMAGNETIC_WAVE_TOKEN
            if data_args.mm_use_im_start_end:
                image_replace_token = DEFAULT_IMAGE_START_TOKEN + image_replace_token + DEFAULT_IMAGE_END_TOKEN
                video_replace_token = DEFAULT_VIDEO_START_TOKEN + video_replace_token + DEFAULT_VIDEO_END_TOKEN
                audio_replace_token = DEFAULT_AUDIO_START_TOKEN + audio_replace_token + DEFAULT_AUDIO_END_TOKEN
                depth_replace_token = DEFAULT_DEPTH_START_TOKEN + depth_replace_token + DEFAULT_DEPTH_END_TOKEN
                thermal_replace_token = DEFAULT_THERMAL_START_TOKEN + thermal_replace_token + DEFAULT_THERMAL_END_TOKEN
                electromagnetic_wave_replace_token = DEFAULT_ELECTROMAGNETIC_WAVE_START_TOKEN + electromagnetic_wave_replace_token + DEFAULT_ELECTROMAGNETIC_WAVE_END_TOKEN

            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, image_replace_token)
            sentence["value"] = sentence["value"].replace(DEFAULT_VIDEO_TOKEN, video_replace_token)
            sentence["value"] = sentence["value"].replace(DEFAULT_AUDIO_TOKEN, audio_replace_token)
            sentence["value"] = sentence["value"].replace(DEFAULT_DEPTH_TOKEN, depth_replace_token)
            sentence["value"] = sentence["value"].replace(DEFAULT_THERMAL_TOKEN, thermal_replace_token)
            sentence["value"] = sentence["value"].replace(DEFAULT_ELECTROMAGNETIC_WAVE_TOKEN, electromagnetic_wave_replace_token)
            # ======================================================================================================

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    pass


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    multimodal_mode: str = None,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    multimodal_tokens={
        'image': DEFAULT_IMAGE_TOKEN,
        'video': DEFAULT_VIDEO_TOKEN,
        'audio': DEFAULT_AUDIO_TOKEN,
        'depth': DEFAULT_DEPTH_TOKEN,
        'thermal': DEFAULT_THERMAL_TOKEN,
        'electromagnetic_wave': DEFAULT_ELECTROMAGNETIC_WAVE_TOKEN,
    }
    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations
    assert (multimodal_mode==None) == (has_image==False)

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, image_token=multimodal_tokens[multimodal_mode], return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    input_ids=[]
    targets=[]
    for conversation in conversations:
        one_input_ids=[tokenizer.bos_token_id]
        one_targets=[IGNORE_INDEX]

        rounds = conversation.split(conv.sep2)
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            parts[1] += conv.sep2

            if has_image:
                part0_ids=tokenizer_image_token(parts[0], tokenizer, image_token=multimodal_tokens[multimodal_mode])[1:]
                part1_ids=tokenizer_image_token(parts[1], tokenizer, image_token=multimodal_tokens[multimodal_mode])[1:]
            else:
                part0_ids=tokenizer(parts[0]).input_ids[1:]
                part1_ids=tokenizer(parts[1]).input_ids[1:]

            one_input_ids+=part0_ids
            one_targets+=[IGNORE_INDEX]*len(part0_ids)
            
            one_input_ids+=part1_ids
            one_targets+=part1_ids

        input_ids.append(one_input_ids)
        targets.append(one_targets)
    input_ids=torch.tensor(input_ids)
    targets=torch.tensor(targets)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    pass


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    pass


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    multimodal_mode: str = None,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
    #     return preprocess_plain(sources, tokenizer)
    # if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
    #     return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    # if conversation_lib.default_conversation.version == "mpt":
    #     return preprocess_mpt(sources, tokenizer)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, multimodal_mode=multimodal_mode, has_image=has_image)
    
def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        # ================================================
        list_data_dict = []
        data_list=json.load(open(data_path[0], "r"))['sft_data']
        for data in data_list:
            data = json.load(open(data, "r"))
            for i in data:
                i['id'] = len(list_data_dict)
                list_data_dict.append(i)
        # ================================================

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    # @property
    # def lengths(self):
    #     length_list = []
    #     for sample in self.list_data_dict:
    #         img_tokens = 128 if 'image' in sample else 0
    #         length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
    #     return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            # ===========================================================================
            cur_len = cur_len if ('image' in sample or 'video' in sample or 'audio' in sample or 'depth' in sample or 'thermal' in sample or 'electromagnetic_wave' in sample) else -cur_len
            # ===========================================================================
            length_list.append(cur_len)
        return length_list
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        # ======================================================================================================
        if sources[0]['multimodal_mode']=="none":  # only text
            sources = copy.deepcopy([e["conversations"] for e in sources])
            data_dict = preprocess(sources, self.tokenizer, has_image=False)
        elif sources[0]['multimodal_mode']=='image':
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            image_processor = self.data_args.image_processor
            image_file = image_file if isinstance(image_file, list) else [image_file]
            image_file = order_pick_k(image_file, MAX_IMAGE_LENGTH)
            # print(f"total {len(self.list_data_dict[i]['image'])} now {len(image_file)}")
            image = [Image.open(os.path.join(image_folder, file)).convert('RGB') for file in image_file]
            if self.data_args.image_aspect_ratio == 'pad':
                image = [expand2square(i, tuple(int(x * 255) for x in image_processor.image_mean)) for i in image]
            try:
                image = [image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in image]
            except Exception as e:
                logging.warning(f'load image file error: ',e)
                image = [torch.zeros((3, 224, 224), dtype=torch.float32) for i in image]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            data_dict = preprocess(sources, self.tokenizer, multimodal_mode='image', has_image=True)

        elif sources[0]['multimodal_mode']=='video':
            video_file = self.list_data_dict[i]['video']
            video_folder = self.data_args.video_folder
            video_processor = self.data_args.video_processor
            video_file = video_file if isinstance(video_file, list) else [video_file]
            video_file = order_pick_k(video_file, MAX_VIDEO_LENGTH)
            video = [os.path.join(video_folder, file) for file in video_file]
            try:
                image = [video_processor(i, return_tensors='pt')['pixel_values'][0] for i in video]  # fake image
            except Exception as e:
                logging.warning(f'load video file error: ',e)
                image = [torch.zeros((3, 8, 224, 224), dtype=torch.float32) for i in video]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            data_dict = preprocess(sources, self.tokenizer, multimodal_mode='video', has_image=True)

        elif sources[0]['multimodal_mode']=='audio':
            audio_file = self.list_data_dict[i]['audio']
            audio_folder = self.data_args.audio_folder
            audio_processor = self.data_args.audio_processor
            audio_file = audio_file if isinstance(audio_file, list) else [audio_file]
            audio_file = order_pick_k(audio_file, MAX_AUDIO_LENGTH)
            audio = [os.path.join(audio_folder, file) for file in audio_file]
            try:
                image = [audio_processor(i, return_tensors='pt')['pixel_values'][0] for i in audio]
            except Exception as e:
                logging.warning(f'load audio file error: ',e)
                image = [torch.zeros((3, 126, 1036), dtype=torch.float32) for i in audio]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            data_dict = preprocess(sources, self.tokenizer, multimodal_mode='audio', has_image=True)

        elif sources[0]['multimodal_mode']=='depth':
            depth_file = self.list_data_dict[i]['depth']
            depth_folder = self.data_args.depth_folder
            depth_processor = self.data_args.depth_processor
            depth_file = depth_file if isinstance(depth_file, list) else [depth_file]
            depth_file = order_pick_k(depth_file, MAX_DEPTH_LENGTH)
            depth = [os.path.join(depth_folder, file) for file in depth_file]
            try:
                image = [depth_processor(i, return_tensors='pt')['pixel_values'][0] for i in depth]
            except Exception as e:
                logging.warning(f'load depth file error: ',e)
                image = [torch.zeros((3, 224, 224), dtype=torch.float32) for i in depth]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            data_dict = preprocess(sources, self.tokenizer, multimodal_mode='depth', has_image=True)

        elif sources[0]['multimodal_mode']=='thermal':
            thermal_file = self.list_data_dict[i]['thermal']
            thermal_folder = self.data_args.thermal_folder
            thermal_processor = self.data_args.thermal_processor
            thermal_file = thermal_file if isinstance(thermal_file, list) else [thermal_file]
            thermal_file = order_pick_k(thermal_file, MAX_THERMAL_LENGTH)
            thermal = [os.path.join(thermal_folder, file) for file in thermal_file]
            try:
                image = [thermal_processor(i, return_tensors='pt')['pixel_values'][0] for i in thermal]  # fake image
            except Exception as e:
                logging.warning(f'load thermal file error: ',e)
                image = [torch.zeros((3, 224, 224), dtype=torch.float32) for i in thermal]
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            data_dict = preprocess(sources, self.tokenizer, multimodal_mode='thermal', has_image=True)
        elif sources[0]['multimodal_mode']=='electromagnetic_wave':
            electromagnetic_wave_folder = self.data_args.electromagnetic_wave_folder
            with open(os.path.join(electromagnetic_wave_folder, self.list_data_dict[i]['electromagnetic_wave']), 'r', encoding='utf-8') as f:
                image = json.load(f)
            image = image + [0] * (self.data_args.mm_hidden_size-len(image))
            image = torch.tensor([[image]], dtype=torch.float32)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            data_dict = preprocess(sources, self.tokenizer, multimodal_mode='electromagnetic_wave', has_image=True)
        # ==========================================================================================================

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                                labels=data_dict["labels"][0])
        # image exist in the data

        multimodal_mode_index_map={
            'image': 1,
            'video': 2,
            'audio': 3,
            'depth': 4,
            'thermal': 5,
            'electromagnetic_wave': 6,
        }

        if self.list_data_dict[i]['multimodal_mode'] !='none':
            data_dict['image'] = image
            data_dict['multimodal_mode'] = torch.tensor(multimodal_mode_index_map[self.list_data_dict[i]['multimodal_mode']])
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            crop_size = {'height': 224, 'width': 224}  # dummy image
            data_dict['image'] = [torch.zeros(3, crop_size['height'], crop_size['width'])]
            data_dict['multimodal_mode'] = torch.tensor(1)
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        # ======================================================================================================
        # origin image, if batch_size=6: [[image], [image], [video], [image, image], [video, video], [video, image]]
        '''
            will be converted to a sequence of list, if batch size=6:
            [
                image(3, 224, 224),      # sample 1
                image(3, 224, 224),      # sample 2
                video(8, 3, 224, 224),   # sample 3
                image(3, 224, 224),      # sample 4
                image(3, 224, 224),      # sample 4
                video(8, 3, 224, 224),   # sample 5
                video(8, 3, 224, 224),   # sample 5
                video(8, 3, 224, 224),   # sample 6
                image(3, 224, 224),      # sample 6
            ]
        '''
        if 'image' in instances[0]:
            images = [instance['image'][0] for instance in instances]
            multimodal_modes = [instance['multimodal_mode'] for instance in instances]

            # # adapt to multi-video or multi-image or multi-image & video
            # new_images = []
            # for image in images:
            #     if type(image) is list:
            #         for i in image:
            #             new_images.append(i)
            #     else:
            #         new_images.append(image)
            # images = new_images

        # ==========Too many videos or images may lead to OOM, so we encode them one by one======================
            batch['images'] = images
            batch['multimodal_modes'] = multimodal_modes
        #     if all(x is not None and x.shape == images[0].shape for x in images):  # if all images or all videos
        #         batch['images'] = torch.stack(images)
        #     else:
        #         batch['images'] = images
        else:
            raise ValueError(f'pretrain, {instances}')
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    # ==========================================================================
    if model_args.image_tower is not None or model_args.video_tower is not None or model_args.audio_tower is not None or model_args.depth_tower is not None or model_args.thermal_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    # =============================================================================================================
    if model_args.image_tower is not None or model_args.video_tower is not None or model_args.audio_tower is not None or model_args.depth_tower is not None or model_args.thermal_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        if model_args.image_tower is not None:
            image_tower = model.get_image_tower()
            image_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            data_args.image_processor = image_tower.image_processor
            data_args.is_multimodal = True
        if model_args.video_tower is not None:
            video_tower = model.get_video_tower()
            video_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            data_args.video_processor = video_tower.video_processor
            data_args.is_multimodal = True
            data_args.num_frames = video_tower.config.num_frames
        if model_args.audio_tower is not None:
            audio_tower = model.get_audio_tower()
            audio_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            data_args.audio_processor = audio_tower.audio_processor
            data_args.is_multimodal = True
        if model_args.depth_tower is not None:
            depth_tower = model.get_depth_tower()
            depth_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            data_args.depth_processor = depth_tower.depth_processor
            data_args.is_multimodal = True
        if model_args.thermal_tower is not None:
            thermal_tower = model.get_thermal_tower()
            thermal_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
            data_args.thermal_processor = thermal_tower.thermal_processor
            data_args.is_multimodal = True
    # =============================================================================================================


        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side

        # =============================================================================================================
        tokenizer_model_max_length = training_args.tokenizer_model_max_length
        model.config.tokenizer_model_max_length = tokenizer.model_max_length if tokenizer_model_max_length is None else tokenizer_model_max_length
        # =============================================================================================================
        
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        data_args.mm_hidden_size = model.config.mm_hidden_size
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
