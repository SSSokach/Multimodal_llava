#    Copyright 2023 Haotian Liu
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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import *
from .multimodal_projector.builder import build_vision_projector

from videollava.constants import *


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        self.encoder_tower_dict={}
        if getattr(config, "mm_image_tower", None) is not None:
            self.image_tower = build_image_tower(config, delay_load=True)
            if self.image_tower is not None:
                self.encoder_tower_dict['image']=self.image_tower
        if getattr(config, "mm_video_tower", None) is not None:
            self.video_tower = build_video_tower(config, delay_load=True)
            if self.video_tower is not None:
                self.encoder_tower_dict['video']=self.video_tower
        if getattr(config, "mm_audio_tower", None) is not None:
            self.audio_tower = build_audio_tower(config, delay_load=True)
            if self.audio_tower is not None:
                self.encoder_tower_dict['audio']=self.audio_tower
        if getattr(config, "mm_depth_tower", None) is not None:
            self.depth_tower = build_depth_tower(config, delay_load=True)
            if self.depth_tower is not None:
                self.encoder_tower_dict['depth']=self.depth_tower
        if getattr(config, "mm_thermal_tower", None) is not None:
            self.thermal_tower = build_thermal_tower(config, delay_load=True)
            if self.thermal_tower is not None:
                self.encoder_tower_dict['thermal']=self.thermal_tower
        if getattr(config, "mm_image_tower", None) is not None or \
            getattr(config, "mm_video_tower", None) is not None or \
            getattr(config, "mm_audio_tower", None) is not None or \
            getattr(config, "mm_depth_tower", None) is not None or \
            getattr(config, "mm_thermal_tower", None) is not None:
            self.mm_projector = build_vision_projector(config)

    def get_image_tower(self):
        image_tower = getattr(self, 'image_tower', None)
        if type(image_tower) is list:
            image_tower = image_tower[0]
        return image_tower

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower
    
    def get_audio_tower(self):
        audio_tower = getattr(self, 'audio_tower', None)
        if type(audio_tower) is list:
            audio_tower = audio_tower[0]
        return audio_tower

    def get_depth_tower(self):
        depth_tower = getattr(self, 'depth_tower', None)
        if type(depth_tower) is list:
            depth_tower = depth_tower[0]
        return depth_tower

    def get_thermal_tower(self):
        thermal_tower = getattr(self, 'thermal_tower', None)
        if type(thermal_tower) is list:
            thermal_tower = thermal_tower[0]
        return thermal_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        # ==============================================
        image_tower = model_args.image_tower
        video_tower = model_args.video_tower
        audio_tower = model_args.audio_tower
        depth_tower = model_args.depth_tower
        thermal_tower = model_args.thermal_tower
        assert model_args.image_tower is not None or model_args.video_tower is not None or model_args.audio_tower is not None or model_args.depth_tower is not None or model_args.thermal_tower is not None
        # ==============================================
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter

        # ==========================================================================

        self.config.mm_image_tower = image_tower
        if image_tower is not None:
            if self.get_image_tower() is None:
                image_tower = build_image_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.image_tower = [image_tower]
                else:
                    self.image_tower = image_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    image_tower = self.image_tower[0]
                else:
                    image_tower = self.image_tower
                image_tower.load_model()

        self.config.mm_video_tower = video_tower
        if video_tower is not None:
            if self.get_video_tower() is None:
                video_tower = build_video_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.video_tower = [video_tower]
                else:
                    self.video_tower = video_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    video_tower = self.video_tower[0]
                else:
                    video_tower = self.video_tower
                video_tower.load_model()

        self.config.mm_audio_tower = audio_tower
        if audio_tower is not None:
            if self.get_audio_tower() is None:
                audio_tower = build_audio_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.audio_tower = [audio_tower]
                else:
                    self.audio_tower = audio_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    audio_tower = self.audio_tower[0]
                else:
                    audio_tower = self.audio_tower
                audio_tower.load_model()

        self.config.mm_depth_tower = depth_tower
        if depth_tower is not None:
            if self.get_depth_tower() is None:
                depth_tower = build_depth_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.depth_tower = [depth_tower]
                else:
                    self.depth_tower = depth_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    depth_tower = self.depth_tower[0]
                else:
                    depth_tower = self.depth_tower
                depth_tower.load_model()

        self.config.mm_thermal_tower = thermal_tower
        if thermal_tower is not None:
            if self.get_thermal_tower() is None:
                thermal_tower = build_thermal_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.thermal_tower = [thermal_tower]
                else:
                    self.thermal_tower = thermal_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    thermal_tower = self.thermal_tower[0]
                else:
                    thermal_tower = self.thermal_tower
                thermal_tower.load_model()

        # ==========================================================================

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        # ==========================================================================
        if image_tower is not None or video_tower is not None or audio_tower is not None or depth_tower is not None or thermal_tower is not None:
            self.config.mm_hidden_size = max(getattr(image_tower, 'hidden_size', -1),
                                             getattr(video_tower, 'hidden_size', -1),
                                             getattr(audio_tower, 'hidden_size', -1),
                                             getattr(depth_tower, 'hidden_size', -1),
                                             getattr(thermal_tower, 'hidden_size', -1))
        # ===================================================================================

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_image_tower(self):
        return self.get_model().get_image_tower()

    def get_video_tower(self):
        return self.get_model().get_video_tower()
    
    def get_audio_tower(self):
        return self.get_model().get_audio_tower()

    def get_depth_tower(self):
        return self.get_model().get_depth_tower()

    def get_thermal_tower(self):
        return self.get_model().get_thermal_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_image_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_videos(self, videos):  # [mini_b, c, t, h, w]
        b, _, t, _, _ = videos.shape
        video_features = self.get_model().get_video_tower()(videos)  # [mini_b, t, n, c]
        video_features = self.get_model().mm_projector(video_features)
        return video_features
    
    def encode_audios(self, audios):
        audio_features = self.get_model().get_audio_tower()(audios)
        audio_features = self.get_model().mm_projector(audio_features)
        return audio_features

    def encode_depths(self, depths):
        depth_features = self.get_model().get_depth_tower()(depths)
        depth_features = self.get_model().mm_projector(depth_features)
        return depth_features

    def encode_thermals(self, thermals):
        thermal_features = self.get_model().get_thermal_tower()(thermals)
        thermal_features = self.get_model().mm_projector(thermal_features)
        return thermal_features

    def encode_electromagnetic_waves(self, electromagnetic_waves):
        electromagnetic_wave_features = self.get_model().mm_projector(electromagnetic_waves)
        return electromagnetic_wave_features
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, multimodal_modes
    ):
        # ====================================================================================================
        image_tower = self.get_image_tower()
        video_tower = self.get_video_tower()
        audio_tower = self.get_audio_tower()
        depth_tower = self.get_depth_tower()
        thermal_tower = self.get_thermal_tower()
        if images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and (image_tower is not None or video_tower is not None) and images is not None and input_ids.shape[1] == 1:
                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        multimodal_modes=[one.item() for one in multimodal_modes]
        image_idx = [idx for idx, img_mode in enumerate(multimodal_modes) if img_mode == 1]
        video_idx = [idx for idx, img_mode in enumerate(multimodal_modes) if img_mode == 2]
        audio_idx = [idx for idx, img_mode in enumerate(multimodal_modes) if img_mode == 3]
        depth_idx = [idx for idx, img_mode in enumerate(multimodal_modes) if img_mode == 4]
        thermal_idx = [idx for idx, img_mode in enumerate(multimodal_modes) if img_mode == 5]
        electromagnetic_wave_idx = [idx for idx, img_mode in enumerate(multimodal_modes) if img_mode == 6]

        images_minibatch = torch.stack([images[idx] for idx in image_idx]) if len(image_idx) > 0 else []
        videos_minibatch = torch.stack([images[idx] for idx in video_idx]) if len(video_idx) > 0 else []
        audios_minibatch = torch.stack([images[idx] for idx in audio_idx]) if len(audio_idx) > 0 else []
        depths_minibatch = torch.stack([images[idx] for idx in depth_idx]) if len(depth_idx) > 0 else []
        thermals_minibatch = torch.stack([images[idx] for idx in thermal_idx]) if len(thermal_idx) > 0 else []
        electromagnetic_waves_minibatch = torch.stack([images[idx] for idx in electromagnetic_wave_idx]) if len(electromagnetic_wave_idx) > 0 else []

        tmp_image_features = [None] * (len(image_idx) + len(video_idx) + len(audio_idx) + len(depth_idx) + len(thermal_idx) + len(electromagnetic_wave_idx))
        if getattr(images_minibatch, 'ndim', 0) > 0:  # batch consists of images, [mini_b, c, h, w]
            image_features_minibatch = self.encode_images(images_minibatch)  # [mini_b, l, c]
            for i, pos in enumerate(image_idx):
                tmp_image_features[pos] = image_features_minibatch[i]

        if getattr(videos_minibatch, 'ndim', 0) > 0:  # batch consists of videos, [mini_b, c, t, h, w]
            video_features_minibatch = self.encode_videos(videos_minibatch)  # fake list [mini_b, t, l, c]
            for i, pos in enumerate(video_idx):
                t = video_features_minibatch[i].shape[0]
                tmp_image_features[pos] = [video_features_minibatch[i][j] for j in range(t)]
        
        if getattr(audios_minibatch, 'ndim', 0) > 0:
            audio_features_minibatch = self.encode_audios(audios_minibatch)
            for i, pos in enumerate(audio_idx):
                tmp_image_features[pos] = audio_features_minibatch[i]

        if getattr(depths_minibatch, 'ndim', 0) > 0:
            depth_features_minibatch = self.encode_depths(depths_minibatch)
            for i, pos in enumerate(depth_idx):
                tmp_image_features[pos] = depth_features_minibatch[i]

        if getattr(thermals_minibatch, 'ndim', 0) > 0:
            thermal_features_minibatch = self.encode_thermals(thermals_minibatch)
            for i, pos in enumerate(thermal_idx):
                tmp_image_features[pos] = thermal_features_minibatch[i]
                        
        if getattr(electromagnetic_waves_minibatch, 'ndim', 0) > 0:
            electromagnetic_wave_features_minibatch = self.encode_electromagnetic_waves(electromagnetic_waves_minibatch)
            for i, pos in enumerate(electromagnetic_wave_idx):
                tmp_image_features[pos] = electromagnetic_wave_features_minibatch[i]

        new_tmp = []
        for image in tmp_image_features:
            # print(len(new_tmp), len(image))
            if isinstance(image, list):
                t = len(image)
                for i in range(t):
                    new_tmp.append(image[i])
                # print('add video')
            else:
                new_tmp.append(image)
        image_features = new_tmp
        # print(len(image_features), *[i.shape for i in image_features])
        # print(len(image_features), image_features[0].shape)
        # ====================================================================================================

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- TODO: double check
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            # print(num_images, cur_input_ids)
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    # print(cur_image_idx)
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_DEPTH_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_THERMAL_PATCH_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_ELECTROMAGNETIC_WAVE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            old_tokenizer_length=len(tokenizer)
            tokenizer.add_tokens([DEFAULT_IMAGE_START_TOKEN, DEFAULT_IMAGE_END_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_VIDEO_START_TOKEN, DEFAULT_VIDEO_END_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_AUDIO_START_TOKEN, DEFAULT_AUDIO_END_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_DEPTH_START_TOKEN, DEFAULT_DEPTH_END_TOKEN], special_tokens=True)
            tokenizer.add_tokens([DEFAULT_ELECTROMAGNETIC_WAVE_START_TOKEN, DEFAULT_ELECTROMAGNETIC_WAVE_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            num_new_tokens=len(tokenizer)-old_tokenizer_length
            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
