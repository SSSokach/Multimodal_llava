import torch
from torch import nn
from transformers import AutoConfig

from .image.configuration_image import LanguageBindImageConfig
from .image.modeling_image import LanguageBindImage
from .image.tokenization_image import LanguageBindImageTokenizer
from .image.processing_image import LanguageBindImageProcessor

from .video.configuration_video import LanguageBindVideoConfig
from .video.modeling_video import LanguageBindVideo
from .video.tokenization_video import LanguageBindVideoTokenizer
from .video.processing_video import LanguageBindVideoProcessor

from .depth.configuration_depth import LanguageBindDepthConfig
from .depth.modeling_depth import LanguageBindDepth
from .depth.tokenization_depth import LanguageBindDepthTokenizer
from .depth.processing_depth import LanguageBindDepthProcessor

from .audio.configuration_audio import LanguageBindAudioConfig
from .audio.modeling_audio import LanguageBindAudio
from .audio.tokenization_audio import LanguageBindAudioTokenizer
from .audio.processing_audio import LanguageBindAudioProcessor

from .thermal.configuration_thermal import LanguageBindThermalConfig
from .thermal.modeling_thermal import LanguageBindThermal
from .thermal.tokenization_thermal import LanguageBindThermalTokenizer
from .thermal.processing_thermal import LanguageBindThermalProcessor



config_dict = {
    'thermal': LanguageBindThermalConfig,
    'image': LanguageBindImageConfig,
    'video': LanguageBindVideoConfig,
    'depth': LanguageBindDepthConfig,
    'audio': LanguageBindAudioConfig
}
model_dict = {
    'thermal': LanguageBindThermal,
    'image': LanguageBindImage,
    'video': LanguageBindVideo,
    'depth': LanguageBindDepth,
    'audio': LanguageBindAudio
}
transform_dict = {
    'video': LanguageBindVideoProcessor,
    'audio': LanguageBindAudioProcessor,
    'depth': LanguageBindDepthProcessor,
    'thermal': LanguageBindThermalProcessor,
    'image': LanguageBindImageProcessor,
}

class LanguageBind(nn.Module):
    def __init__(self, clip_type=('thermal', 'image', 'video', 'depth', 'audio'), use_temp=True, cache_dir='./cache_dir'):
        super(LanguageBind, self).__init__()
        self.use_temp = use_temp
        self.modality_encoder = {}
        self.modality_proj = {}
        self.modality_scale = {}
        self.modality_config = {}
        for c in clip_type:
            pretrained_ckpt = f'LanguageBind/LanguageBind_{c.capitalize()}'
            model = model_dict[c].from_pretrained(pretrained_ckpt, cache_dir=cache_dir)
            self.modality_encoder[c] = model.vision_model
            self.modality_proj[c] = model.visual_projection
            self.modality_scale[c] = model.logit_scale
            self.modality_config[c] = model.config
        self.modality_encoder['language'] = model.text_model
        self.modality_proj['language'] = model.text_projection

        self.modality_encoder = nn.ModuleDict(self.modality_encoder)
        self.modality_proj = nn.ModuleDict(self.modality_proj)

    def forward(self, inputs):
        outputs = {}
        for key, value in inputs.items():
            value = self.modality_encoder[key](**value)[1]
            value = self.modality_proj[key](value)
            value = value / value.norm(p=2, dim=-1, keepdim=True)
            if self.use_temp:
                if key != 'language':
                    value = value * self.modality_scale[key].exp()
            outputs[key] = value
        return outputs

def to_device(x, device):
    out_dict = {k: v.to(device) for k, v in x.items()}
    return out_dict




class LanguageBindImageTower(nn.Module):
    def __init__(self, image_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.image_tower_name = image_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = LanguageBindImageConfig.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)

    ############################################################
    def load_model(self):
        model = LanguageBindImage.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)
        self.image_tower = model.vision_model
        self.image_tower.requires_grad_(False)

        self.image_processor = LanguageBindImageProcessor(model.config)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.image_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            # print('images', images.shape)
            image_forward_outs = self.image_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            # print('image_forward_outs', len(image_forward_outs), image_forward_outs[0].shape)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            # print('image_features', image_features.shape)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.image_tower.embeddings.class_embedding.dtype  #############

    @property
    def device(self):
        return self.image_tower.embeddings.class_embedding.device  ##############

    @property
    def config(self):
        if self.is_loaded:
            return self.image_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class LanguageBindVideoTower(nn.Module):
    def __init__(self, video_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.video_tower_name = video_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = LanguageBindVideoConfig.from_pretrained(self.video_tower_name, cache_dir=self.cache_dir)

    ############################################################
    def load_model(self):
        model = LanguageBindVideo.from_pretrained(self.video_tower_name, cache_dir=self.cache_dir)
        self.video_processor = LanguageBindVideoProcessor(model.config)


        # model = LanguageBindImage.from_pretrained('LanguageBind/LanguageBind_Image', cache_dir=self.cache_dir)
        self.video_tower = model.vision_model
        self.video_tower.requires_grad_(False)


        self.is_loaded = True


    def feature_select(self, video_forward_outs):
        video_features = video_forward_outs.hidden_states[self.select_layer]  # b t n c
        return video_features  # return all
        # b, t, n, c = video_features.shape
        # if self.select_feature == 'patch':
        #     video_features = video_features[:, :, 1:]
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        # return video_features

    @torch.no_grad()
    def forward(self, videos):
        if type(videos) is list:
            video_features = []
            for video in videos:
                video_forward_out = self.video_tower(video.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                video_feature = self.feature_select(video_forward_out).to(video.dtype)
                video_features.append(video_feature)
        else:
            video_forward_outs = self.video_tower(videos.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            video_features = self.feature_select(video_forward_outs).to(videos.dtype)

        return video_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.video_tower.embeddings.class_embedding.dtype  #############
        # return torch.randn(1).cuda().dtype

    @property
    def device(self):
        return self.video_tower.embeddings.class_embedding.device  ##############
        # return torch.randn(1).cuda().device

    @property
    def config(self):
        if self.is_loaded:
            return self.video_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class LanguageBindAudioTower(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.audio_tower_name = audio_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = LanguageBindAudioConfig.from_pretrained(self.audio_tower_name, cache_dir=self.cache_dir)

    ############################################################
    def load_model(self):
        model = LanguageBindAudio.from_pretrained(self.audio_tower_name, cache_dir=self.cache_dir)
        self.audio_processor = LanguageBindAudioProcessor(model.config)


        # model = LanguageBindImage.from_pretrained('LanguageBind/LanguageBind_Image', cache_dir=self.cache_dir)
        self.audio_tower = model.vision_model
        self.audio_tower.requires_grad_(False)

        self.is_loaded = True


    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs.hidden_states[self.select_layer]  # b t n c
        return audio_features  # return all

    @torch.no_grad()
    def forward(self, audios):
        if type(audios) is list:
            audio_features = []
            for audio in audios:
                audio_forward_out = self.audio_tower(audio.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                audio_feature = self.feature_select(audio_forward_out).to(audio.dtype)
                audio_features.append(audio_feature)
        else:
            audio_forward_outs = self.audio_tower(audios.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            audio_features = self.feature_select(audio_forward_outs).to(audios.dtype)

        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.embeddings.class_embedding.dtype  #############
        # return torch.randn(1).cuda().dtype

    @property
    def device(self):
        return self.audio_tower.embeddings.class_embedding.device  ##############
        # return torch.randn(1).cuda().device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class LanguageBindDepthTower(nn.Module):
    def __init__(self, depth_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.depth_tower_name = depth_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = LanguageBindDepthConfig.from_pretrained(self.depth_tower_name, cache_dir=self.cache_dir)

    ############################################################
    def load_model(self):
        model = LanguageBindDepth.from_pretrained(self.depth_tower_name, cache_dir=self.cache_dir)
        self.depth_processor = LanguageBindDepthProcessor(model.config)


        # model = LanguageBindImage.from_pretrained('LanguageBind/LanguageBind_Image', cache_dir=self.cache_dir)
        self.depth_tower = model.vision_model
        self.depth_tower.requires_grad_(False)


        self.is_loaded = True


    def feature_select(self, depth_forward_outs):
        depth_features = depth_forward_outs.hidden_states[self.select_layer]  # b t n c
        return depth_features  # return all
        # b, t, n, c = depth_features.shape
        # if self.select_feature == 'patch':
        #     depth_features = depth_features[:, :, 1:]
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        # return depth_features

    @torch.no_grad()
    def forward(self, depths):
        if type(depths) is list:
            depth_features = []
            for depth in depths:
                depth_forward_out = self.depth_tower(depth.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                depth_feature = self.feature_select(depth_forward_out).to(depth.dtype)
                depth_features.append(depth_feature)
        else:
            depth_forward_outs = self.depth_tower(depths.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            depth_features = self.feature_select(depth_forward_outs).to(depths.dtype)

        return depth_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.depth_tower.embeddings.class_embedding.dtype  #############
        # return torch.randn(1).cuda().dtype

    @property
    def device(self):
        return self.depth_tower.embeddings.class_embedding.device  ##############
        # return torch.randn(1).cuda().device

    @property
    def config(self):
        if self.is_loaded:
            return self.depth_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


class LanguageBindThermalTower(nn.Module):
    def __init__(self, thermal_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.thermal_tower_name = thermal_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = LanguageBindThermalConfig.from_pretrained(self.thermal_tower_name, cache_dir=self.cache_dir)

    ############################################################
    def load_model(self):
        model = LanguageBindThermal.from_pretrained(self.thermal_tower_name, cache_dir=self.cache_dir)
        self.thermal_processor = LanguageBindThermalProcessor(model.config)


        # model = LanguageBindImage.from_pretrained('LanguageBind/LanguageBind_Image', cache_dir=self.cache_dir)
        self.thermal_tower = model.vision_model
        self.thermal_tower.requires_grad_(False)


        self.is_loaded = True


    def feature_select(self, thermal_forward_outs):
        thermal_features = thermal_forward_outs.hidden_states[self.select_layer]  # b t n c
        return thermal_features  # return all
        # b, t, n, c = thermal_features.shape
        # if self.select_feature == 'patch':
        #     thermal_features = thermal_features[:, :, 1:]
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        # return thermal_features

    @torch.no_grad()
    def forward(self, thermals):
        if type(thermals) is list:
            thermal_features = []
            for thermal in thermals:
                thermal_forward_out = self.thermal_tower(thermal.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                thermal_feature = self.feature_select(thermal_forward_out).to(thermal.dtype)
                thermal_features.append(thermal_feature)
        else:
            thermal_forward_outs = self.thermal_tower(thermals.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            thermal_features = self.feature_select(thermal_forward_outs).to(thermals.dtype)

        return thermal_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.thermal_tower.embeddings.class_embedding.dtype  #############
        # return torch.randn(1).cuda().dtype

    @property
    def device(self):
        return self.thermal_tower.embeddings.class_embedding.device  ##############
        # return torch.randn(1).cuda().device

    @property
    def config(self):
        if self.is_loaded:
            return self.thermal_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


