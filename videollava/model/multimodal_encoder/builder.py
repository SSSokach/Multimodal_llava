import os
from .clip_encoder import CLIPVisionTower
from .languagebind import LanguageBindImageTower, LanguageBindVideoTower, LanguageBindAudioTower, LanguageBindDepthTower, LanguageBindThermalTower

# ============================================================================================================

def build_image_tower(image_tower_cfg, **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    is_absolute_path_exists = os.path.exists(image_tower)
    if image_tower.endswith('LanguageBind_Image'):
        return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
    if is_absolute_path_exists or image_tower.startswith("openai") or image_tower.startswith("laion"):
        return CLIPVisionTower(image_tower, args=image_tower_cfg, **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    if video_tower.endswith('LanguageBind_Video_merge'):
        return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown video tower: {video_tower}')

def build_audio_tower(audio_tower_cfg, **kwargs):
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))
    if audio_tower.endswith('LanguageBind_Audio'):
        return LanguageBindAudioTower(audio_tower, args=audio_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown audio tower: {audio_tower}')

def build_depth_tower(depth_tower_cfg, **kwargs):
    depth_tower = getattr(depth_tower_cfg, 'mm_depth_tower', getattr(depth_tower_cfg, 'depth_tower', None))
    if depth_tower.endswith('LanguageBind_Depth'):
        return LanguageBindDepthTower(depth_tower, args=depth_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown depth tower: {depth_tower}')

def build_thermal_tower(thermal_tower_cfg, **kwargs):
    thermal_tower = getattr(thermal_tower_cfg, 'mm_thermal_tower', getattr(thermal_tower_cfg, 'thermal_tower', None))
    if thermal_tower.endswith('LanguageBind_Thermal'):
        return LanguageBindThermalTower(thermal_tower, args=thermal_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown thermal tower: {thermal_tower}')
# ============================================================================================================
