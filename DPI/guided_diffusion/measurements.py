'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''
import drjit as dr
import mitsuba as mi
mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')
from mitsuba.scalar_rgb import Transform4f as T

from abc import ABC, abstractmethod
from functools import partial
import yaml
from torch.nn import functional as F
import torch


from util.resizer import Resizer
from util.img_utils import fft2_m,imread

import numpy as np
import random


import math

from  torch.cuda.amp import autocast

import os
import glob
import warnings

# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if 'raytracing' in name:
        name='raytracing' 
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


class LinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        # calculate A * X
        pass

    @abstractmethod
    def transpose(self, data, **kwargs):
        # calculate A^T * X
        pass
    
    def ortho_project(self, data, **kwargs):
        # calculate (I - A^T * A)X
        return data - self.transpose(self.forward(data, **kwargs), **kwargs)

    def project(self, data, measurement, **kwargs):
        # calculate (I - A^T * A)Y - AX
        return self.ortho_project(measurement, **kwargs) - self.forward(data, **kwargs)


@register_operator(name='noise')
class DenoiseOperator(LinearOperator):
    def __init__(self, device):
        self.device = device
    
    def forward(self, data):
        return data

    def transpose(self, data):
        return data
    
    def ortho_project(self, data):
        return data

    def project(self, data):
        return data



class NonLinearOperator(ABC):
    @abstractmethod
    def forward(self, data, **kwargs):
        pass

    def project(self, data, measurement, **kwargs):
        return data + measurement - self.forward(data) 


@register_operator(name='raytracing')
class RaytracingOperator(NonLinearOperator):
    def __init__(self,scene_name,image_path,n_images,ldr,scene_path,camera_path,illumi_gamma,illumi_scale,illumi_normalize,texture_res,device,
                 prior=None,param_keys=None,optimizer=None,relighting=None,init_values=None):
        self.device = torch.device(device)
        self.ldr=ldr
            
        
        
        self.scene_name=scene_name
        self.image_path=image_path
        self.scene=mi.load_file(scene_path)
        self.cam_scene=mi.load_file(camera_path)
        self.params = mi.traverse(self.scene)
        self.cam_params = mi.traverse(self.cam_scene)
        
        self.texture_res=texture_res
        self.n_images=n_images
        self.rgb_images = []                                
        for i in range(self.n_images):
            if self.ldr:  
                rgb_img = imread('{}/{}.png'.format(self.image_path,i),gamma=1)[:,:,:3]
            else:
                rgb_img = imread('{}/{}.exr'.format(self.image_path,i))[:,:,:3]
        
            
            self.rgb_images.append(rgb_img.to(self.device))

        self.param_keys = {
            'basecolor': 'OBJMesh.bsdf.base_color.data',
            'roughness': 'OBJMesh.bsdf.roughness.data',
            'metallic': 'OBJMesh.bsdf.metallic.data',
            'normal': None,
        }
        if param_keys:
            self.param_keys.update(param_keys)
        if self.param_keys['normal'] is None:
            self.param_keys['normal'] = self._find_first_param([
                'OBJMesh.bsdf.normalmap.data',
                'OBJMesh.bsdf.normal.data',
                'OBJMesh.normalmap.data',
            ])
        self.has_normal_param = self.param_keys['normal'] in self.params
        for required in ('basecolor', 'roughness', 'metallic'):
            if self.param_keys[required] not in self.params:
                raise KeyError(
                    "Mitsuba scene parameter '{}' for {} was not found. "
                    "Set measurement.operator.param_keys.{} in the task config.".format(
                        self.param_keys[required], required, required
                    )
                )

        self.prior_config = prior or {}
        self.init_values = {
            'basecolor': 0.01,
            'roughness': 0.1,
            'metallic': 0.1,
            'normal': [0.5, 0.5, 1.0],
        }
        if init_values:
            self.init_values.update(init_values)

        self.optimizer_config = {
            'lr': 1e-2,
            'basecolor_lr': 1e-2,
            'roughness_lr': 1e-2,
            'metallic_lr': 1e-2,
            'normal_lr': 1e-2,
        }
        if optimizer:
            self.optimizer_config.update(optimizer)

        self.loss_fn = torch.nn.MSELoss()
        
        self.gamma=illumi_gamma
        self.scale=illumi_scale
        self.normal=illumi_normalize

        self.relighting_config = {
            'enabled': False,
            'lambda_initial': 0.2,
            'lambda_final': 0.5,
            'rank_weight': 1.0,
            'image_weight': 1.0,
            'env_shift': 64,
            'use_normal_consistency': True,
            'spp': 8,
        }
        if relighting:
            self.relighting_config.update(relighting)
        if self.relighting_config['enabled'] and not self.has_normal_param:
            warnings.warn('Relighting normal consistency is disabled because no normal texture parameter was found.')
            self.relighting_config['use_normal_consistency'] = False

        self._initial_basecolor = self._initial_texture(
            self.prior_config.get('albedo_path') or self.prior_config.get('basecolor_path'),
            'basecolor',
            channels=3,
            default_value=self.init_values['basecolor'],
            gamma=self.prior_config.get('albedo_gamma', 1.0),
        )
        self._initial_roughness = self._initial_texture(
            self.prior_config.get('roughness_path'),
            'roughness',
            channels=1,
            default_value=self.init_values['roughness'],
            gamma=self.prior_config.get('roughness_gamma', 1.0),
        )
        self._initial_metallic = self._initial_texture(
            self.prior_config.get('metallic_path'),
            'metallic',
            channels=1,
            default_value=self.init_values['metallic'],
            gamma=self.prior_config.get('metallic_gamma', 1.0),
        )
        self._initial_normal = self._initial_texture(
            self.prior_config.get('normal_path'),
            'normal',
            channels=3,
            default_value=self.init_values['normal'],
            gamma=self.prior_config.get('normal_gamma', 1.0),
            signed=self.prior_config.get('normal_space', 'rgb01') == 'signed',
        )
        self.reset_optimizer()

    def _find_first_param(self, candidates):
        for key in candidates:
            if key in self.params:
                return key
        return None

    def _scene_texture(self, name, channels):
        key = self.param_keys.get(name)
        if key in self.params:
            tex = self.params[key].torch().clone().to(self.device).float()
            if tex.ndim == 2:
                tex = tex.unsqueeze(-1)
            if channels == 1:
                tex = tex[:, :, :1]
            elif tex.shape[-1] < channels:
                tex = tex.repeat(1, 1, channels)
            else:
                tex = tex[:, :, :channels]
            return self._resize_texture(tex, channels)
        return None

    def _constant_texture(self, value, channels):
        if isinstance(value, (list, tuple)):
            tex = torch.tensor(value, device=self.device, dtype=torch.float32).view(1, 1, -1)
            if tex.shape[-1] < channels:
                tex = tex.repeat(1, 1, channels)
            tex = tex[:, :, :channels]
            return tex.expand(self.texture_res, self.texture_res, channels).clone()
        return torch.full([self.texture_res, self.texture_res, channels], float(value), device=self.device)

    def _resize_texture(self, tex, channels):
        if tex.shape[:2] == (self.texture_res, self.texture_res):
            return tex
        tex = tex.permute(2, 0, 1).unsqueeze(0)
        tex = F.interpolate(tex, size=(self.texture_res, self.texture_res), mode='bilinear', align_corners=False)
        tex = tex.squeeze(0).permute(1, 2, 0)
        if channels == 1 and tex.shape[-1] != 1:
            tex = tex.mean(dim=-1, keepdim=True)
        return tex

    def _load_texture(self, path, channels, gamma, signed=False):
        tex = imread(path, gamma=gamma).to(self.device).float()
        if tex.ndim == 2:
            tex = tex.unsqueeze(-1)
        if tex.shape[-1] == 4:
            tex = tex[:, :, :3]
        if signed:
            tex = (tex + 1.0) * 0.5
        if channels == 1:
            tex = tex[:, :, :1] if tex.shape[-1] == 1 else tex[:, :, :3].mean(dim=-1, keepdim=True)
        elif tex.shape[-1] < channels:
            tex = tex.repeat(1, 1, channels)
        else:
            tex = tex[:, :, :channels]
        tex = self._resize_texture(tex, channels)
        return torch.nan_to_num(tex, nan=0.0).clamp(0.0, 1.0)

    def _initial_texture(self, path, name, channels, default_value, gamma=1.0, signed=False):
        if path:
            return self._load_texture(path, channels, gamma=gamma, signed=signed)
        scene_tex = self._scene_texture(name, channels)
        if scene_tex is not None and self.prior_config.get('use_scene_{}_init'.format(name), False):
            return torch.nan_to_num(scene_tex, nan=0.0).clamp(0.0, 1.0)
        return self._constant_texture(default_value, channels)

    def _make_optimizable(self, tex):
        tex = tex.detach().clone().to(self.device).float()
        tex.requires_grad = True
        return tex

    def _build_optimizer(self):
        groups = [
            {'params': [self.basecolor], 'lr': self.optimizer_config['basecolor_lr']},
            {'params': [self.roughness], 'lr': self.optimizer_config['roughness_lr']},
            {'params': [self.metallic], 'lr': self.optimizer_config['metallic_lr']},
        ]
        if self.has_normal_param:
            groups.append({'params': [self.normal_map], 'lr': self.optimizer_config['normal_lr']})
        self.optimizer = torch.optim.Adam(groups, lr=self.optimizer_config['lr'])

    def _build_relighting_optimizer(self):
        groups = [
            {'params': [self.relit_basecolor], 'lr': self.optimizer_config['basecolor_lr']},
            {'params': [self.relit_roughness], 'lr': self.optimizer_config['roughness_lr']},
            {'params': [self.relit_metallic], 'lr': self.optimizer_config['metallic_lr']},
        ]
        if self.has_normal_param:
            groups.append({'params': [self.relit_normal_map], 'lr': self.optimizer_config['normal_lr']})
        self.relighting_optimizer = torch.optim.Adam(groups, lr=self.optimizer_config['lr'])

    def reset_optimizer(self):
        self.basecolor = self._make_optimizable(self._initial_basecolor.clamp(1e-8, 1.0 - 1e-8))
        self.metallic = self._make_optimizable(self._initial_metallic.clamp(1e-8, 1.0 - 1e-8))
        self.roughness = self._make_optimizable(self._initial_roughness.clamp(1e-8, 1.0 - 1e-8))
        self.normal_map = self._make_optimizable(self._initial_normal.clamp(1e-8, 1.0 - 1e-8))
        self._build_optimizer()
        if self.relighting_config['enabled']:
            self.relit_basecolor = self._make_optimizable(self.basecolor.detach())
            self.relit_metallic = self._make_optimizable(self.metallic.detach())
            self.relit_roughness = self._make_optimizable(self.roughness.detach())
            self.relit_normal_map = self._make_optimizable(self.normal_map.detach())
            self._build_relighting_optimizer()

    def _set_camera(self, cam):
        if cam == 0:
            self.params['PerspectiveCamera.to_world'] = self.cam_params['PerspectiveCamera.to_world']
        else:
            self.params['PerspectiveCamera.to_world'] = self.cam_params['PerspectiveCamera_{}.to_world'.format(cam)]

    def _envmap_from_sample(self, data, detach=False):
        data_scale = (data + 1.) / 2
        data_scale = data_scale.clamp(0, 1)
        data_scale = self.scale * torch.pow(data_scale / self.normal, self.gamma)
        if detach:
            data_scale = data_scale.detach()

        envmap = torch.ones([self.texture_res, self.texture_res + 1, 3], device=self.device)
        envmap[:, :, :3] *= 1e-8
        envmap[:, :self.texture_res, :3] = data_scale.squeeze().permute(1, 2, 0)
        envmap[:, self.texture_res, :3] = envmap[:, self.texture_res - 1, :3]
        return envmap

    def _novel_envmap(self, envmap):
        shift = int(self.relighting_config.get('env_shift', 64))
        if shift <= 0:
            shift = random.randint(1, max(1, self.texture_res - 1))
        novel = torch.roll(envmap, shifts=shift, dims=1)
        novel[:, self.texture_res, :3] = novel[:, self.texture_res - 1, :3]
        return novel

    def _rank_one_residual(self, first, second):
        matrix = torch.stack([first.reshape(-1), second.reshape(-1)], dim=0).float()
        singular_values = torch.linalg.svdvals(matrix)
        if singular_values.numel() <= 1:
            return singular_values.sum() * 0.0
        return torch.sum(singular_values[1:] ** 2) / matrix.shape[1]

    def _consistency_loss(self, basecolor, normal_map, ref_basecolor, ref_normal_map):
        loss = self._rank_one_residual(basecolor, ref_basecolor)
        if self.relighting_config['use_normal_consistency'] and self.has_normal_param:
            loss = loss + self._rank_one_residual(normal_map, ref_normal_map)
        return loss

    def _relighting_lambda(self, t):
        start = float(self.relighting_config['lambda_initial'])
        end = float(self.relighting_config['lambda_final'])
        progress = 1.0 - max(0.0, min(1.0, float(t) / 0.8))
        return start + (end - start) * progress

    def _clamp_materials(self, basecolor, roughness, metallic, normal_map):
        basecolor.data = torch.nan_to_num(basecolor.data.clamp(1e-8, 1.0 - 1e-8))
        metallic.data = torch.nan_to_num(metallic.data.clamp(1e-8, 1.0 - 1e-8), nan=0.0)
        roughness.data = torch.nan_to_num(roughness.data.clamp(1e-8, 1.0 - 1e-8), nan=1.0)
        normal_map.data = torch.nan_to_num(normal_map.data.clamp(1e-8, 1.0 - 1e-8), nan=0.5)
        
############# rendering method ###################        
    @dr.wrap_ad(source='torch', target='drjit')
    def render_envmap(self,envmap,spp=256):
        
        self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera.to_world']
        
        self.params['EnvironmentMapEmitter.data']=envmap  
        self.params.update()
        rendered_img=mi.render(self.scene, self.params, spp=spp)

        return rendered_img
       
        
    @dr.wrap_ad(source='torch', target='drjit')
    def render_camera_envmap(self,envmap,spp=256,cam=0):
        if cam==0:
            self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera.to_world']
        else:
            self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera_{}.to_world'.format(cam)]

        self.params['EnvironmentMapEmitter.data']=envmap
        self.params.update()
        rendered_img=mi.render(self.scene, self.params, spp=spp)
        return rendered_img

    def render_multiview(self,envmap,spp=256,cam=None):
        cam = random.randint(0,self.n_images-1) if cam is None else cam
        rendered_gt=self.rgb_images[cam].to(self.device)
        rendered_img=self.render_camera_envmap(envmap,spp=spp,cam=cam)

        return rendered_gt,rendered_img
    
    @dr.wrap_ad(source='torch', target='drjit')
    def render_camera_with_material(self,envmap,basecolor,roughness,metallic,normal_map,spp=256,cam=0):
        if cam==0:
            self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera.to_world']
        else:
            self.params['PerspectiveCamera.to_world']=self.cam_params['PerspectiveCamera_{}.to_world'.format(cam)]

        self.params['EnvironmentMapEmitter.data']=envmap
        self.params[self.param_keys['basecolor']]=basecolor
        self.params[self.param_keys['roughness']]=roughness
        self.params[self.param_keys['metallic']]=metallic
        if self.has_normal_param:
            self.params[self.param_keys['normal']]=normal_map

        self.params.update()
        rendered_img=mi.render(self.scene, self.params, spp=spp)
        return rendered_img

    def render_multiview_with_material(self,envmap,basecolor,roughness,metallic,normal_map=None,spp=256,cam=None):
        cam = random.randint(0,self.n_images-1) if cam is None else cam
        rendered_gt=self.rgb_images[cam].to(self.device)
        if normal_map is None:
            normal_map = self.normal_map
        rendered_img=self.render_camera_with_material(envmap,basecolor,roughness,metallic,normal_map,spp=spp,cam=cam)
        return rendered_gt,rendered_img
    

############# forward method ###################       
    def forward(self, data,spp=16, **kwargs):
        envmap = self._envmap_from_sample(data)
        rendered_img=self.render_envmap(envmap,spp=spp) 
        rendered_img=rendered_img.permute(2,0,1).unsqueeze(0)

        return rendered_img
    
    def forward_gt(self,spp=16, **kwargs):

        cam=random.randint(0,self.n_images-1)
        rendered_img=self.rgb_images[cam].to(self.device)
            
        return rendered_img.permute(2,0,1).unsqueeze(0)
    
    def forward_multiview(self, data,spp=16, **kwargs):
        envmap = self._envmap_from_sample(data)
        rendered_gt,rendered_img=self.render_multiview(envmap,spp=spp)

        rendered_gt=rendered_gt.permute(2,0,1).unsqueeze(0)
        rendered_img=rendered_img.permute(2,0,1).unsqueeze(0)

        return rendered_gt,rendered_img    
   
    def _update_relighting_branch(self, envmap, t):
        if not self.relighting_config['enabled']:
            return None

        novel_envmap = self._novel_envmap(envmap.detach())
        cam = random.randint(0, self.n_images - 1)
        spp = int(self.relighting_config.get('spp', 8))
        weight = self._relighting_lambda(t) * float(self.relighting_config['rank_weight'])
        image_weight = float(self.relighting_config['image_weight'])

        self.relighting_optimizer.zero_grad()
        relit_target = self.render_camera_with_material(
            novel_envmap,
            self.basecolor.detach(),
            self.roughness.detach(),
            self.metallic.detach(),
            self.normal_map.detach(),
            spp=spp,
            cam=cam,
        ).detach()
        relit_render = self.render_camera_with_material(
            novel_envmap,
            self.relit_basecolor,
            self.relit_roughness,
            self.relit_metallic,
            self.relit_normal_map,
            spp=spp,
            cam=cam,
        )
        relit_loss = image_weight * self.loss_fn(relit_render, relit_target)
        relit_loss = relit_loss + weight * self._consistency_loss(
            self.basecolor.detach(),
            self.normal_map.detach(),
            self.relit_basecolor,
            self.relit_normal_map,
        )
        if ~(torch.isnan(relit_loss) | torch.isinf(relit_loss)):
            relit_loss.backward()
            self.relighting_optimizer.step()
            self._clamp_materials(self.relit_basecolor, self.relit_roughness, self.relit_metallic, self.relit_normal_map)

        return weight * self._consistency_loss(
            self.basecolor,
            self.normal_map,
            self.relit_basecolor.detach(),
            self.relit_normal_map.detach(),
        )

    def update_material(self, data,spp=32,t=0., **kwargs):
        envmap = self._envmap_from_sample(data, detach=True)
        relighting_loss = self._update_relighting_branch(envmap, t)

        self.optimizer.zero_grad()
        with autocast():
            rendered_gt,rendered_img=self.render_multiview_with_material(
                envmap,
                self.basecolor,
                self.roughness,
                self.metallic,
                self.normal_map,
                spp=spp,
            )
            loss = self.loss_fn(rendered_img, rendered_gt)
            if relighting_loss is not None:
                loss = loss + relighting_loss

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            self.optimizer.step()
            self._clamp_materials(self.basecolor, self.roughness, self.metallic, self.normal_map)
           
        del(loss,rendered_img,envmap)
    

        
# =============
# Noise classes
# =============


__NOISE__ = {}

def register_noise(name: str):
    def wrapper(cls):
        if __NOISE__.get(name, None):
            raise NameError(f"Name {name} is already defined!")
        __NOISE__[name] = cls
        return cls
    return wrapper

def get_noise(name: str, **kwargs):
    if __NOISE__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    noiser = __NOISE__[name](**kwargs)
    noiser.__name__ = name
    return noiser

class Noise(ABC):
    def __call__(self, data):
        return self.forward(data)
    
    @abstractmethod
    def forward(self, data):
        pass

@register_noise(name='clean')
class Clean(Noise):
    def forward(self, data):
        return data

@register_noise(name='gaussian')
class GaussianNoise(Noise):
    def __init__(self, sigma):
        self.sigma = sigma
    
    def forward(self, data):
        return data + torch.randn_like(data, device=data.device) * self.sigma


@register_noise(name='poisson')
class PoissonNoise(Noise):
    def __init__(self, rate):
        self.rate = rate

    def forward(self, data):
        '''
        Follow skimage.util.random_noise.
        '''

        # TODO: set one version of poisson
       
        # version 3 (stack-overflow)
        import numpy as np
        data = (data + 1.0) / 2.0
        data = data.clamp(0, 1)
        device = data.device
        data = data.detach().cpu()
        data = torch.from_numpy(np.random.poisson(data * 255.0 * self.rate) / 255.0 / self.rate)
        data = data * 2.0 - 1.0
        data = data.clamp(-1, 1)
        return data.to(device)

        # version 2 (skimage)
        # if data.min() < 0:
        #     low_clip = -1
        # else:
        #     low_clip = 0

    
        # # Determine unique values in iamge & calculate the next power of two
        # vals = torch.Tensor([len(torch.unique(data))])
        # vals = 2 ** torch.ceil(torch.log2(vals))
        # vals = vals.to(data.device)

        # if low_clip == -1:
        #     old_max = data.max()
        #     data = (data + 1.0) / (old_max + 1.0)

        # data = torch.poisson(data * vals) / float(vals)

        # if low_clip == -1:
        #     data = data * (old_max + 1.0) - 1.0
       
        # return data.clamp(low_clip, 1.0)
