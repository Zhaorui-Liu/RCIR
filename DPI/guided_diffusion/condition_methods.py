from abc import ABC, abstractmethod
import torch
import time

__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser
    
    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]

        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm
        
@register_conditioning_method(name='ps')
class PosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        return x_t, norm
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm

    
@register_conditioning_method(name='rt-ps+')
class RayTracingPosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale_method = kwargs.get('scale_method', 'paper')
        self.use_log = kwargs.get('use_log', False)
        self.scales = kwargs.get('scales', [0.1,1.0])
        self.rho_min = kwargs.get('rho_min', 0.1)
        self.rho_max = kwargs.get('rho_max', 1.0)
        self.rho_transition = kwargs.get('rho_transition', 0.6)
        self.joint_optimization_start = kwargs.get('joint_optimization_start', 0.8)
        self.grad_clip = kwargs.get('grad_clip', 0.1)
        self.spp=kwargs.get('spp', 16)

    def _posterior_scale(self, timestep):
        if self.scale_method == 'paper':
            if timestep > self.rho_transition:
                return self.rho_min
            progress = (self.rho_transition - timestep) / max(self.rho_transition, 1e-8)
            return self.rho_min + progress * (self.rho_max - self.rho_min)

        if self.scale_method=='step':
            if timestep>=0.9:
                return 0.01
            elif timestep<0.9 and timestep>=0.5:
                return self.scales[0]
            elif timestep<0.5 and timestep>=0.:
                return self.scales[1]

        if self.scale_method=='linear':
            if timestep>=0.9:
                return 0.01
            elif timestep<0.9 and timestep>=0.5:
                return 0.1-(timestep-0.9)*(self.scales[0]-0.1)/(0.9-0.5)
            elif timestep<0.5:
                return self.scales[0]+(timestep-0.5)*(self.scales[0]-self.scales[1])/0.5

        raise NotImplementedError

    def conditioning(self, x_prev, x_t, x_0_hat, measurement,timestep, **kwargs):
        x_0_hat_noise = x_0_hat+ 0.05 * torch.rand_like(x_0_hat) 
        x_0_hat_noise=x_0_hat_noise.clamp(-1.,1.)
        
        measurement,estimation= self.operator.forward_multiview(x_0_hat_noise)
        
        if self.use_log:            
            measurement,estimation=torch.log(measurement+1),torch.log(estimation+1)
        
        
        difference = measurement - estimation
        
        norm= torch.linalg.norm(difference) 
   
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev,retain_graph =False)[0]
        norm_grad=torch.nan_to_num(norm_grad)
        grad_max = torch.abs(norm_grad).max().item()
        if grad_max > self.grad_clip:
            norm_grad = norm_grad * (self.grad_clip / grad_max)

        scale = self._posterior_scale(timestep)
            
        x_t -= norm_grad * scale
     
        if timestep < self.joint_optimization_start:
            self.operator.update_material(x_0_hat,self.spp,t=timestep)
        
        norm_d=norm.detach()
        
        del norm_grad,estimation,norm,difference

        return x_t, norm_d
