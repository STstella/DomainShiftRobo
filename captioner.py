
from abc import ABC, abstractmethod

# Abstract class for captioning


class Captioner(ABC):
    def __init__(self, device='cuda:0'):
        self.device = device

    @abstractmethod
    def _init_models(self):
        pass
    
    # [Modified] The interface parameter list has been updated to include return_sts=False
    @abstractmethod
    def caption(self, imgs, user_prompt=None, return_sts=False):
        pass

    @abstractmethod
    def stop(self):
        pass