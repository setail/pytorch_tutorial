import torch
from torchvision import transforms


class ScaleTransform():

    def __init__(self, scale_factor):
        self._scale_factor = scale_factor

    def __call__(self, sample):
        sample = sample * self._scale_factor
        return sample


class AddTransform():

    def __init__(self, add_value):
        self._add_value = add_value

    def __call__(self, sample):
        sample = sample + self._add_value
        return sample


data = torch.rand(3, 5)
scale_transform = ScaleTransform(10)
data_after_scale = scale_transform(data)
add_transform = AddTransform(50)
data_after_add = add_transform(data_after_scale)
print(f'origin data:\n{data}')
print(f'data after scale transform:\n{data_after_scale}')
print(f'data after add transform:\n{data_after_add}')

composed_transform = transforms.Compose([ScaleTransform(10), AddTransform(50)])
data_after_composed_transform = composed_transform(data)
print(f'data after add transform:\n{data_after_add}')
print(f'data after composed transform:\n{data_after_composed_transform}')
