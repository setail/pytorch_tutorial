import torchvision
from torchvision import transforms
from PIL import Image

# load image
path = f'./sample.png'
with open(path, 'rb') as f:
    img = Image.open(f)
    img_parse = img.convert('RGB')

# transform on PIL Image
# Resize
data_transform = transforms.Resize((150, 150))

# Crop
# data_transform = transforms.RandomCrop(100)
# data_transform = transforms.CenterCrop(100)
# data_transform = transforms.RandomResizedCrop(100)

# Flip
# data_transform = transforms.RandomHorizontalFlip(0.5)
# data_transform = transforms.RandomVerticalFlip(0.5)

img_transform = data_transform(img_parse)
print(f'type before transform: {type(img_parse)}')
print(f'type after transform: {type(img_transform)}')

# img.show()
# img_transform.show()

# Conversion between PIL Image and Tensor(C, H, W)
# Conversion to Tensor
data_transform = transforms.ToTensor()
img_tensor = data_transform(img_parse)
print(f'type before transform: {type(img_parse)}')
print(f'type after transform: {type(img_tensor)}')
print(f'tensor after transform: {img_tensor.shape}')

# Conversion to PIL
data_transform = transforms.ToPILImage()
img_pil = data_transform(img_tensor)
# img_pil.show()

# transform on Tensor
img_tensor_crop = img_tensor[:, :3, :3]
print('before normalization:\n', img_tensor_crop)
means = img_tensor_crop.mean([1, 2])
std = img_tensor_crop.std([1, 2])
data_transform = transforms.Normalize([0.1708, 0.1882, 0.1834], [0.0048, 0.0068, 0.0061])
img_norm = data_transform(img_tensor_crop)
print('after normalization:\n', img_norm)
