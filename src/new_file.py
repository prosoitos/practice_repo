import torch
from datasets import load_dataset

set5 = load_dataset('eugenesiow/Set5', 'bicubic_x4', split='validation')

print(set5)
len(set5)
set5[0]
set5.shape
set5.column_names
set5.features
set5.format

set5.set_format('torch', columns=['hr', 'lr'])
set5.format

#################################################################################################################

import kornia
from PIL import Image
import torch
from torchvision import transforms

berlin1_lr = Image.open("/home/marie/parvus/pwg/wtm/slides/static/img/upscaling/lr/berlin_1945_1.jpg")
berlin1_hr = Image.open("/home/marie/parvus/pwg/wtm/slides/static/img/upscaling/hr/berlin_1945_1.png")

berlin1_lr.show()

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
        ])

berlin1_lr_t = preprocess(berlin1_lr)
berlin1_hr_t = preprocess(berlin1_hr)

berlin1_lr_t.shape
berlin1_hr_t.shape

batch_berlin1_lr_t = torch.unsqueeze(berlin1_lr_t, 0)
batch_berlin1_hr_t = torch.unsqueeze(berlin1_hr_t, 0)

batch_berlin1_lr_t.shape
batch_berlin1_hr_t.shape

psnr_value = kornia.metrics.psnr(batch_berlin1_lr_t, batch_berlin1_hr_t, max_val=1.0)
psnr_value
psnr_value.item()

ssim_map = kornia.metrics.ssim(batch_berlin1_lr_t, batch_berlin1_hr_t, window_size=5, max_val=1.0, eps=1e-12)
ssim_map
ssim_map.mean()
ssim_map.mean().item()
