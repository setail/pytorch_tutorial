import torch

if torch.cuda.is_available():
    device = torch.device('cuda')
    # 创建时指定device
    x = torch.ones(5, device=device)
    # 创建时移动到device
    y = torch.ones(5).to(device)
    # 创建时移动到gpu上
    # x = torch.ones(5).cuda()
    # x = torch.ones(5).cpu()

    # 两个GPU的tensor相加，还是GPU tensor
    z = x + y
    print(z.numpy())