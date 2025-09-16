
DINOV3_MEAN = (0.430, 0.411, 0.296)
DINOV3_STD = (0.213, 0.156, 0.143)
DINOV3_MEAN_STD = {
    'mean': DINOV3_MEAN,
    'std': DINOV3_STD
}


def make_transform(resize_size: int = 224):
    import torchvision.transforms as transforms

    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    normalize = transforms.Normalize(
        mean=DINOV3_MEAN,
        std=DINOV3_STD,
    )
    return transforms.Compose([to_tensor, resize, normalize])


tfm = make_transform()
