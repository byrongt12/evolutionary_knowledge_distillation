import torchvision.transforms as transforms


# transforms.RandomRotation(degrees=(0, 180))
# transforms.RandomPerspective(distortion_scale=0.6, p=1.0)
# transforms.ColorJitter(brightness=.5, hue=.3)
# transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75))
# transforms.RandomAutocontrast()
# transforms.RandomEqualize()
# transforms.RandAugment()
# transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))
# transforms.RandomEqualize()
# transforms.RandomAdjustSharpness(sharpness_factor=2)
# transforms.RandomHorizontalFlip(),
# transforms.RandomCrop(32, padding=4),
# transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
# transforms.ColorJitter(brightness=0.2, contrast=0.1, saturation=0.2, hue=0.2),


# transforms.RandomHorizontalFlip(),
# transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
# transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=[-0.01, 0.01]),
# transforms.RandomAffine(degrees=(-15, 15), translate=(0, 0.1), scale=(0.9, 1)),


def trainingAugmentation():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 1)),
        transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=[-0.01, 0.01]),
        transforms.RandomAffine(degrees=(-15, 15), translate=(0, 0.1), scale=(0.9, 1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])

    return transform


def testingAugmentation():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761))
    ])

    return transform
