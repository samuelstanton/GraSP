import torch
import torchvision
import torchvision.transforms as transforms
import os
import hydra


def get_transforms(dataset):
    transform_train = None
    transform_test = None
    if dataset == 'mnist':
        # transforms.Normalize((0.1307,), (0.3081,))
        t = transforms.Normalize((0.5,), (0.5,))
        transform_train = transforms.Compose([transforms.ToTensor(),t
                                              ])

        transform_test = transforms.Compose([transforms.ToTensor(),
                                             t])

    if dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    if dataset == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])

    if dataset == 'cinic-10':
        # cinic_directory = '/path/to/cinic/directory'
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cinic_mean, cinic_std)])

    if dataset == 'tiny_imagenet':
        tiny_mean = [0.48024578664982126, 0.44807218089384643, 0.3975477478649648]
        tiny_std = [0.2769864069088257, 0.26906448510256, 0.282081906210584]
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(tiny_mean, tiny_std)])

    if dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    assert transform_test is not None and transform_train is not None, 'Error, no dataset %s' % dataset
    return transform_train, transform_test


def get_dataloader(dataset_name, train_batch_size, test_batch_size, num_workers=2, dataset_dir='./data',
                   subsample_ratio=1.0, **kwargs):
    transform_train, transform_test = get_transforms(dataset_name)
    dataset_dir = os.path.join(hydra.utils.get_original_cwd(), dataset_dir)
    dataset_dir = os.path.normpath(dataset_dir)

    trainset, testset = None, None
    if dataset_name == 'mnist':
        trainset = torchvision.datasets.MNIST(root=dataset_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.MNIST(root=dataset_dir, train=False, download=True, transform=transform_test)

    if dataset_name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=dataset_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=dataset_dir, train=False, download=True, transform=transform_test)

    if dataset_name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(root=dataset_dir, train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=dataset_dir, train=False, download=True, transform=transform_test)

    if dataset_name == 'cinic-10':
        trainset = torchvision.datasets.ImageFolder(dataset_dir + '/cinic-10/trainval', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(dataset_dir + '/cinic-10/test', transform=transform_test)

    if dataset_name == 'tiny_imagenet':
        num_workers = 16
        trainset = torchvision.datasets.ImageFolder(dataset_dir + '/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(dataset_dir + '/val', transform=transform_test)

    if dataset_name == 'imagenet':
        num_workers = 16
        trainset = torchvision.datasets.ImageFolder(dataset_dir + '/train', transform=transform_train)
        testset = torchvision.datasets.ImageFolder(dataset_dir + '/val', transform=transform_test)

    assert trainset is not None and testset is not None, 'Error, no dataset %s' % dataset_name

    if subsample_ratio < 1.0:
        num_train = int(len(trainset) * subsample_ratio)
        num_unused = len(trainset) - num_train
        assert num_train > 0
        trainset, _ = torch.utils.data.random_split(trainset, [num_train, num_unused])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,
                                              num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,
                                             num_workers=num_workers)

    return trainloader, testloader