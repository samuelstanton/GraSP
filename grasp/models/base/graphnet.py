import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from .graph_utils import GraphEdge, GraphLayer
from .primitive_ops import Identity, FactorizedReduce, ReLUConvBN
from .init_utils import weights_init


__all__ = ['graphnet']
_AFFINE = True


def graphnet(depth=32, dataset='cifar10'):
    assert (depth - 2) % 6 == 0, 'Depth must be = 6n + 2, got %d' % depth
    n = (depth - 2) // 6
    if dataset == 'cifar10':
        num_classes = 10
    elif dataset == 'cifar100':
        num_classes = 100
    elif dataset == 'tiny_imagenet':
        num_classes = 200
    elif dataset == 'imagenet':
        num_classes = 1000
    else:
        raise NotImplementedError('Dataset [%s] is not supported.' % dataset)
    return GraphNet([n]*3, num_classes)


class GraphNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10):
        super().__init__()
        num_planes = [32, 64, 128]
        num_nodes = [3 * n for n in num_blocks]
        self.in_planes = num_planes[0]

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(self.in_planes, affine=_AFFINE)
        self.stage1 = GraphBlock(num_nodes[0], num_planes[0], num_planes[0])
        self.stage2 = GraphBlock(num_nodes[1], num_planes[0], num_planes[1], downsample=True)
        self.stage3 = GraphBlock(num_nodes[2], num_planes[1], num_planes[2], downsample=True)
        self.linear = nn.Linear(num_planes[-1], num_classes)
        self.apply(weights_init)

    def forward(self, x):
        out = F.relu(self.bn(self.conv1(x)))
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class GraphBlock(torch.nn.Sequential):
    def __init__(self, num_nodes, in_planes, planes, downsample=False):
        modules = []
        if downsample:
            modules.extend([
                ("downsample_conv", torch.nn.Conv2d(in_planes, planes, kernel_size=1, stride=2, bias=False)),
                ("downsample_bn", torch.nn.BatchNorm2d(planes, affine=False))
            ])
        graph_layer = make_graph_layer(num_nodes, planes)
        modules.append(("graph_layer", graph_layer))
        super().__init__(OrderedDict(modules))


def make_graph_layer(num_nodes, planes):
    edge_count = 0
    adj_dict = {}
    edges = []
    op_factory_dict = make_factory_dict(planes, planes, 1)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            adj_dict.setdefault(j, [])
            adj_dict[j].append((i, edge_count))

            op_dict = {op_name: d['cls'](**d['kwargs']) for op_name, d in op_factory_dict.items()}
            edges.append(GraphEdge(op_dict))
            edge_count += 1
    return GraphLayer(edges, adj_dict)


def make_factory_dict(c_in, c_out, stride):
    op_dict = OrderedDict([
        # ('none', {
        #     'cls': primitive_ops.ZeroOp,
        #     'kwargs': dict(strides=stride)
        # }),
        ('avg_pool_3x3', {
            'cls': torch.nn.AvgPool2d,
            'kwargs': dict(kernel_size=3, stride=stride, padding=1, count_include_pad=False)
        }),
        ('skip_connect', {
            'cls': Identity if stride == 1 else FactorizedReduce,
            'kwargs': {} if stride == 1 else dict(in_channels=c_in, out_channels=c_out)
        }),
        ('nor_conv_1x1', {
            'cls': ReLUConvBN,
            'kwargs': dict(in_channels=c_in, out_channels=c_out, stride=stride,
                           kernel_size=1, padding=0, dilation=1, bias=False, affine=False)
        }),
        ('nor_conv_3x3', {
            'cls': ReLUConvBN,
            'kwargs': dict(in_channels=c_in, out_channels=c_out, stride=stride,
                           kernel_size=3, padding=1, dilation=1, bias=False, affine=False)
        }),
    ])
    return op_dict


def get_rand_op_masks(graph_net, target_ratio):
    op_masks = OrderedDict()
    for m in graph_net.modules():
        if isinstance(m, GraphLayer):
            for edge in m.edges:
                op_masks[edge] = (torch.rand(edge.num_ops) >= target_ratio)
    return op_masks


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('graphnet'):
            print(net_name)
            test(globals()[net_name]())
            print()