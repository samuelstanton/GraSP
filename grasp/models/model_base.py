import torch.nn as nn
from collections import OrderedDict
import hydra

from grasp.utils.prune_utils import filter_weights
from grasp.utils.graph_utils import GraphEdge
from grasp.models.primitive_ops import Identity
from grasp.utils.common_utils import try_cuda
from grasp.utils.init_utils import weights_init
from grasp.models import VGG, ResNet
from grasp.models.resnet import BasicBlock


def get_network(hydra_cfg):
    if hydra_cfg.network.type == 'vgg':
        print('Network: VGG, BatchNorm Status: %s' % hydra_cfg.network.params.batchnorm)
        return VGG(dataset=str(hydra_cfg.dataset), **hydra_cfg.network.params)
    elif hydra_cfg.network.type == 'resnet':
        return ResNet(BasicBlock, **hydra_cfg.network.params)
    elif hydra_cfg.network.type == 'graphnet':
        return hydra.utils.instantiate(hydra_cfg.network)
    else:
        raise NotImplementedError('Network unsupported ' + str(hydra_cfg.network))


def get_model_base(hydra_cfg):
    network = get_network(hydra_cfg)
    network_type = hydra_cfg.network.type
    network_depth = hydra_cfg.network.params.depth
    dataset_name = hydra_cfg.dataset.name
    model_base = ModelBase(network_type, network_depth, dataset_name, network)
    model_base = try_cuda(model_base)
    model_base.model.apply(weights_init)
    return model_base


class ModelBase(object):

    def __init__(self, network, depth, dataset, model=None):
        self._network = network
        self._depth = depth
        self._dataset = dataset
        self.model = model
        self.masks = None
        if self.model is None:
            self.model = get_network(network, depth, dataset)

    def get_ratio_at_each_layer(self):
        # assert self.masks is not None, 'Masks should be generated first.'
        res = dict()
        total_weights, total_params = 0, 0
        remained = 0
        # for m in self.masks.keys():
        for m in self.model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, GraphEdge)):
                mask = self.masks.get(m, None) if self.masks else None
                if mask is not None:
                    res[m] = (mask.sum() / mask.numel()).item() * 100
                    total_weights += mask.numel()
                    remained += mask.sum().item()
                else:
                    res[m] = 100.0
                    total_weights += m.weight.numel()
                    remained += m.weight.numel()
            if hasattr(m, 'bias') and m.bias is not None:
                total_params += m.bias.numel()
        total_params += total_weights

        res['ratio'] = remained/total_weights * 100
        res['total_weights'] = total_weights
        res['total_params'] = total_params
        res['remaining_params'] = remained
        return res

    @property
    def num_params(self):
        num_params = 0
        for m in self.model.modules():
            if hasattr(m, 'weight'):
                if m.weight is not None:
                    num_params += m.weight.numel()
            if hasattr(m, 'bias'):
                if m.bias is not None:
                    num_params += m.bias.numel()
        return num_params

    @property
    def num_ops(self):
        num_layers = 0
        op_classes = (nn.Linear, nn.Conv2d, nn.AvgPool2d, Identity)
        for m in self.model.modules():
            if isinstance(m, op_classes):
                num_layers += 1
        return num_layers

    def get_unmasked_weights(self):
        """Return the weights that are unmasked.
        :return dict, key->module, val->list of weights
        """
        assert self.masks is not None, 'Masks should be generated first.'
        res = dict()
        for m in self.masks.keys():
            res[m] = filter_weights(m.weight, self.masks[m])
        return res

    def get_masked_weights(self):
        """Return the weights that are masked.
        :return dict, key->module, val->list of weights
        """
        assert self.masks is not None, 'Masks should be generated first.'
        res = dict()
        for m in self.masks.keys():
            res[m] = filter_weights(m.weight, 1-self.masks[m])
        return res

    def register_mask(self, masks=None):
        # self.masks = None
        self.unregister_mask()
        if masks is not None:
            self.masks = masks
        assert self.masks is not None, 'Masks should be generated first.'
        for m in self.masks.keys():
            m.register_forward_pre_hook(self._forward_pre_hooks)

    def unregister_mask(self):
        for m in self.model.modules():
            m._backward_hooks = OrderedDict()
            m._forward_pre_hooks = OrderedDict()

    def _forward_pre_hooks(self, m, input):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            # import pdb; pdb.set_trace()
            mask = self.masks[m]
            m.weight.data.mul_(mask)
        else:
            raise NotImplementedError('Unsupported ' + m)

    def get_name(self):
        return '%s_%s%s' % (self._dataset, self._network, self._depth)

    def train(self):
        self.model = self.model.train()
        return self

    def eval(self):
        self.model = self.model.eval()
        return self

    def cpu(self):
        self.model = self.model.cpu()
        return self

    def cuda(self):
        self.model = self.model.cuda()
        return self

