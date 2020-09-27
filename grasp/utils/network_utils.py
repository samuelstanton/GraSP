import torch
from torch import nn
from tqdm import tqdm
from grasp.utils.common_utils import try_cuda
from grasp.utils.graph_utils import GraphEdge


def stablize_bn(net, trainloader, device='cuda'):
    """Iterate over the dataset for stabilizing the
    BatchNorm statistics.
    """
    net = net.train()
    for batch, (inputs, _) in enumerate(trainloader):
        inputs = inputs.to(device)
        net(inputs)


def get_grad_norm(net, dataset, norm_factor=1.):
    module_types = (nn.Conv2d, nn.Linear, GraphEdge)
    weights = []
    for m in net.modules():
        if isinstance(m, module_types):
            weights.append(m.weight)
    grads = [torch.zeros_like(w) for w in weights]
    net.zero_grad()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
    for inputs, targets in tqdm(dataloader):
        inputs, targets = try_cuda(inputs), try_cuda(targets)
        logits = net(inputs)
        loss = nn.functional.cross_entropy(logits, targets, reduction='sum') / len(dataset)
        batch_grads = get_gradients(loss, weights, allow_unused=True, create_graph=False)
        for i, batch_grad in enumerate(batch_grads):
            grads[i] += batch_grad
    abs_grad_norm = torch.stack([grad.pow(2).mean() for grad in grads]).mean().sqrt()
    rel_grad_norm = (abs_grad_norm / norm_factor).item()
    return rel_grad_norm


def get_gradients(output, weights, allow_unused, create_graph):
    grads = list(torch.autograd.grad(output, weights, allow_unused=allow_unused, create_graph=create_graph))
    for i, weight in enumerate(weights):
        if grads[i] is None:
            grads[i] = torch.zeros_like(weight)
    return grads
