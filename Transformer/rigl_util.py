import torch


EXCLUDED_TYPES = (torch.nn.BatchNorm2d, )


def get_weighted_layers(model, i=0, layers=None, linear_layers_mask=None):
    if layers is None:
        layers = []
    if linear_layers_mask is None:
        linear_layers_mask = []

    items = model._modules.items()
    if i == 0:
        items = [(None, model)]

    for layer_name, p in items:
        # print(layer_name, p)
        if isinstance(p, torch.nn.Linear):
            # print(1, layer_name, p)
            # exit()
            layers.append([p])
            linear_layers_mask.append(1)
        elif layer_name == 'generator':
            continue
        else:
            _, linear_layers_mask, i = get_weighted_layers(p, i=i + 1, layers=layers, linear_layers_mask=linear_layers_mask)
    
    return layers, linear_layers_mask, i 



def get_W(model, return_linear_layers_mask=False):
    layers, linear_layers_mask, _ = get_weighted_layers(model)
    # print(layers, linear_layers_mask)
    # exit()
    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1
        # print(idx)
        W.append(layer[idx].weight)
    # exit()

    assert len(W) == len(linear_layers_mask)

    if return_linear_layers_mask:
        return W, linear_layers_mask
    return W
