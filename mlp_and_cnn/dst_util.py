import torch


EXCLUDED_TYPES = (torch.nn.BatchNorm2d, )
LAYER_INDEX=0

def get_weighted_layers_mlp(model, layers=None, linear_layers_mask=None, chain_list=[]):
    global LAYER_INDEX
    
    if layers is None:
        layers = []
    if linear_layers_mask is None:
        linear_layers_mask = []

    
    items = model._modules.items()

    # if i == 0:
    #     items = [("model", model)]
    for layer_name, p in items:
        if layer_name == "last_layer":
            continue
        elif isinstance(p, torch.nn.Linear):
            
            layers.append([p])
            chain_list.append(LAYER_INDEX)
            linear_layers_mask.append(1)
            LAYER_INDEX += 1
        else:
            _, linear_layers_mask = get_weighted_layers_mlp(p, layers=layers, linear_layers_mask=linear_layers_mask, chain_list=chain_list)
    
    return layers, linear_layers_mask

def get_W(model, return_linear_layers_mask=False):
    global LAYER_INDEX

    chain_list = []

    layers, linear_layers_mask = get_weighted_layers_mlp(model, chain_list = chain_list)
    # exit()
    # print(layers)
    # exit()
    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1
        print(layer[idx].weight.shape)
        W.append(layer[idx].weight)

    assert len(W) == len(linear_layers_mask)
    assert len(W) == LAYER_INDEX

    if return_linear_layers_mask:
        return W, linear_layers_mask, chain_list
    return W, chain_list
