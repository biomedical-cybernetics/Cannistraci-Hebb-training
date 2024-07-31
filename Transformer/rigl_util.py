import torch


EXCLUDED_TYPES = (torch.nn.BatchNorm2d, )
LAYER_INDEX=0

def get_weighted_layers_Transformer(model, layers=None, linear_layers_mask=None, qk_chain_list=[], chain_list=[]):
    global LAYER_INDEX
    
    if layers is None:
        layers = []
    if linear_layers_mask is None:
        linear_layers_mask = []

    
    items = model._modules.items()

    # if i == 0:
    #     items = [("model", model)]
    for layer_name, p in items:
        if layer_name == "linear_keys":
            qk_chain_list.append([LAYER_INDEX])
        elif layer_name == "linear_query":
            qk_chain_list[-1].append(LAYER_INDEX)
        
        elif layer_name in ["linear_values", "w_1"]:
            chain_list.append([LAYER_INDEX])

        elif layer_name in ["final_linear", "w_2"]:
            chain_list[-1].append(LAYER_INDEX)

        # else:
        #     print(f"layer name: {layer_name} not in chain lists")

        if isinstance(p, torch.nn.Linear):
            layers.append([p])
            linear_layers_mask.append(1)
            
            LAYER_INDEX += 1

        elif layer_name == 'generator':
            continue
        else:
            _, linear_layers_mask = get_weighted_layers_Transformer(p, layers=layers, linear_layers_mask=linear_layers_mask, qk_chain_list=qk_chain_list, chain_list=chain_list)
    
    return layers, linear_layers_mask

def get_W(model, return_linear_layers_mask=False):
    global LAYER_INDEX

    qk_chain_list = []
    chain_list = []

    layers, linear_layers_mask = get_weighted_layers_Transformer(model, qk_chain_list = qk_chain_list, chain_list = chain_list)
    
    W = []
    for layer in layers:
        idx = 0 if hasattr(layer[0], 'weight') else 1

        W.append(layer[idx].weight)

    assert len(W) == len(linear_layers_mask)
    assert len(W) == LAYER_INDEX

    if return_linear_layers_mask:
        return W, linear_layers_mask, chain_list, qk_chain_list
    return W, chain_list, qk_chain_list
