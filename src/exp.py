
def modify_config(args, date, meta_cfg, exp_cfg):

    # init dict for modified args
    exp_dict = {}

    # skip as needed
    if date is None:
        return args, exp_dict

    # reset meta config (scalable)
    exp_dict['contrastive'] = (meta_cfg[0] == 't')              # default: f (normal weights)
    exp_dict['kdl_weight'] = int(meta_cfg[1]) * 5               # default: 2 (10)

    # reset training config
    assert exp_cfg[-4] in ('p', 'c')                            # dataset: p (pascal) | c (coco)
    exp_dict['layers'] = 50 if exp_cfg[-3] == 'm' else 101      # backbone: m (res50) | n (res101)
    exp_dict['shot'] = int(exp_cfg[-2])                         # shot: 1 | 5
    exp_dict['train_split'] = int(exp_cfg[-1])                  # split: 0 | 1 | 2 | 3

    # apply experiment settings
    for k, v in exp_dict.items():
        args.__setattr__(k, v)

    return args, exp_dict
