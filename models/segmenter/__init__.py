# from .git_submodule.segm.utils import distributed
# import git_submodule.segm.utils.torch as ptu

from .segmenter_git_submodule.segm.config import load_config

from .segmenter_git_submodule.segm.model.factory import create_segmenter

def segmenter(
    img_height,
    img_width,
    backbone="vit_tiny_patch16_384",
    dropout=0.0,
    normalization=None,
    drop_path = 0.1,
    n_classes=4,
    decoder="mask_transformer",
):
    # # start distributed mode
    # ptu.set_gpu_mode(True)
    # distributed.init_process()

    # set up configuration
    cfg = load_config()
    model_cfg = cfg["model"][backbone]

    if "mask_transformer" in decoder:
        decoder_cfg = cfg["decoder"]["mask_transformer"]

    # model config
    model_cfg["image_size"] = (img_height, img_width)
    model_cfg["backbone"] = backbone
    model_cfg["dropout"] = dropout
    model_cfg["drop_path_rate"] = drop_path
    decoder_cfg["name"] = decoder
    model_cfg["decoder"] = decoder_cfg

    if normalization:
        model_cfg["normalization"] = normalization

    # model
    net_kwargs = model_cfg
    net_kwargs["n_cls"] = n_classes
    model = create_segmenter(net_kwargs)

    return model