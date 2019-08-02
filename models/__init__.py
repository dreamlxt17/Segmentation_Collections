from __future__ import absolute_import

from .unet3d import Unet3D
from .unet3dv2 import Unet3DV2, Unet3Dstage1, Unet3Dstage2
from .spacing_unet import SpacingUnet4, SpacingUnet5, SpacingUnet6
from .spacing_unet3 import SpacingUnet3
from .spacing_unet1 import SpacingUnet0, SpacingUnet1, SpacingUnet2
from .unet3dv3 import Unet3DV3
from .unet_tumor import Unet3D_Tumor, SpacingTumor0, SpacingTumor7, PSPNetTumor, PSPUnetTumor, Unet3D_TumorV2
from .densepspunet import DensePspUnet, DensePspUnetV2, DensePspUnetV3, DensePspUnetV4, DensePspUnetV4_1
from .vnet import DialResUNet as VNet
from .vnet import VNet as VNet0
from .spacing_unet2 import SpacingUnet22_2, SpacingUnet22_4
from .vnet_baseline import VNet00, VNetSimple, VNetStage2
from .hourglass3d import HourglassV1

__factory = {
    'unet3d': Unet3D,
    'unet3dv2': Unet3DV2,
    'spacing_unet1': SpacingUnet1,
    'spacing_unet2': SpacingUnet2,
    'spacing_unet3': SpacingUnet3,
    'spacing_unet4': SpacingUnet4,
    'spacing_unet5': SpacingUnet5,
    'spacing_unet6': SpacingUnet6,
    'spacing_unet0': SpacingUnet0,
    'unet3dv3': Unet3DV3,
    'unet_tumor': Unet3D_Tumor,
    'spacing_tumor7': SpacingTumor7,
    'spacing_tumor0': SpacingTumor0,
    'pspnet_tumor': PSPNetTumor,
    'pspunet_tumor': PSPUnetTumor,
    'densepsp_unet': DensePspUnet,
    'densepsp_unetv2': DensePspUnetV2,
    'densepsp_unetv3': DensePspUnetV3,
    'densepsp_unetv4': DensePspUnetV4,
    'densepsp_unetv4_1': DensePspUnetV4_1,
    'vnet': VNet,
    'vnet0': VNet0,
    'vnet00': VNet00,
    'spacing_unet22_2': SpacingUnet22_2,
    'spacing_unet22_4': SpacingUnet22_4,
    'vnet_simple': VNetSimple,
    'unet_tumor_v2':Unet3D_TumorV2,
    'unet3dstage1': Unet3Dstage1,
    'unet3dstage2': Unet3Dstage2,
    'vnetstage2': VNetStage2,
    'hourglass': HourglassV1
}


def get_names():
    return __factory.keys()


def init_model(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](*args, **kwargs)
