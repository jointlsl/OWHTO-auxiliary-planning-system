from .unet2d import UNet as unet2d
from .unet2plus import NestedUNet as unet2plus
from .attUNet   import AttU_Net   as attUNet
from .DeepPose import DeepPose as deeppose
from .MobileNet import MobileNet


def get_net(s):
    return {
        'unet2d': unet2d,
        'unet++': unet2plus,
        'attunet': attUNet,
        'deeppose': deeppose,
        'mobilenet': MobileNet
    }[s.lower()]

