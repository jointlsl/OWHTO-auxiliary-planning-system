
from .unet2d import UNet
from .DeepPose import DeepPose
from .MobileNet import MobileNet
from .StackedHourGlass import StackedHourGlass


def get_net(s):
    return {
        'unet2d': UNet,
        'deeppose': DeepPose,
        'mobilenet': MobileNet,
        'stackedhourglass': StackedHourGlass,
    }[s.lower()]