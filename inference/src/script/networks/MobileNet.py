import torch.nn as nn
from torchvision import models



class MobileNet(nn.Module):
    # def __init__(self, feat_dim=12):
	def __init__(self, nJoints):
		super(MobileNet, self).__init__()
		self.nJoints = nJoints
		self.backbone_net = models.mobilenet_v2(pretrained=True)
		self.feature_extractor = self.backbone_net.features
		self.avgpool = nn.AdaptiveAvgPool2d(1)
		self.fc_pose = nn.Linear(1280, self.nJoints*2)
	##ondef
	def forward(self, Input):
		x = self.feature_extractor(Input)
		x = self.avgpool(x)
		x = x.reshape(x.size(0), -1)
		predict = self.fc_pose(x)
		return predict