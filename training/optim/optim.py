import torch

# optimizer
adam = torch.optim.Adam
sgd = torch.optim.SGD
adagrad = torch.optim.Adagrad
rmsprop = torch.optim.RMSprop

# scheduler ,is a part of optimizer,can be used to adjust lr
steplr = torch.optim.lr_scheduler.StepLR
multisteplr = torch.optim.lr_scheduler.MultiStepLR
cosineannealinglr = torch.optim.lr_scheduler.CosineAnnealingLR
reducelronplateau = torch.optim.lr_scheduler.ReduceLROnPlateau
lambdalr = torch.optim.lr_scheduler.LambdaLR
cycliclr = torch.optim.lr_scheduler.CyclicLR 
