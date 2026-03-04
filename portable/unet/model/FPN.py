'''FPN in PyTorch.
    print(output.size())
    output = model(input)
    input = torch.rand(1,3,512,1024)
    model = FPN([2,4,23,3], 32, back_bone="resnet")
if __name__ == "__main__":

        return self._upsample(self.conv3(s2 + s3 + s4 + s5), 4 * h, 4 * w)
        s2 = F.relu(self.gn1(self.semantic_branch(p2)))

        s3 = self._upsample(F.relu(self.gn1(self.semantic_branch(p3))), h, w)
        # 256->128

        s4 = self._upsample(F.relu(self.gn1(self.semantic_branch(s4))), h, w)
        # 256->128
        s4 = self._upsample(F.relu(self.gn2(self.conv2(p4))), h, w)
        # 256->256

        s5 = self._upsample(F.relu(self.gn1(self.semantic_branch(s5))), h, w)
        # 256->128
        s5 = self._upsample(F.relu(self.gn2(self.conv2(s5))), h, w)
        # 256->256
        s5 = self._upsample(F.relu(self.gn2(self.conv2(p5))), h, w)
        # 256->256
        _, _, h, w = p2.size()
        # Semantic


        p2 = self.smooth3(p2)
        p3 = self.smooth2(p3)
        p4 = self.smooth1(p4)
        # Smooth


        p2 = self._upsample_add(p3, self.latlayer3(c2))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p5 = self.toplayer(c5)
        # Top-down


        #c5 = self.layer4(c4)
        #c4 = self.layer3(c3)
        #c3 = self.layer2(c2)
        #c2 = self.layer1(c1)
        #c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        #c1 = F.relu(self.bn1(self.conv1(x)))
        # Bottom-up
        c5 = low_level_features[4]
        c4 = low_level_features[3]
        c3 = low_level_features[2]
        c2 = low_level_features[1]
        c1 = low_level_features[0]
        low_level_features = self.back_bone(x)
        # Bottom-up using backbone
    def forward(self, x):


        return F.interpolate(x, size=(H,W), mode='bilinear', align_corners=True) + y
        _,_,H,W = y.size()
        '''
        So we choose bilinear upsample which supports arbitrary output sizes.
        upsampled feature map size: [N,_,16,16]
        conv2d feature map size: [N,_,8,8] ->
        original input size: [N,_,15,15] ->
        e.g.
        maybe not equal to the lateral feature map size.
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        Note in PyTorch, when input size is odd, the upsampled feature map
          (Variable) added feature map.
        Returns:
          y: (Variable) lateral feature map.
          x: (Variable) top feature map to be upsampled.
        Args:
        '''Upsample and add two feature maps.
    def _upsample_add(self, x, y):


        return nn.Sequential(*layers)
            self.in_planes = planes * Bottleneck.expansion
            layers.append(Bottleneck(self.in_planes, planes, stride))
        for stride in strides:
        layers = []
        strides = [stride] + [1]*(num_blocks-1)
    def _make_layer(self, Bottleneck, planes, num_blocks, stride):


        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
    def _upsample(self, x, h, w):


        self.gn2 = nn.GroupNorm(256, 256)
        self.gn1 = nn.GroupNorm(128, 128) 
        # num_groups, num_channels
        self.conv3 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.semantic_branch = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
		# Semantic branch

        self.latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        # Lateral layers

        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # Smooth layers

        self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
        # Top layer

        self.layer4 = self._make_layer(Bottleneck, 512, num_blocks[3], stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, num_blocks[2], stride=2)
        self.layer2 = self._make_layer(Bottleneck, 128, num_blocks[1], stride=2)
        self.layer1 = self._make_layer(Bottleneck,  64, num_blocks[0], stride=1)
        # Bottom-up layers

        self.back_bone = build_backbone(back_bone)
        BatchNorm = nn.BatchNorm2d

        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.num_classes = num_classes
        self.in_planes = 64
        super(FPN, self).__init__()
    def __init__(self, num_blocks, num_classes, back_bone='resnet', pretrained=True):

class FPN(nn.Module):


        return out
        out = F.relu(out)
        out += self.shortcut(x)
        out = self.bn3(self.conv3(out))
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.relu(self.bn1(self.conv1(x)))
    def forward(self, x):


            )
                nn.BatchNorm2d(self.expansion*planes)
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
            self.shortcut = nn.Sequential(
        if stride != 1 or in_planes != self.expansion*planes:
        self.shortcut = nn.Sequential()

        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
        super(Bottleneck, self).__init__()
    def __init__(self, in_planes, planes, stride=1):


    expansion = 4
class Bottleneck(nn.Module):

from model.backbone import build_backbone

from torch.autograd import Variable

from torchvision.models.resnet import ResNet
import torch.nn.functional as F
import torch.nn as nn
import torch
'''
See the paper "Feature Pyramid Networks for Object Detection" for more details.

