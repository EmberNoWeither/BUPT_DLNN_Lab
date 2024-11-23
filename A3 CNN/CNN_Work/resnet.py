from torch import nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4#
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # block的输入层卷积
        self.bn1 = nn.BatchNorm2d(planes)  # 归一化处理，使得不会因数据过大而导致网络性能的不稳定
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  
                               padding=1, bias=False)#block的中间层卷积
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)#block的输出层卷积
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample#判断是否是conv block
        self.stride = stride#不同stage的stride不同，除了stage1的stride为1，其余stage均为2
 
    def forward(self, x):
        residual = x
        # 卷积操作，就是指的是identity block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:  
            residual = self.downsample(x)
        # 相加
        out += residual
        out = self.relu(out)
 
        return out
 
 
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10, in_dims=3):  # block即为Bottleneck模型，layers可控制传入的Bottleneck
        self.inplanes = 64  # 初始输入通道数为64
        super(ResNet, self).__init__()  
        # 把stage前面的卷积处理
        self.conv1 = nn.Conv2d(in_dims, 64, kernel_size=7, stride=2, padding=3,
                                bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        # 64，128，256，512是指扩大4倍之前的维度
        # 四层stage,layer表示有几个block块,可见后3个stage的stride全部为2
        self.layer1 = self._make_layer(block, 64, layers[0]) 
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #最后的池化与全连接
        self.avgpool = nn.AvgPool2d(7)  # 这里默认stride为7
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # _make_layer方法用来构建ResNet网络中的4个blocks
    def _make_layer(self, block, planes, blocks, stride=1):  
        # self.inplanes为上个box_block的输出channel,planes为当前box_block块的输入channel
        downsample = None
        # stride不为1时，残差结构输出纬度变化
        # 输入通道数不等于输出通道数，也需要downsample，即block旁边的支路需要进行卷积
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        #conv block部分
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))  # 将每个blocks的第一个residual结构保存在layers列表中
        self.inplanes = planes * block.expansion#得到第一个 conv block的输出，作为identity的输入

        #identity block部分
        for i in range(1, blocks):  # 该部分是将每个blocks的剩下residual结构保存在layers列表中，这样就完成了一个blocks的构造
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # 前面部分的卷积，不是layer的卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 四个层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        x = self.fc(x)

        return x
    
def resnet50():
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    # 获取特征提取部分
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # 获取分类部分
    classifier = list([model.layer4, model.avgpool])
    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)
    return features, classifier