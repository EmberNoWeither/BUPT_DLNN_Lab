from runner import Runner
from cnn_module import LeNet, AlexNet, AGGNet
from resnet import ResNet, Bottleneck, BasicBlock
from utils import draw_module_result

lenet = LeNet(1)
lenet_bl = LeNet(1, 0.2)
lenet_cl = LeNet(1, 0.5)

lenet_drop = LeNet(1, 0.2)
alex = AlexNet(1,0.5)
alex_nodrop = AlexNet(1,droup_rate=0.0)
alex_2 = AlexNet(1, 0.0)

lenet_3 = LeNet(3)
alex_3 = AlexNet(3)
alex_4 = AlexNet(3)

resnet50 = ResNet(Bottleneck, [3, 4, 6, 3])

resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])
resnet18_2 = ResNet(BasicBlock, [2, 2, 2, 2])

agg = [resnet18, alex_4]
agg = AGGNet(agg)

epochs=10

# runner_lenet = Runner(
#     module=lenet,
#     batch_size=128,
#     num_workers=32,
#     lr=5e-4,
#     resize=32,
#     epochs=epochs,
#     datasets='MINIST',
#     set_model_name='LeNet_1'
# )

# runner_lenet2 = Runner(
#     module=lenet_bl,
#     batch_size=128,
#     num_workers=32,
#     lr=5e-4,
#     resize=32,
#     epochs=epochs,
#     datasets='MINIST',
#     set_model_name='LeNet_2'
# )

# runner_lenet3 = Runner(
#     module=lenet_cl,
#     batch_size=128,
#     num_workers=32,
#     lr=5e-4,
#     resize=32,
#     epochs=epochs,
#     datasets='MINIST',
#     set_model_name='LeNet_3'
# )

# runner_alex_3 = Runner(
#     module=alex,
#     batch_size=128,
#     num_workers=32,
#     lr=5e-4,
#     resize=224,
#     epochs=epochs,
#     set_model_name='Alex_3',
#     kf=False,
#     weight_decay=0.001,
#     datasets='MINIST'
# )

# runner_alex_2 = Runner(
#     module=alex_2,
#     batch_size=128,
#     num_workers=32,
#     lr=5e-4,
#     resize=224,
#     epochs=epochs,
#     kf=False,
#     set_model_name='Alex_2',
#     weight_decay=0.005,
#     datasets='MINIST'
# )

# runner_alex_1 = Runner(
#     module=alex_nodrop,
#     batch_size=128,
#     num_workers=32,
#     lr=5e-4,
#     resize=224,
#     epochs=epochs,
#     kf=False,
#     set_model_name='Alex_1',
#     datasets='MINIST'
# )

# runner_lenet_cifar = Runner(
#     module=lenet_3,
#     batch_size=128,
#     num_workers=32,
#     lr=5e-4,
#     resize=32,
#     epochs=epochs,
#     kf=False,
#     # datasets='MINIST',
#     set_model_name='LeNet'
# )

runner_alex_cifar = Runner(
    module=alex_3,
    batch_size=128,
    num_workers=32,
    lr=5e-4,
    resize=224,
    epochs=epochs,
    kf=False,
    set_model_name='Alex',
    # datasets='MINIST'
)

runer_res18_cifar = Runner(
    module=resnet18_2,
    batch_size=128,
    num_workers=32,
    lr=5e-4,
    resize=224,
    epochs=epochs,
    kf=False,
    set_model_name='Resnet18',
    # datasets='MINIST'
)

runner_resnet_cifar = Runner(
    module=resnet50,
    batch_size=128,
    num_workers=32,
    lr=5e-4,
    resize=224,
    epochs=epochs,
    kf=False,
    set_model_name='Resnet50',
    # datasets='MINIST'
)

runner_agg_cifar = Runner(
    module=agg,
    batch_size=128,
    num_workers=32,
    lr=5e-4,
    resize=224,
    epochs=epochs,
    kf=False,
    set_model_name='averageNet',
)

runners = []

# runners.append(runner_lenet)
# runners.append(runner_lenet2)
# runners.append(runner_lenet3)
# runners.append(runner_lenet_cifar)
runners.append(runner_alex_cifar)
runners.append(runer_res18_cifar)
runners.append(runner_resnet_cifar)
runners.append(runner_agg_cifar)
# runners.append(runner_alex_3)

if __name__ == '__main__':
    for runner in runners:
        runner.train()
        
    draw_module_result(runners, 'AGG-CIFAR', False)
    