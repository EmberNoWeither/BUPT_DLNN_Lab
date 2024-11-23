import numpy as npp
import cupy as np
import os
from datasets_prepare import load_dataset
from optimizer import (Optimizer, Optimizer_Adam, Optimizer_Momentum)
from mlp import MLP
import matplotlib.pyplot as plt
import json
from softmax_classifier import SoftMaxClassifier
from concurrent.futures import ThreadPoolExecutor,wait,ALL_COMPLETED
import matplotlib.ticker as ticker
import math
from tqdm import tqdm

classifier_miniBatch = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_adam = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_momentum = MLP(3, [28*28, 256, 30], isBatchNorm=False)

classifier_miniBatch_withBatchNorm = MLP(3, [28*28, 256, 30], isBatchNorm=True)
classifier_adam_withBatchNorm = MLP(3, [28*28, 256, 30], isBatchNorm=True)
classifier_momentum_withBatchNorm = MLP(3, [28*28, 256, 30], isBatchNorm=True)

classifier_miniBatch_withL2 = MLP(3, [28*28, 256, 30], isBatchNorm=False, isL2=True)
classifier_adam_withL2 = MLP(3, [28*28, 256, 30], isBatchNorm=False, isL2=True)
classifier_momentum_withL2 = MLP(3, [28*28, 256, 30], isBatchNorm=False, isL2=True)

classifier_miniBatch_withL2andBN = MLP(3, [28*28, 256, 30], isBatchNorm=True, isL2=True)
classifier_adam_withL2andBN = MLP(3, [28*28, 256, 30], isBatchNorm=True, isL2=True)
classifier_momentum_withL2andBN = MLP(3, [28*28, 256, 30], isBatchNorm=True, isL2=True)

classifier_miniBatch_withL2andBN_lossL2 = MLP(3, [28*28, 256, 30], isBatchNorm=True, isL2=True, lossf='L2')
classifier_adam_withL2andBN_lossL2 = MLP(3, [28*28, 256, 30], isBatchNorm=True, isL2=True, lossf='L2')
classifier_momentum_withL2andBN_lossL2 = MLP(3, [28*28, 256, 30], isBatchNorm=True, isL2=True, lossf='L2')

softmax_classifier_miniBatch = SoftMaxClassifier(optim='mini-batch')
softmax_classifier_adam = SoftMaxClassifier()
softmax_classifier_momentum = SoftMaxClassifier(optim='momentum', beta=0.9)

classifier_adam_lr5e_1 = MLP(3, [28*28, 256, 30], isBatchNorm=False, lr=5e-1)
classifier_adam_lr5e_2 = MLP(3, [28*28, 256, 30], isBatchNorm=False, lr=5e-2)
classifier_adam_lr5e_3 = MLP(3, [28*28, 256, 30], isBatchNorm=False, lr=5e-3)
classifier_adam_lr5e_4 = MLP(3, [28*28, 256, 30], isBatchNorm=False, lr=5e-4)


softmax_classifier_adam_lossL2 = SoftMaxClassifier(lr=5e-3,lossf='L2')
classifier_adam_lossL2 = MLP(3, [28*28, 256, 30], lossf='L2')

softmax_classifier_adam_lr5e_1 = SoftMaxClassifier(lr=5e-1)
softmax_classifier_adam_lr5e_2 = SoftMaxClassifier(lr=5e-2)
softmax_classifier_adam_lr5e_3 = SoftMaxClassifier(lr=5e-3)
softmax_classifier_adam_lr5e_4 = SoftMaxClassifier(lr=5e-4)

softmax_classifier_adam_beta03 = SoftMaxClassifier(beta=0.3)
softmax_classifier_adam_beta05 = SoftMaxClassifier(beta=0.5)
softmax_classifier_adam_beta07 = SoftMaxClassifier(beta=0.7)
softmax_classifier_adam_beta09 = SoftMaxClassifier(beta=0.9)

softmax_classifier_momentum_beta03 = SoftMaxClassifier(beta=0.3, optim='momentum')
softmax_classifier_momentum_beta05 = SoftMaxClassifier(beta=0.5, optim='momentum')
softmax_classifier_momentum_beta07 = SoftMaxClassifier(beta=0.7, optim='momentum')
softmax_classifier_momentum_beta09 = SoftMaxClassifier(beta=0.9, optim='momentum')

classifier_momentum_lr5e_1 = MLP(3, [28*28, 256, 30], isBatchNorm=False, lr=5e-1)
classifier_momentum_lr5e_2 = MLP(3, [28*28, 256, 30], isBatchNorm=False, lr=5e-2)
classifier_momentum_lr5e_3 = MLP(3, [28*28, 256, 30], isBatchNorm=False, lr=5e-3)
classifier_momentum_lr5e_4 = MLP(3, [28*28, 256, 30], isBatchNorm=False, lr=5e-4)

classifier_miniBatch20 = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_miniBatch50 = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_miniBatch100 = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_miniBatch200 = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_miniBatch300 = MLP(3, [28*28, 256, 30], isBatchNorm=False)

classifier_momentum20 = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_momentum50 = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_momentum100 = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_momentum200 = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_momentum300 = MLP(3, [28*28, 256, 30], isBatchNorm=False)

classifier_adam20 = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_adam50 = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_adam100 = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_adam200 = MLP(3, [28*28, 256, 30], isBatchNorm=False)
classifier_adam300 = MLP(3, [28*28, 256, 30], isBatchNorm=False)

classifier_adam_beta03 = MLP(3, [28*28, 256, 30], isBatchNorm=False, beta=0.3)
classifier_adam_beta05 = MLP(3, [28*28, 256, 30], isBatchNorm=False, beta=0.5)
classifier_adam_beta07 = MLP(3, [28*28, 256, 30], isBatchNorm=False, beta=0.7)
classifier_adam_beta09 = MLP(3, [28*28, 256, 30], isBatchNorm=False, beta=0.9)

classifier_momentum_beta03 = MLP(3, [28*28, 256, 30], isBatchNorm=False, beta=0.3)
classifier_momentum_beta05 = MLP(3, [28*28, 256, 30], isBatchNorm=False, beta=0.5)
classifier_momentum_beta07 = MLP(3, [28*28, 256, 30], isBatchNorm=False, beta=0.7)
classifier_momentum_beta09 = MLP(3, [28*28, 256, 30], isBatchNorm=False, beta=0.9)

mlp_small_size_adam = MLP(3, [28*28, 64, 10], isBatchNorm=False)
mlp_middle_size_adam = MLP(3, [28*28, 128, 25], isBatchNorm=False)
mlp_big_size_adam = MLP(3, [28*28, 256, 30], isBatchNorm=False)
mlp_huge_size_adam = MLP(3, [28*28, 512, 64], isBatchNorm=False)
mlp_Morehuge_size_adam = MLP(3, [28*28, 1024, 128], isBatchNorm=False)
mlp_hugest_size_adam = MLP(3, [28*28, 2048, 256], isBatchNorm=False)

mlp_overFit_size_adam = MLP(3, [28*28, 5096, 128], isBatchNorm=False)
mlp_overFit_size_adam_L2_001 = MLP(3, [28*28, 5096, 128], isBatchNorm=False, isL2=True, lambd=1e-2)
mlp_overFit_size_adam_L2_0001 = MLP(3, [28*28, 5096, 128], isBatchNorm=False, isL2=True, lambd=1e-3)
mlp_overFit_size_adam_L2_00001 = MLP(3, [28*28, 5096, 128], isBatchNorm=False, isL2=True, lambd=1e-4)
mlp_overFit_size_adam_Droup = MLP(3, [28*28, 2048, 256], isBatchNorm=False, isdropout=True, dropout=[0.0,0.1,0.1])

mlp_overFitadam_Droup_01 = MLP(3, [28*28, 5096, 128], isBatchNorm=False, isdropout=True, dropout=[0.0,0.1,0.1])
mlp_overFitadam_Droup_001 = MLP(3, [28*28, 5096, 128], isBatchNorm=False, isdropout=True, dropout=[0.0,0.01,0.01])

mlp_overFit_size_adam_DroupBN = MLP(3, [28*28, 2048, 256], isBatchNorm=True, isdropout=True, dropout=[0.0,0.1,0.1])
mlp_overFit_size_adam_DroupL2 = MLP(3, [28*28, 2048, 256], isBatchNorm=False, isdropout=True, isL2=True, dropout=[0.0,0.1,0.1])
mlp_overFit_size_adam_DroupL2BN = MLP(3, [28*28, 2048, 256], isBatchNorm=True, isdropout=True, isL2=True, dropout=[0.0,0.1,0.1])

mlp_short_length_adam = MLP(2, [28*28, 100], isBatchNorm=False)
mlp_normal_length_adam = MLP(3, [28*28, 100, 70], isBatchNorm=False)
mlp_long_length_adam = MLP(4, [28*28, 100, 70, 50], isBatchNorm=False)

mlp_longest_adam = MLP(5, [28*28, 100, 70, 50, 30], isBatchNorm=False)
mlp_longest_adam_BN = MLP(5, [28*28, 100, 70, 50, 30], isBatchNorm=True)
mlp_longest_adam_Droup = MLP(5, [28*28, 100, 70, 50, 30], isBatchNorm=False, isdropout=True, dropout=[0.0,0.1,0.1,0.1,0.1])
mlp_longest_adam_L2 = MLP(5, [28*28, 100, 70, 50, 30], isBatchNorm=False, isL2=True)

classifier_adam_dropout = MLP(3, [28*28, 256, 30], isBatchNorm=False, isdropout=True, dropout=[0.0,0.1,0.1])

classifiers = {
    # 'classifier_miniBatch' : classifier_miniBatch,
    # 'classifier_adam' : classifier_adam,
    # 'classifier_momentum' : classifier_momentum,
    
    # 'mlp_small_size_adam' : mlp_small_size_adam,
    # 'mlp_middle_size_adam' : mlp_middle_size_adam,
    # 'mlp_big_size_adam' : mlp_big_size_adam,
    # 'mlp_huge_size_adam' : mlp_huge_size_adam,
    # 'mlp_Morehuge_size_adam' : mlp_Morehuge_size_adam,
    # 'mlp_hugest_size_adam' : mlp_hugest_size_adam,
    
    'HUGE_adam' : mlp_overFit_size_adam,
    'HUGE_adam_L2_1e-2' : mlp_overFit_size_adam_L2_001,
    # 'HUGE_adam_L2_1e-3' : mlp_overFit_size_adam_L2_0001,
    'HUGE_adam_L2_1e-4' : mlp_overFit_size_adam_L2_00001,
    # 'mlp_HUGE_adam_Droup' : mlp_overFit_size_adam_Droup,
    # 'mlp_HUGE_adam_DroupBN':mlp_overFit_size_adam_DroupBN,
    # 'mlp_HUGE_adam_DroupL2' : mlp_overFit_size_adam_DroupL2,
    # 'mlp_HUGE_adam_DroupL2BN' : mlp_overFit_size_adam_DroupL2BN
    
    # 'mlp_short_length_adam' : mlp_short_length_adam,
    # 'mlp_normal_length_adam' : mlp_normal_length_adam,
    # 'mlp_long_length_adam' : mlp_long_length_adam,
    
    # 'mlp_longest_adam' : mlp_longest_adam,
    # 'mlp_longest_adam_BN' : mlp_longest_adam_BN,
    # 'mlp_longest_adam_Droup' : mlp_longest_adam_Droup,
    # 'mlp_longest_adam_L2' : mlp_longest_adam_L2,
    
    # 'classifier_miniBatch_withBatchNorm' : classifier_miniBatch_withBatchNorm,
    # 'classifier_adam_withBatchNorm' : classifier_adam_withBatchNorm,
    # 'classifier_momentum_withBatchNorm' : classifier_momentum_withBatchNorm,
    
    # 'classifier_miniBatch_withL2' : classifier_miniBatch_withL2,
    # 'classifier_adam_withL2' : classifier_adam_withL2,
    # 'classifier_momentum_withL2' : classifier_momentum_withL2,
    
    # 'classifier_miniBatch_withL2andBN' : classifier_miniBatch_withL2andBN,
    # 'classifier_adam_withL2andBN' : classifier_adam_withL2andBN,
    # 'classifier_momentum_withL2andBN' : classifier_momentum_withL2andBN,
    # 'classifier_miniBatch_withL2andBN_lossL2' : classifier_miniBatch_withL2andBN_lossL2,
    # 'classifier_adam_withL2andBN_lossL2' : classifier_adam_withL2andBN_lossL2,
    # 'classifier_momentum_withL2andBN_lossL2' : classifier_momentum_withL2andBN_lossL2,
    # 'softmax_classifier_miniBatch' : softmax_classifier_miniBatch,
    # 'softmax_classifier_adam' : softmax_classifier_adam,
    # 'softmax_classifier_momentum' : softmax_classifier_momentum,
    
    # 'classifier_adam_lr5e-1' : classifier_adam_lr5e_1,
    # 'classifier_adam_lr5e-2' : classifier_adam_lr5e_2,
    # 'classifier_adam_lr5e-3' : classifier_adam_lr5e_3,
    # 'classifier_adam_lr5e-4' : classifier_adam_lr5e_4,
    
    # 'classifier_momentum_lr5e_1' : classifier_momentum_lr5e_1,
    # 'classifier_momentum_lr5e_2' : classifier_momentum_lr5e_2,
    # 'classifier_momentum_lr5e_3' : classifier_momentum_lr5e_3,
    # 'classifier_momentum_lr5e_4' : classifier_momentum_lr5e_4,
    
    # 'classifier_miniBatch16' : classifier_miniBatch20,
    # 'classifier_miniBatch32' : classifier_miniBatch50,
    # 'classifier_miniBatch64' : classifier_miniBatch100,
    # 'classifier_miniBatch128' : classifier_miniBatch200,
    # 'classifier_miniBatch256' : classifier_miniBatch300,
    
    # 'classifier_momentum16' : classifier_momentum20,
    # 'classifier_momentum32' : classifier_momentum50,
    # 'classifier_momentum64' : classifier_momentum100,
    # 'classifier_momentum128' : classifier_momentum200,
    # 'classifier_momentum256' : classifier_momentum300,
    
    # 'classifier_adam16' : classifier_adam20,
    # 'classifier_adam32' : classifier_adam50,
    # 'classifier_adam64' : classifier_adam100,
    # 'classifier_adam128' : classifier_adam200,
    # 'classifier_adam256' : classifier_adam300,
    
    # 'softmax_classifier_adam_lr5e_1' : softmax_classifier_adam_lr5e_1,
    # 'softmax_classifier_adam_lr5e_2' : softmax_classifier_adam_lr5e_2,
    # 'softmax_classifier_adam_lr5e_3' : softmax_classifier_adam_lr5e_3,
    # 'softmax_classifier_adam_lr5e_4' : softmax_classifier_adam_lr5e_4,
    
    # 'softmax_classifier_adam_beta03' : softmax_classifier_adam_beta03,
    # 'softmax_classifier_adam_beta05' : softmax_classifier_adam_beta05,
    # 'softmax_classifier_adam_beta07' : softmax_classifier_adam_beta07,
    # 'softmax_classifier_adam_beta09' : softmax_classifier_adam_beta09,
    
    # 'classifier_adam_beta03' : classifier_adam_beta03,
    # 'classifier_adam_beta05' : classifier_adam_beta05,
    # 'classifier_adam_beta07' : classifier_adam_beta07,
    # 'classifier_adam_beta09' : classifier_adam_beta09,
    
    # 'classifier_momentum_beta03' : classifier_momentum_beta03,
    # 'classifier_momentum_beta05' : classifier_momentum_beta05,
    # 'classifier_momentum_beta07' : classifier_momentum_beta07,
    # 'classifier_momentum_beta09' : classifier_momentum_beta09,
    
    # 'softmax_classifier_adam_lossL2' : softmax_classifier_adam_lossL2,
    
    # 'classifier_adam_lossL2' : classifier_adam_lossL2,
    
    # 'classifier_adam_dropout': classifier_adam_dropout,
}


def test(classifier:MLP, x_test=None, y_test=None
         , test_bsz=200, dataset_prepared=False):

    if not dataset_prepared:
        x_train, y_train, x_test, y_test = load_dataset(300, test_bsz)

    Scr = []
    for x, y in zip(x_test, y_test):
        cls = classifier.predict(x, bsz=test_bsz)
        y_true = []
        for yy in y:
            y_true.append(npp.argmax(np.asnumpy(yy)))

        score = 0.0
        for cl, gtd in zip(cls, y_true):
            if cl == gtd:
                score += 1
        score /= len(y_true)
        Scr.append(score)

    Scr = np.array(Scr)
    Scr = np.mean(Scr)
    print("Mean Score: " + str(Scr))
    return Scr


def train(x_train, y_train, x_valid, y_valid, x_test, y_test, classifier : MLP, model_name : str, optim : Optimizer, batch_size=300, test_batch=50, epochs=10):
    
    J = []
    ep = []
    scr = []
    Step = []
    steps = 0
    test_scores = 0.0
    
    for epoch in range(epochs):
        classifier.is_test = False
        with tqdm(total=len(x_train)) as t:
            for x, y in zip(x_train, y_train):
                    t.set_description(model_name + "_Epoch %i" %epoch)
                    classifier.is_test = False
                    steps += 1
                    loss = classifier.train_steps(x, y, batch_size, optim, epoch)
                    t.set_postfix(Loss='%.4f'%loss)
                    J.append(loss)
                    Step.append(steps)
                    
                    t.update(1)

        # if epoch % 5 == 0:
        classifier.is_test = True
        score = test(classifier, x_valid, y_valid, test_batch, True)
        
        # if isinstance(classifier, MLP):
        #     classifier.save_weights(file_name= './' + model_name +'_epoch_'+str(epoch)+'.json')
        scr.append(score)
        ep.append(epoch)
    #     log_vars = {
    #         'steps':steps,
    #         'score':scr,
    #         'epoch':epoch,
    #         'loss':J
    #     }
    # log_vars = json.dumps(log_vars)
    
    # log_File = './'+ model_name + '_train_log.json'
    # with open(log_File, 'a') as f:
    #     f.write(log_vars)
    #     print('Save Log Success!')
    
    classifier.is_test = True
    test_scores = test(classifier, x_test, y_test, test_bsz=1, dataset_prepared=True)

    return Step, J, ep, scr, test_scores


def single_muti_train(epochs=10, batch_size=300, test_batch=50, lr=0.005, png_title=''):
    Steps = []
    Js = []
    eps = []
    scrs = []
    test_scores = []
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_dataset(batch_size, test_batch)
    # x_test, y_test = selfmake_dataset_load()
    
    for classifier_name, classifier in zip(list(classifiers.keys()), list(classifiers.values())):
        if math.fabs(lr-5e-3) > 1e-6:
            classifier.lr = lr
        
        # 进行优化器判断
        if classifier_name.count('miniBatch'):
            optim = Optimizer(classifier, lr=classifier.lr)
        elif classifier_name.count('adam'):
            optim = Optimizer_Adam(classifier, lr=classifier.lr, beta=classifier.beta)
        else:
            optim = Optimizer_Momentum(classifier, lr=classifier.lr, beta=classifier.beta)
            
        Step, J, ep, scr, test_score = train(x_train, y_train, x_valid, y_valid, x_test, y_test, classifier, classifier_name, optim, batch_size=batch_size, epochs=epochs)
        
        Steps.append(Step)
        Js.append(J)
        eps.append(ep)
        scrs.append(scr)
        test_scores.append(test_score)
        
    plt.figure()
    # loss map 
    for Step, J, classifier_name in zip(Steps, Js, list(classifiers.keys())):
        plt.plot(Step, [j.get() for j in J], label=classifier_name)
    plt.legend(loc='upper right')
    plt.xlabel("steps:")
    plt.ylabel("loss:")
    plt.savefig(png_title + "_contrast_train_loss.png")
    plt.show()
    
    plt.figure()
    np.set_printoptions(precision=4)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    # score map
    for ep, scr, classifier_name in zip(eps, scrs, list(classifiers.keys())):
        plt.plot(ep, [sc.get() for sc in scr], label=classifier_name)
        for x, y in zip(ep, scr):
            plt.text(x, y.get(), '%.4f'%y.get(),fontsize=8)
    plt.legend(loc='upper right')
    plt.xlabel("epochs:")
    plt.ylabel("score:")
    
    plt.savefig(png_title + "_contrast_train_scr.png")
    plt.show()
    
    plt.figure()
    classifier_names = list(classifiers.keys())
    plt.plot(range(len(list(classifiers.keys()))), [item.get() for item in test_scores])
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    for x, y in zip(list(range(len(list(classifiers.keys())))), test_scores):
        plt.text(x, y.get(), y.get(),fontsize=8)
    plt.xticks(range(len(list(classifiers.keys()))), classifier_names, fontsize=6)
    plt.legend(loc='upper right')
    plt.xlabel("epochs:")
    plt.ylabel("score:")
    plt.savefig(png_title + "_contrast_test_scr.png")
    plt.show()


def muti_contrast_train(epochs=10, batch_size=300, test_batch=50, lr=0.005, png_title=''):
    Steps = []
    Js = []
    eps = []
    scrs = []
    exec_list = []
    
    if isinstance(batch_size, list):
        Batch_sizes = batch_size.copy()
        X_train, Y_train, X_valid, Y_valid = [], [], [], []
        for batch_size in Batch_sizes:
            x_train, y_train, x_valid, y_valid, x_test, y_test = load_dataset(batch_size, test_batch)
            X_train.append(x_train)
            Y_train.append(y_train)
            X_valid.append(x_valid)
            Y_valid.append(y_valid)
            
        for x_train, y_train, x_valid, y_valid, bsz, classifier_name, classifier in zip(X_train, Y_train, X_valid, Y_valid, Batch_sizes,
                                                                                        list(classifiers.keys()), list(classifiers.values())):
            # x_test, y_test = selfmake_dataset_load()
            pool = ThreadPoolExecutor(max_workers=8)
            
            if math.fabs(lr-5e-3) > 1e-6:
                classifier.lr = lr
            
            # 进行优化器判断
            if classifier_name.count('miniBatch'):
                optim = Optimizer(classifier, lr=classifier.lr)
            elif classifier_name.count('adam'):
                optim = Optimizer_Adam(classifier, lr=classifier.lr, beta=classifier.beta)
            else:
                optim = Optimizer_Momentum(classifier, lr=classifier.lr, beta=classifier.beta)
                
            future = pool.submit(train,x_train, y_train, x_valid, y_valid, x_test, y_test, classifier, classifier_name, optim, batch_size=bsz, epochs=epochs)
            exec_list.append(future)
        wait(exec_list, return_when=ALL_COMPLETED)
            
    else:
        x_train, y_train, x_valid, y_valid, x_test, y_test = load_dataset(batch_size, test_batch)
        # x_test, y_test = selfmake_dataset_load()
        
        pool = ThreadPoolExecutor(max_workers=8)
        idx = 0
        for classifier_name, classifier in zip(list(classifiers.keys()), list(classifiers.values())):
            if math.fabs(lr-5e-3) > 1e-6:
                classifier.lr = lr
            # 进行优化器判断
            if classifier_name.count('miniBatch'):
                optim = Optimizer(classifier, lr=classifier.lr)
            elif classifier_name.count('adam'):
                optim = Optimizer_Adam(classifier, lr=classifier.lr, beta=classifier.beta)
            else:
                optim = Optimizer_Momentum(classifier, lr=classifier.lr, beta=classifier.beta)
                
            future = pool.submit(train,x_train, y_train, x_valid, y_valid, x_test, y_test, classifier, classifier_name, optim, batch_size=batch_size, epochs=epochs)
            exec_list.append(future)
        
        wait(exec_list, return_when=ALL_COMPLETED)
        
    test_scores = []
    for exec in exec_list:
        Step, J, ep, scr, test_score = exec.result()
        
        Steps.append(Step)
        Js.append(J)
        eps.append(ep)
        scrs.append(scr)
        test_scores.append(test_score)
        
    plt.figure()
    # loss map 
    for Step, J, classifier_name in zip(Steps, Js, list(classifiers.keys())):
        plt.plot(Step, [j.get() for j in J], label=classifier_name)
    plt.legend(loc='upper right')
    plt.xlabel("steps:")
    plt.ylabel("loss:")
    plt.savefig(png_title + "_contrast_train_loss.png")
    plt.show()
    
    plt.figure()
    np.set_printoptions(precision=4)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    # score map
    for ep, scr, classifier_name in zip(eps, scrs, list(classifiers.keys())):
        plt.plot(ep, [sc.get() for sc in scr], label=classifier_name)
        for x, y in zip(ep, scr):
            plt.text(x, y.get(), '%.4f'%y.get(),fontsize=8)
    plt.legend(loc='upper right')
    plt.xlabel("epochs:")
    plt.ylabel("score:")
    
    plt.savefig(png_title + "_contrast_train_scr.png")
    plt.show()
    
    plt.figure()
    classifier_names = list(classifiers.keys())
    plt.plot(range(len(list(classifiers.keys()))), [item.get() for item in test_scores])
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    for x, y in zip(list(range(len(list(classifiers.keys())))), test_scores):
        plt.text(x, y.get(), y.get(),fontsize=8)
    plt.xticks(range(len(list(classifiers.keys()))), classifier_names, fontsize=6)
    plt.legend(loc='upper right')
    plt.xlabel("epochs:")
    plt.ylabel("score:")
    plt.savefig(png_title + "_contrast_test_scr.png")
    plt.show()

if __name__ == '__main__':
    muti_contrast_train(15, png_title='MLP_OverL2DroupRG', lr=0.0005, batch_size=64)
    # single_muti_train(15, png_title='MLP_WidthRG', lr=0.0005, batch_size=64)

