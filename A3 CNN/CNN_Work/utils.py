import matplotlib.pyplot as plt
import torch
from typing import List
from runner import Runner
import os
from os.path import join as pjoin
import sys

def draw_module_result(runner_list : List[Runner], png_title, use_kf=True):
    
    save_path = pjoin('./', png_title)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    
    train_losses_list = []
    train_accs_list = []
    val_losses_list = []
    val_accs_list = []
    test_losses_list = []
    test_accs_list = []
    for runner in runner_list:
        a, b = runner._get_train_result()
        c, d = runner._get_valid_result()
        e, f = runner._get_test_result()
        train_losses_list.append(a)
        train_accs_list.append(b)
        val_losses_list.append(c)
        val_accs_list.append(d)
        test_losses_list.append(e)
        test_accs_list.append(f)
        
        
    if use_kf:
        plt.figure()
        for val_loss, runner in zip(val_losses_list, runner_list):
            plt.plot(list(range(len(val_loss))), val_loss, label=runner.get_model_name())
            for x, y in zip(list(range(len(val_loss))), val_loss):
                plt.text(x, y, '%.4f'%y,fontsize=8)
        plt.legend(loc='upper right')
        plt.ylabel("loss:")
        plt.savefig(pjoin(save_path, png_title + "_Kfvalid_loss.png"))
        plt.show()
        
        plt.figure()
        for val_acc, runner in zip(val_accs_list, runner_list):
            plt.plot(list(range(len(val_acc))), val_acc, label=runner.get_model_name())
            for x, y in zip(list(range(len(val_acc))), val_acc):
                plt.text(x, y, '%.4f'%y,fontsize=8)
        plt.legend(loc='upper right')
        plt.ylabel("score:")
        plt.savefig(pjoin(save_path, png_title + "_Kfvalid_acc.png"))
        plt.show()
        
        return
    
    for train_loss, valid_loss, runner in zip(train_losses_list, val_losses_list, runner_list):
        plt.plot([i for i in range(runner.epochs)], train_loss, label=runner.get_model_name()+"_Train")
        plt.plot([i for i in range(runner.epochs)], valid_loss, label=runner.get_model_name()+"_V")
        if runner.epochs < 10:
            for x, y in zip([i for i in range(runner.epochs)], train_loss):
                plt.text(x, y, '%.4f'%y,fontsize=8)
            for x, y in zip([i for i in range(runner.epochs)], valid_loss):
                plt.text(x, y, '%.4f'%y,fontsize=8)
    plt.legend(loc='upper right')
    # plt.xlabel("epochs:")
    plt.ylabel("loss:")
    plt.savefig(pjoin(save_path, png_title + "_contrast_train&valid_loss.png"))
    plt.show()
    
    
    plt.figure()
    #train acc map
    for scr, runner in zip(train_accs_list, runner_list):
        plt.plot([i for i in range(runner.epochs)], scr, label=runner.get_model_name())
        if runner.epochs < 10:
            for x, y in zip([i for i in range(runner.epochs)], scr):
                plt.text(x, y, '%.4f'%y,fontsize=8)
    plt.legend(loc='upper right')
    # plt.xlabel("epochs:")
    plt.ylabel("acc:")
    plt.savefig(pjoin(save_path, png_title + "_contrast_train_scr.png"))
    plt.show()
    
    
    plt.figure()
    #train-valid acc map
    for train_scr, val_scr, runner in zip(train_accs_list, val_accs_list, runner_list):
        plt.plot([i for i in range(runner.epochs)], train_scr, label=runner.get_model_name()+"_Train")
        plt.plot([i for i in range(runner.epochs)], val_scr, label=runner.get_model_name()+"_V")
        if runner.epochs < 10:
            for x, y in zip([i for i in range(runner.epochs)], train_scr):
                plt.text(x, y, '%.4f'%y,fontsize=8)
            for x, y in zip([i for i in range(runner.epochs)], val_scr):
                plt.text(x, y, '%.4f'%y,fontsize=8)
    plt.legend(loc='upper right')
    # plt.xlabel("epochs:")
    plt.ylabel("score:")
    plt.savefig(pjoin(save_path, png_title + "_contrast_train&valid_scr.png"))
    plt.show()
    
    
    plt.figure()
    plt.plot([i for i in range(len(runner_list))], test_accs_list)
    for x, y in zip([i for i in range(len(runner_list))], test_accs_list):
        plt.text(x, y[0], '%.4f'%y[0],fontsize=8)
    plt.xticks(range(len(runner_list)), [runn.get_model_name() for runn in runner_list], fontsize=6)
    plt.xlabel("models:")
    plt.ylabel("score:")
    plt.savefig(pjoin(save_path, png_title + "_contrast_test_scr.png"))
    plt.show()