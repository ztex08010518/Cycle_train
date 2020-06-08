"""
Author: Benny
Date: Nov 2019
Instance accuracy ... classification accuracy
Class accuracy ... consider the number of each class
"""
from dataset import ModelNetDataLoader
import os
import argparse
import logging
import sys
import torch
import numpy as np
import provider
from tqdm import tqdm
from models.pointnet_cls import *

# Official
parser = argparse.ArgumentParser("PointNet")
parser.add_argument("--batch_size", type=int, default=24, help="Batch size in training [default: 24]")
parser.add_argument("--model", default="pointnet_cls", help="Model name [default: pointnet_cls]")
parser.add_argument("--epoch",  default=200, type=int, help="Number of epoch in training [default: 200]")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate in training [default: 0.001]")
parser.add_argument("--gpu", type=str, default="0", help="Specify gpu device [default: 0]")
parser.add_argument("--num_in_points", type=int, default=1024, help="Number of input points [default: 1024]")
parser.add_argument("--num_out_points", type=int, default=1024, help="Number of output points [default: 1024]")
parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer for training [default: Adam]")
parser.add_argument("--output_dir", type=str, default="/eva_data/psa/code/outputs/Cycle", help="Experiment root")
parser.add_argument("--decay_rate", type=float, default=1e-4, help="Decay rate [default: 1e-4]")
# MSN_PointNet
parser.add_argument("--sparsify_mode", type=str, default="zorder", choices=["zorder", "multizorder"], help="Sparsify mode")
parser.add_argument("--dataset_mode", type=str, default="ModelNet40", choices=["ModelNet40", "ModelNet10"], help="Dataset mode. PLZ choose in [ModelNet40, ModelNet10]")
args = parser.parse_args()

# HYPER PARAMETER
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu # GPU devices
num_classes = {"ModelNet40": 40, "ModelNet10": 10}
num_class = num_classes[args.dataset_mode] # Number of class (default for ModelNet40)

train_save = {
        "partial":{"instance_acc": [], "cls": [], "EMD": [], "expansion": []},
        "recon":{"instance_acc": [], "cls": [], "EMD": [], "expansion": []}
} # Save training accuracy and loss
test_save = {
        "partial":{"instance_acc": [], "class_acc": [], "cls": [], "EMD": []},
        "recon":{"instance_acc": [], "class_acc": [], "cls": [], "EMD": []}
} # Save testing accuracy and loss

def create_output_dir():
    # Create output directry according to sparsify mode, normalize, trans_feat
    if args.output_dir == "/eva_data/psa/code/outputs/Cycle":
        mode_dir = args.sparsify_mode + "_" + str(args.num_in_points) + "_two_encoder_onehot_cls10"
        output_dir = os.path.join(args.output_dir, args.dataset_mode + "_cls", mode_dir)
    else:
        output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    return output_dir

def set_logger(log_dir):
    # Setup LOG file format
    global logger
    logger = logging.getLogger(args.model)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(os.path.join(log_dir, args.model + ".txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

def log_string(message):
    # Write message into log.txt
    logger.info(message)
    print(message)

def backup_python_file(backup_dir):
    os.system("cp ./train.py {}".format(backup_dir))
    os.system("cp ./dataset.py {}".format(backup_dir))
    os.system("cp ./models/pointnet_cls.py {}".format(backup_dir))

def create_dataloader():
    print("Load " + args.dataset_mode + " as dataset ...")

    # Create training dataloader
    TRAIN_DATASET = ModelNetDataLoader(num_in_points=args.num_in_points, num_out_points=args.num_out_points, split="train", dataset_mode=args.dataset_mode, sparsify_mode=args.sparsify_mode)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # Create testing dataloader
    TEST_DATASET = ModelNetDataLoader(num_in_points=args.num_in_points, num_out_points=args.num_out_points, split="test", dataset_mode=args.dataset_mode, sparsify_mode=args.sparsify_mode)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    return trainDataLoader, testDataLoader

def create_network():
    # Create network (classifier) and criterion
    classifier = CycleNet(num_class, num_out_points=args.num_out_points).cuda()
    criterion = get_loss().cuda()

    # Try load pretrained weights
    try:
        checkpoint = torch.load(os.path.join(checkpoints_dir, "best_model.pth"))
        start_epoch = checkpoint["epoch"]
        classifier.load_state_dict(checkpoint["model_state_dict"])
        log_string("Use pretrain model")
    except:
        log_string("No existing model, starting training from scratch...")
        start_epoch = 0

    # Setup optimizer
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    # Setup scheduler for optimizer
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    return classifier, criterion, optimizer, scheduler, start_epoch

def train(classifier, trainDataLoader, optimizer, scheduler, criterion):
    # TRAIN MODE
    loss = dict()
    mean_correct = {"partial": [], "recon": []}
    scheduler.step()
    for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
        # Get points and target from trainDataLoader
        partial, target, complete = data
        partial = partial.data.numpy()

        # Do something like augmentation
        partial = provider.random_point_dropout(partial)
        partial[:,:, 0:3] = provider.random_scale_point_cloud(partial[:,:, 0:3])
        partial[:,:, 0:3] = provider.shift_point_cloud(partial[:,:, 0:3])
        partial = torch.Tensor(partial)
        target = target[:, 0].long()

        partial = partial.transpose(2, 1)
        partial, target, complete = partial.cuda(), target.cuda(), complete.cuda()
        optimizer.zero_grad()

        classifier = classifier.train()
        for input_mode in ["partial", "recon"]:
            # Start training
            pred_cls, recon_output, expansion_penalty = classifier(partial, input_mode)

            # Get loss
            loss_cls, loss_EMD = criterion(pred_cls, target, recon_output, complete, 0.005, 50)
            loss["cls_" + input_mode] = loss_cls
            loss["EMD_" + input_mode] = loss_EMD
            loss["expansion_" + input_mode] = expansion_penalty
            train_save[input_mode]["cls"].append(loss_cls.item()) # Save classification loss
            train_save[input_mode]["EMD"].append(loss_EMD.mean().item()) # Save reconstruction loss
            train_save[input_mode]["expansion"].append(expansion_penalty.mean().item()) # Save reconstruction loss

            # Compute number of correct
            pred_choice = pred_cls.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            mean_correct[input_mode].append(correct.item() / float(partial.size()[0]))

            # Put reconstruction result to second round
            partial = recon_output.transpose(2, 1)

        partial_loss = loss["cls_partial"]*10 + loss["EMD_partial"].mean() + loss["expansion_partial"].mean() * 0.1
        recon_loss = loss["cls_recon"]*10 + loss["EMD_recon"].mean() + loss["expansion_recon"].mean() * 0.1
        total_loss = partial_loss + recon_loss
        total_loss.backward()
        optimizer.step()

    # Record training instance accuracy
    for input_mode in ["partial", "recon"]:
        train_save[input_mode]["instance_acc"].append(np.mean(mean_correct[input_mode]))


def test(model, criterion, testDataLoader):
    mean_correct = {"partial": [], "recon": []}
    class_acc = {"partial": np.zeros((num_class, 3)), "recon": np.zeros((num_class, 3))}
    for batch_id, data in tqdm(enumerate(testDataLoader), total=len(testDataLoader)):
        # Get points and target from testDataLoader
        partial, target, complete = data
        target = target[:, 0].long()
        partial = partial.transpose(2, 1)
        partial, target, complete = partial.cuda(), target.cuda(), complete.cuda()

        # Evaluate by PointNet model
        classifier = model.eval()
        for input_mode in ["partial", "recon"]:
            pred_cls, recon_output, _ = classifier(partial, input_mode)

            # Get loss
            loss_cls, loss_EMD = criterion(pred_cls, target, recon_output, complete, 0.004, 3000)
            test_save[input_mode]["cls"].append(loss_cls.item()) # Save classification loss
            test_save[input_mode]["EMD"].append(loss_EMD.mean().item()) # Save reconstruction loss

            pred_choice = pred_cls.data.max(1)[1] # prediction results

            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].data).cpu().sum()
                class_acc[input_mode][cat, 0] += classacc.item() / float(partial[target == cat].size()[0]) # Compute accuracy of certain class
                class_acc[input_mode][cat, 1] += 1 # Compute number of certain class
            correct = pred_choice.eq(target.data).cpu().sum() # Total number of correct results
            mean_correct[input_mode].append(correct.item() / float(partial.size()[0])) # Mean instance accuracy within one batch size
            
            # Put reconstruction result to second round
            partial = recon_output.transpose(2, 1)


    for input_mode in ["partial", "recon"]:    
        class_acc[input_mode][:, 2] =  class_acc[input_mode][:, 0] / class_acc[input_mode][:, 1] # The class accuracy of each class
        test_save[input_mode]["class_acc"].append(np.mean(class_acc[input_mode][:, 2])) # Mean class accuracy (all objects)
        test_save[input_mode]["instance_acc"].append(np.mean(mean_correct[input_mode])) # Mean instance accuracy (all objects)


if __name__ == "__main__":
    # Create output direcotry
    output_dir = create_output_dir()
    backup_dir = os.path.join(output_dir, "backup")
    checkpoints_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Backup important .py file
    backup_python_file(backup_dir)

    # Setup LOG file format
    set_logger(log_dir)
    log_string("Argument parameter: {}".format(args))

    # Create training and testing dataloader
    trainDataLoader, testDataLoader = create_dataloader()

    # Create network (classifier), optimizer, scheduler
    classifier, criterion, optimizer, scheduler, start_epoch = create_network()

    # Setup parameters for training and testing
    global_epoch = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    # Start training
    logger.info("Start training...")
    for epoch in range(start_epoch, args.epoch):
        log_string("Epoch %d (%d/%s):" % (global_epoch+1, epoch+1, args.epoch))

        # TRAIN MODE
        train(classifier, trainDataLoader, optimizer, scheduler, criterion)
        partial_EMD_loss = np.mean(train_save["partial"]["EMD"][-len(trainDataLoader):])
        recon_EMD_loss = np.mean(train_save["recon"]["EMD"][-len(trainDataLoader):])

        log_string("Train partial EMD loss: %f\trecon EMD loss: %f" % (partial_EMD_loss, recon_EMD_loss))
        log_string("Train partial instance acc: %f\trecon instance acc: %f" % (train_save["partial"]["instance_acc"][-1], train_save["recon"]["instance_acc"][-1]))

        # TEST MODE
        with torch.no_grad():
            test(classifier.eval(), criterion, testDataLoader)

            if test_save["recon"]["instance_acc"][-1] >= best_instance_acc:
                best_instance_acc = test_save["recon"]["instance_acc"][-1]
                best_epoch = epoch + 1

            if test_save["recon"]["class_acc"][-1] >= best_class_acc:
                best_class_acc = test_save["recon"]["class_acc"][-1]

            partial_EMD_loss = np.mean(test_save["partial"]["EMD"][len(testDataLoader):])
            recon_EMD_loss = np.mean(test_save["recon"]["EMD"][len(testDataLoader):])
            log_string("Test partial EMD loss: %f\trecon EMD loss: %f" % (partial_EMD_loss, recon_EMD_loss))
            log_string("Test partial instance acc: %f\trecon instance acc: %f" % (test_save["partial"]["instance_acc"][-1], test_save["recon"]["instance_acc"][-1]))
            log_string("Test partial class acc: %f\trecon class acc: %f" % (test_save["partial"]["class_acc"][-1], test_save["recon"]["class_acc"][-1]))
            log_string("Best Instance Accuracy: %f, Class Accuracy: %f" % (best_instance_acc, best_class_acc))

        # Save best training details
        if test_save["recon"]["instance_acc"][-1] >= best_instance_acc:
            logger.info("Save model...")
            save_path = os.path.join(checkpoints_dir, "best_model.pth")
            log_string("Saving at %s" % (save_path))
            state = {
                "epoch": best_epoch,
                "instance_acc": test_save["recon"]["instance_acc"][-1],
                "class_acc": test_save["recon"]["class_acc"][-1],
                "model_state_dict": classifier.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }
            torch.save(state, save_path)
        global_epoch += 1

        # Save weights and [training, testing] results
        # if epoch % 5 == 0:
        #     torch.save(state, os.path.join(checkpoints_dir, "model_%d.pth" %(epoch)))
        np.save(os.path.join(output_dir, "train_save.npy"), train_save)
        np.save(os.path.join(output_dir, "test_save.npy"), test_save)

    logger.info("End of training...")
