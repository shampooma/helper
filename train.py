import os
import math
import torch
import numpy as np

from tqdm import tqdm

def train(
    run_name,
    run_id,
    n_classes,
    monitor_class,
    loader_train,
    loader_valid,
    model,
    loss_function,
    metric_function,
    ckpts_path,
    ob_size,
    eb_size,
    optimizer_class=torch.optim.Adam,
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    epoch_limit=500,
    repeat_per_epoch=1,
    init_lr=1e-3,
    lr_adjust_ratio=0.5,
    lr_patience=5,
    lr_minimum=0,
    early_stop_patience=20,
    drop_last_train=True,
    drop_last_valid=False,
):
    '''
        Run in notebook
        ob: operation batch
        eb: effective batch
        gt: ground truth

        eb_size // ob_size should be 0
    '''
    # Variables
    loaders = {
        "train": loader_train,
        "valid": loader_valid
    }

    drop_lasts = {
        "train": drop_last_train,
        "valid": drop_last_valid
    }

    # Define paths
    ckpt_path = f"{ckpts_path}/{run_name}{run_id}"
    misc_path = f"{ckpt_path}/misc.txt"
    model_train_path = f"{ckpt_path}/model_train.pt"
    model_valid_path = f"{ckpt_path}/model_valid.pt"
    history_path = f"{ckpt_path}/history.csv"

    # Setup ckpts
    if not os.path.isdir(ckpt_path):
        os.mkdir(ckpt_path)

    history_file = open(history_path, 'w')
    misc_file = open(misc_path, 'a')

    # Setup model
    model.to(device)

    # Optimizer
    optimizer=optimizer_class(model.parameters(),lr=init_lr)

    # write csv header
    csv_str = ""
    csv_str += "epoch"

    for phase in ["train", "valid"]:
        for i in range(n_classes):
            csv_str += f",{phase}_{i}_loss"

        for i in range(n_classes):
            csv_str += f",{phase}_{i}_metric"

    csv_str += ",lr\n"
    history_file.write(csv_str)
    history_file.flush()

    # Training
    current_lr=init_lr
    best_valid_loss = 1.
    best_train_loss = 1.
    valid_unimprove_count = 0
    train_unimprove_count = 0

    for epoch in range(epoch_limit):
        print(f"Epoch: {epoch}")

        # Early stop
        if valid_unimprove_count >= early_stop_patience:
            print('Early stop')

            break
    ##### ------- BREAK ------- #####

        # Learning rate adjust
        if train_unimprove_count != 0 and train_unimprove_count % lr_patience == 0:
            # Discard valid unimprove count during period of train not improve
            current_lr *= lr_adjust_ratio

            if current_lr < lr_minimum:
                print('Learning rate too small')
                break
    ##### ------- BREAK ------- #####

            valid_unimprove_count -= min(valid_unimprove_count, lr_patience)

            ckpt=torch.load(model_train_path)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            print(f"Learning rate adjust lr: {current_lr}")

        # Train valid
        # History variables
        stat = {
            "train": {
                "losses": [],
                "metrics": []
            },
            "valid": {
                "losses": [],
                "metrics": []
            },
        }

        for phase in ["train", "valid"]:
            print(f"{phase}")

            # Set model to train or eval
            model.train() if phase == "train" else model.eval()

            # Variables
            losses = []
            metrics = []

            ob_amount = None
            data_amount = None

            if drop_lasts[phase]:
                ob_amount = int((len(loaders[phase].dataset) // eb_size) * eb_size / ob_size)
                data_amount = ob_amount * ob_size
            else:
                data_amount = len(loaders[phase].dataset)
                ob_amount = math.ceil(data_amount / ob_size)

            repeat_iter = tqdm(range(repeat_per_epoch), position=0, desc="Repeat per epoch")
            for repeat_index in repeat_iter:
                current_eb_size = None
                batch_loss = None
                batch_metric = None
                current_eb_data_ran_amount = 0
                remain_data_amount  = data_amount

                loader_iter = tqdm(loaders[phase], position=1, desc="Iter loader", leave=False, total=ob_amount)
                for imgs, gts in loader_iter:
                    # Check if still have data
                    if remain_data_amount <= 0:
                        loader_iter.close()
                        break
                ##### ------- BREAK ------- #####

                    # Set optimizer zero grad and init history if it is start of accumulate gradient
                    if current_eb_data_ran_amount == 0:
                        current_eb_size = min(remain_data_amount, eb_size)
                        batch_loss = [0 for _ in range(n_classes)]
                        batch_metric = [0 for _ in range(n_classes)]
                        optimizer.zero_grad()

                    # Set values for current operation batch
                    imgs, gts = imgs.to(device), gts.to(device)

                    # Predict
                    preds=model(imgs)

                    # Calculate && record loss for one bacth
                    class_loss = loss_function(gts, preds)

                    loss = class_loss.sum() / n_classes

                    if phase == "train":
                        loss *=  imgs.shape[0] / current_eb_size
                        loss.backward()

                    for i in range(n_classes):
                        batch_loss[i] += class_loss.detach().cpu().numpy()[i] * (imgs.shape[0] / current_eb_size)

                    # Calculate && record metric for one bacth
                    class_metric = metric_function(gts, preds)

                    for i in range(n_classes):
                        batch_metric[i] += class_metric.detach().cpu().numpy()[i] * (imgs.shape[0] / current_eb_size)

                    # Change variables after one operation batch
                    remain_data_amount -= imgs.shape[0]
                    current_eb_data_ran_amount += imgs.shape[0]

                    # Check if is last batch of effective batch
                    if current_eb_data_ran_amount == current_eb_size:
                        # Step
                        if phase == "train":
                            optimizer.step()

                        # Record batch loss and metric
                        losses.append(batch_loss)
                        metrics.append(batch_metric)

                        current_eb_data_ran_amount = 0

                    # Delete values
                    del imgs, gts, preds, class_loss, loss, class_metric

            # Calculate loss and metric for all batch
            loss = np.mean(losses, axis=0)
            metric = np.mean(metrics, axis=0)

            print(f"mean loss: {np.mean(loss)} | losses: {loss}")
            print(f"mean metric: {np.mean(metric)} | metrics: {metric}")

            # Record loss and metric for all batch
            stat[phase]['losses'] = loss
            stat[phase]['metrics'] = metric

        # Write history
        history_file.write(f"{epoch}")
        for _, phase in stat.items():
            for loss in phase['losses']:
                history_file.write(f",{loss}")

            for metric in phase['metrics']:
                history_file.write(f",{metric}")

        history_file.write(f",{current_lr}")
        history_file.write(f"\n")
        history_file.flush()

        # Save model and calculate stop improve
        val_loss = stat['valid']['losses'][monitor_class]
        train_loss = stat['train']['losses'][monitor_class]

        if val_loss < best_valid_loss:
            print(f'save valid model | old_loss: {best_valid_loss} | new_loss: {val_loss}')
            torch.save(
                {
                    'model': model.state_dict()
                },
                model_valid_path
            )

            best_valid_loss = val_loss
            valid_unimprove_count = 0
        else:
            valid_unimprove_count += 1

        if train_loss < best_train_loss:
            print(f'save train model | old_loss: {best_train_loss} | new_loss: {train_loss}')
            torch.save(
                {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                model_train_path
            )

            best_train_loss = train_loss
            train_unimprove_count = 0
        else:
            train_unimprove_count += 1

        print()

    # Close files
    history_file.close()
    misc_file.close()

if __name__ == '__main__':
    pass
