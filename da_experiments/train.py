
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch
import numpy as np
import os
import argparse
from typing import Dict
from tqdm import tqdm, trange
import random
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
import shutil
import json
from utils import AverageMeter, save_checkpoint, get_model_dir, \
    load_cfg_from_cfg_file, merge_cfg_from_list, find_free_port, \
    setup, cleanup, main_process, to_np_image
import torch.multiprocessing as tmp
import torch.backends.cudnn as cudnn
from datasets import __dict__ as data_dict
from architectures import __dict__ as model_dict
from estimators import __dict__ as methods_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Eval')
    parser.add_argument('--base_config', type=str, required=True, help='config file')
    parser.add_argument('--method_config', type=str, default=True, help='Base config file')
    parser.add_argument('--opts', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.base_config is not None
    cfg = load_cfg_from_cfg_file(args.base_config)
    cfg.update(load_cfg_from_cfg_file(args.method_config))
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def copy_config(args, exp_root):
    # ========== Copy source code ==========
    p = Path(".")
    python_files = list(p.glob('**/*.py'))
    filtered_list = [file for file in python_files if ('checkpoints' not in str(file) and 'archive' not in str(file) and 'results' not in str(file))]
    for file in filtered_list:
        file_dest = exp_root / 'src_code' / file
        file_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file, file_dest)

    # ========== Copy yaml files ==========
    with open(exp_root / 'config.json', 'w') as fp:
        json.dump(args, fp, indent=4)


def hash_config(args: argparse.Namespace):
    res = 0
    for i, (key, value) in enumerate(args.items()):
        if key != "port":
            if type(value) == str:
                hash_ = sum(value.encode())
            elif type(value) in [float, int, bool]:
                hash_ = round(value, 3)
            else:
                hash_ = sum([int(v) if type(v) in [float, int, bool] else sum(v.encode()) for v in value])
            res += hash_ * random.randint(1, 1000000)
    return str(res)[-10:].split('.')[0]


def get_loaders(args, data):
    size = {'MNIST': 28, 'MNISTM': 28, 'CIFAR10': 32, 'USPS': 28, 'SVHN': 28, 'STL10': 32}
    train_transforms = transforms.Compose([transforms.Resize(size[data]),
                                           transforms.CenterCrop(size[data]),
                                           transforms.ToTensor(),
                                           ])

    test_transforms = transforms.Compose([transforms.Resize(size[data]),
                                          transforms.CenterCrop(size[data]),
                                          transforms.ToTensor(),
                                          ])

    train_dataset = data_dict[data](args.data_path, transform=train_transforms, split='train', download=True)
    val_dataset = data_dict[data](args.data_path, transform=test_transforms, split='valid', download=True)
    test_dataset = data_dict[data](args.data_path, transform=test_transforms, split='test', download=True)

    n_classes = np.unique(test_dataset.targets).shape[0]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, drop_last=True)
    return train_loader, val_loader, test_loader, n_classes


def main_worker(rank: int,
                world_size: int,
                args: argparse.Namespace) -> None:
    print(f"==> Running process rank {rank}.")
    setup(args.port, rank, world_size)

    # ============ Setting up the experiment ================
    if args.seed is not None:
        args.seed += rank
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    if main_process(args):
        exp_root = Path(get_model_dir(args))
        exp_root.mkdir(exist_ok=True, parents=True)
        exp_no = hash_config(args)
        exp_root = exp_root / str(exp_no)
        copy_config(args, exp_root)
        visu_dir = os.path.join(args.visu_dir, args.source_data, args.target_data)
        os.makedirs(visu_dir, exist_ok=True)
        print(f"==>  Saving all at {exp_root}")

    # ============ Define loaders ================
    train_loader_s, val_loader_s, test_loader_s, n_classes = get_loaders(args, args.source_data)
    train_loader_t, val_loader_t, test_loader_t, _ = get_loaders(args, args.target_data)

    # ============ Define model ================
    feature_ex_c = model_dict[args.feature_extractor](args).to(rank)
    label_predictor = model_dict['LabelPredictor'](args, feature_ex_c.feature_dim, n_classes).to(rank)
    if not args.source_only:
        feature_ex_d = model_dict[args.feature_extractor](args).to(rank)
        domain_predictor = model_dict['DomainPredictor'](args, feature_ex_c.feature_dim, 2).to(rank)

    # ============ Define metrics ================
    mutual_infos = AverageMeter()
    losses_d = AverageMeter()
    losses_c = AverageMeter()
    mis_loss = AverageMeter()
    precs_c = AverageMeter()
    precs_d = AverageMeter()
    margs_ent = AverageMeter()
    conds_ent = AverageMeter()
    precs_t = AverageMeter()
    simu_params = {param: args[param] for param in args.simu_params}
    if main_process(args):
        metrics: Dict[str, torch.Tensor] = {"train_loss_c": torch.zeros(int(args.num_updates / args.print_freq)),
                                            "train_loss_d": torch.zeros(int(args.num_updates / args.print_freq)),
                                            "train_mi": torch.zeros(int(args.num_updates / args.print_freq)),
                                            "train_cond_ent": torch.zeros(int(args.num_updates / args.print_freq)),
                                            "train_marg_ent": torch.zeros(int(args.num_updates / args.print_freq)),
                                            "train_acc": torch.zeros(int(args.num_updates / args.print_freq)),
                                            "val_acc": torch.zeros(int(args.num_updates / args.eval_freq)),
                                            "test_acc": torch.zeros(int(args.num_updates / args.eval_freq)),
                                            "best_val_acc": torch.zeros(int(args.num_updates / args.eval_freq)),
                                            "best_test_acc": torch.zeros(int(args.num_updates / args.eval_freq)),
                                            }

    # ============ Method ================
    if not args.source_only:
        mi_estimator = methods_dict[args.method](args, feature_ex_c.feature_dim, feature_ex_d.feature_dim).to(rank)

    # ============ Optimizers ================
    context_vars = list(feature_ex_c.parameters()) + list(label_predictor.parameters())
    optimizer_c = torch.optim.Adam(context_vars, lr=args.context_lr, betas=(0.5, 0.999))
    # optimizer = eval(f'torch.optim.{args.optimizer}')
    if not args.source_only:
        domain_vars = list(feature_ex_d.parameters()) + list(domain_predictor.parameters())
        optimizer_d = torch.optim.Adam(domain_vars, lr=args.domain_lr, betas=(0.5, 0.999))
        optimizer_mi = torch.optim.Adam(mi_estimator.parameters(), lr=args.mi_lr, betas=(0.5, 0.999))

    # ============ Start training ================
    best_val_acc = 0.
    best_test_acc = 0.
    loss_fn = nn.CrossEntropyLoss()
    tqdm_bar = trange(args.num_updates)

    def cycle(iterable):
        while True:
            for x in iterable:
                yield x
    iter_train_s = cycle(train_loader_s)
    iter_train_t = cycle(train_loader_t)
    for i in tqdm_bar:

        if i >= args.num_updates:
            break
        x_s, y_s = next(iter_train_s)  # noqa: F821
        x_t, y_t = next(iter_train_t)  # noqa: F821

        # ======== Visualization of samples =========

        if args.visu and i == 0:
            n_samples = 2
            fig = plt.figure(figsize=(4 * n_classes, 4 * n_samples), dpi=100)
            grid = ImageGrid(fig, 111,
                             nrows_ncols=(2 * n_samples, n_classes),
                             axes_pad=(0.4, 0.4),
                             direction='row',
                             )
            for j in range(n_classes):
                source_x = x_s[y_s == j][:n_samples]
                target_x = x_t[y_t == j][:n_samples]
                cat_samples = torch.cat([source_x, target_x], 0)
                for row in range(cat_samples.shape[0]):
                    ax = grid[n_classes * row + j]
                    ax.imshow(to_np_image(cat_samples[row]))
                    ax.axis('off')
            fig.savefig(os.path.join(visu_dir, 'test.png'))

        x_s, y_s = x_s.to(rank), y_s.to(rank)
        x_t, y_t = x_t.to(rank), y_t.to(rank)
        y_d = torch.cat([torch.zeros_like(y_s),
                         torch.ones_like(y_t)])
        current_bs = x_s.size(0)

        # ======== Forward / Backward pass =========

        cat_x = torch.cat([x_s, x_t])
        if args.source_only:
            feat_c = feature_ex_c(cat_x)  # important to feed the concat vector for features stats
            logits_cs = label_predictor(feat_c[:current_bs])
            loss_c = loss_fn(logits_cs, y_s)

            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()
        else:
            feat_c = feature_ex_c(cat_x)
            feat_d = feature_ex_d(cat_x)

            # ======== Estimator update ======
            mi_loss = mi_estimator.learning_loss(feat_c.detach(), feat_d.detach())

            optimizer_mi.zero_grad()
            mi_loss.backward()
            optimizer_mi.step()

            # ====== Content update ======
            logits_cs = label_predictor(feat_c[:current_bs])
            mi, marg_ent, cond_ent = mi_estimator(feat_c, feat_d.detach())
            loss_c = loss_fn(logits_cs, y_s) + args.lambda_c * mi

            optimizer_c.zero_grad()
            loss_c.backward()
            optimizer_c.step()

            # ====== Domain update ======
            logits_d = domain_predictor(feat_d)
            loss_d = loss_fn(logits_d, y_d)

            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

        # ============ Log metrics ============
        logits_ct = label_predictor(feat_c[-current_bs:])
        losses_c.update(loss_c.item(), i == 0)
        precs_c.update((y_s == logits_cs.argmax(-1)).float().mean().item(), i == 0)
        precs_t.update((y_t == logits_ct.argmax(-1)).float().mean().item(), i == 0)
        if not args.source_only:
            losses_d.update(loss_d.item(), i == 0)
            mutual_infos.update(mi.item(), i == 0)
            margs_ent.update(marg_ent.item() if marg_ent else 0, i == 0)
            conds_ent.update(cond_ent.item() if cond_ent else 0, i == 0)
            mis_loss.update(mi_loss.item(), i == 0)
            precs_d.update((y_d == logits_d.argmax(-1)).float().mean().item(), i == 0)

        # ============ Validation ============
        if i % args.eval_freq == 0:
            val_acc = val_fn(val_loader_t, rank, feature_ex_c, label_predictor)
            test_acc = val_fn(test_loader_t, rank, feature_ex_c, label_predictor)
            is_best = (val_acc > best_val_acc)
            best_val_acc = max(val_acc, best_val_acc)
            best_test_acc = test_acc if is_best else best_test_acc
            print('Iteration: [{}/{}] \t Validation_t: {:.3f} Test_t: {:.3f}'.format(
                                                i, args.num_updates, val_acc, test_acc))  # noqa: E126
            if main_process(args):
                save_checkpoint(state={'iter': i,
                                       'feature_ex_c': feature_ex_c.state_dict(),
                                       'feature_ex_d': feature_ex_d.state_dict() if not args.source_only else None,
                                       'label_predictor': label_predictor.state_dict(),
                                       'domain_predictor': domain_predictor.state_dict() if not args.source_only else None,
                                       'best_val_acc': best_val_acc,
                                       'best_test_acc': best_test_acc,
                                       },
                                is_best=is_best,
                                folder=exp_root)
                for k in metrics:
                    if 'val' in k or 'test' in k:
                        metrics[k][int(i / args.eval_freq)] = eval(k)
                for k, e in metrics.items():
                    path = os.path.join(exp_root, f"{k}.npy")
                    np.save(path, e.cpu().numpy())
            path = os.path.join(exp_root, f"train.csv")
            update_csv(simu_params, metrics, i, path)

        # ============ Print / log metrics ============
        if i % args.print_freq == 0 and main_process(args):
            train_loss_c = losses_c.avg
            train_mi = mutual_infos.avg
            train_cond_ent = conds_ent.avg
            train_marg_ent = margs_ent.avg
            train_loss_d = losses_d.avg  # noqa: F841
            train_acc = precs_c.avg
            tqdm_bar.set_description(
                'Iteration: [{0}/{1}] '
                'Loss_c  ({loss.avg:.4f}) '
                'Loss_MI ({loss_mi.avg:.4f}) '
                'MI ({mi.avg:.4f}) '
                'Acc_s ({acc_s.avg:.3f}) '
                'Acc_t  ({acc_t.avg:.3f}) '.format(
                   i, args.num_updates, loss_mi=mis_loss, # noqa: E121
                   mi=mutual_infos, loss=losses_c, acc_s=precs_c, acc_t=precs_t))

            for k in metrics:
                if not ('val' in k or 'test' in k):
                    metrics[k][int(i / args.print_freq)] = eval(k)

    cleanup()


def update_csv(simu_params: dict,
               metrics: dict,
               iteration: int,
               path: str):
    try:
        res = pd.read_csv(path)
    except:
        res = pd.DataFrame({})
    records = res.to_dict('records')
    # Check whether the entry exist already, if yes, simply update the accuracy
    match = False
    for entry in records:
        try:
            match_list = [value == simu_params[param] for param, value in list(entry.items()) if param in simu_params]
            match = (sum(match_list) == len(match_list))
            if match:
                for metric_name, array in metrics.items():
                    past_values = array[array != 0]
                    last_value = -1 if not past_values.numel() else past_values[-1].item()
                    entry[metric_name] = round(last_value, 4)
                entry['iteration'] = iteration
                break
        except:
            continue

    # If entry did not exist, just create it
    if not match:
        new_entry = simu_params.copy()
        for metric_name, array in metrics.items():
            past_values = array[array != 0]
            last_value = -1 if not past_values.numel() else past_values[-1].item()
            new_entry[metric_name] = round(last_value, 4)
        new_entry['iteration'] = iteration
        records.append(new_entry)
    df = pd.DataFrame.from_records(records)
    df.to_csv(path, index=False)


def val_fn(loader, device, feature_extractor, classifier):
    feature_extractor.eval()
    classifier.eval()
    preds = []
    labels = []
    tbar = tqdm(loader)
    with torch.no_grad():
        for x, y in tbar:
            x, y = x.to(device), y.to(device)
            z = feature_extractor(x)
            logits = classifier(z)
            preds.append(logits.argmax(-1))
            labels.append(y)

    preds = torch.cat(preds)
    labels = torch.cat(labels)
    acc = (preds == labels).float().mean()
    feature_extractor.train()
    classifier.train()
    return acc


if __name__ == "__main__":
    args = parse_args()
    world_size = len(args.gpus)
    distributed = world_size > 1
    args.world_size = world_size
    args.distributed = distributed
    args.port = find_free_port()
    if args.debug:
        args.num_updates = 10
        args.print_freq = 1
        args.eval_freq = 5
    tmp.spawn(main_worker,
              args=(world_size, args),
              nprocs=world_size,
              join=True)