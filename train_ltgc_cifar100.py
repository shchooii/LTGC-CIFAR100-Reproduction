import os
import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Sampler, ConcatDataset
from torchvision import transforms, models
from torchvision import datasets as tv_datasets
from PIL import Image
import numpy as np
import random

from loss import FocalLoss, AsymmetricLoss


# =========================
# 1. Sampler / Dataset / CIFAR100-LT DataLoader
# =========================

class BalancedSampler(Sampler):
    """
    클래스별 index bucket을 받아서, 매 step마다 랜덤한 클래스에서 하나씩 뽑아오는 sampler.
    (LT only + balanced=True일 때만 사용. ConcatDataset + balanced는 비활성화)
    """
    def __init__(self, buckets, retain_epoch_size=False):
        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size
    
    def __iter__(self):
        count = self.__len__()
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets])
        else:
            return max([len(bucket) for bucket in self.buckets]) * self.bucket_num


class LT_Dataset(Dataset):
    """
    txt 포맷:
        img_relative_path label
    예:
        train/000001.png 0
        train/000002.png 5
    """
    def __init__(self, root, txt, transform=None, training=False):
        self.img_path = []
        self.labels = []
        self.transform = transform
        self.train = training

        with open(txt) as f:
            # ImageNet_LT_test 특수 케이스는 원 코드 유지
            if 'ImageNet_LT_test' in txt:
                for line in f:
                    tmp = line.split()[0]
                    tmp = tmp[:3] + tmp[13:]
                    pth = os.path.join(root, tmp)
                    self.img_path.append(pth)
                    self.labels.append(int(line.split()[1]))
            else:
                for line in f:
                    path, label = line.split()
                    self.img_path.append(os.path.join(root, path))
                    self.labels.append(int(label))

        self.targets = self.labels  # sampler 등에서 사용

    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(sample)
        else:
            img = transforms.ToTensor()(sample)

        return img, label


class CIFAR100LTDataLoader(DataLoader):
    """
    CIFAR100-LT DataLoader
    - txt 파일 기반 long-tailed CIFAR-100
    - (옵션) gen_dir 의 generated 이미지(ImageFolder)를 train에 concat
    - self.num_classes, self.cls_num_list 제공
    """

    def __init__(self,
                 data_dir: str,
                 batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 1,
                 training: bool = True,
                 balanced: bool = False,
                 retain_epoch_size: bool = True,
                 train_txt: str = "./data_txt/CIFAR100_LT/CIFAR100_LT_train.txt",
                 test_txt: str = "./data_txt/CIFAR100_LT/CIFAR100_LT_test.txt",
                 use_generated: bool = False,
                 gen_dir: Optional[str] = None):

        # --- Transform 정의 ---
        train_trsfm = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        test_trsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # --- Base Dataset (LT) ---
        if training:
            base_dataset = LT_Dataset(data_dir, train_txt, train_trsfm, training=True)
            dataset = base_dataset

            # --- Generated concat (옵션) ---
            if use_generated and gen_dir is not None and os.path.isdir(gen_dir):
                gen_dataset = tv_datasets.ImageFolder(root=gen_dir, transform=train_trsfm)
                dataset = ConcatDataset([base_dataset, gen_dataset])
                print(f"[INFO] Using generated data: "
                      f"base={len(base_dataset)}, gen={len(gen_dataset)}, total={len(dataset)}")
            else:
                if use_generated:
                    print(f"[WARN] use_generated=True 인데 gen_dir='{gen_dir}' 가 없거나 폴더가 아님. "
                          f"LT 데이터만 사용합니다.")

        else:
            dataset = LT_Dataset(data_dir, test_txt, test_trsfm, training=False)

        self.dataset = dataset
        self.n_samples = len(self.dataset)

        # --- 전체 targets 모아서 cls_num_list 계산 (LT or Concat 모두 지원) ---
        all_targets = []

        if isinstance(dataset, ConcatDataset):
            for ds in dataset.datasets:
                if hasattr(ds, "targets"):
                    all_targets.extend(ds.targets)
                elif hasattr(ds, "labels"):
                    all_targets.extend(ds.labels)
                else:
                    raise ValueError("Sub-dataset has no 'targets' or 'labels' attribute.")
        else:
            all_targets = list(dataset.targets)

        targets_np = np.array(all_targets)
        num_classes = len(np.unique(targets_np))
        assert num_classes == 100, f"Expected 100 classes, got {num_classes}"
        self.num_classes = num_classes

        cls_num_list = [0] * num_classes
        for y in targets_np:
            cls_num_list[y] += 1
        self.cls_num_list = cls_num_list

        # --- Balanced Sampler ---
        if balanced and training:
            buckets = [[] for _ in range(num_classes)]
            for idx, label in enumerate(all_targets):
                buckets[label].append(idx)
            sampler = BalancedSampler(buckets, retain_epoch_size)
            shuffle = False
        else:
            sampler = None

        self.shuffle = shuffle
        self.init_kwargs = {
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': True,
        }
        super().__init__(dataset=self.dataset, sampler=sampler, **self.init_kwargs)


# =========================
# 2. HMT grouping / Model / Train utils
# =========================

def compute_hmt_groups_cifar100():
    """
    CIFAR100-LT 전용 head/medium/tail 분할:
      head   : class 0 ~ 39
      medium : class 40 ~ 69
      tail   : class 70 ~ 99
    """
    head = set(range(0, 40))
    medium = set(range(40, 70))
    tail = set(range(70, 100))

    print(f"[HMT] head   : {min(head)} ~ {max(head)}  (#={len(head)})")
    print(f"[HMT] medium : {min(medium)} ~ {max(medium)}  (#={len(medium)})")
    print(f"[HMT] tail   : {min(tail)} ~ {max(tail)}  (#={len(tail)})")

    return {"head": head, "medium": medium, "tail": tail}


def build_model(num_classes: int = 100) -> nn.Module:
    """torchvision ResNet-18 기반 CIFAR-100 classifier (224x224 입력)."""
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def adjust_learning_rate(optimizer, epoch, base_lr, milestones, gamma=0.1):
    """간단한 멀티스텝 LR 스케줄러."""
    lr = base_lr
    for m in milestones:
        if epoch >= m:
            lr *= gamma
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
    epoch: int,
    print_freq: int = 100,
):
    model.train()
    running_loss = 0.0
    correct_top1 = 0
    total = 0

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, pred = outputs.max(1)
        correct_top1 += pred.eq(targets).sum().item()
        total += targets.size(0)

        if (i + 1) % print_freq == 0:
            avg_loss = running_loss / total
            acc1 = 100.0 * correct_top1 / total
            print(
                f"Epoch [{epoch}] Step [{i+1}/{len(loader)}] "
                f"Loss: {avg_loss:.4f} | Acc@1: {acc1:.2f}%"
            )

    epoch_loss = running_loss / total
    epoch_acc1 = 100.0 * correct_top1 / total
    return epoch_loss, epoch_acc1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
):
    model.eval()
    running_loss = 0.0
    correct_top1 = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        _, pred = outputs.max(1)
        correct_top1 += pred.eq(targets).sum().item()
        total += targets.size(0)

    loss = running_loss / total
    acc1 = 100.0 * correct_top1 / total
    return loss, acc1


@torch.no_grad()
def evaluate_groupwise(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    hmt_groups: dict,
):
    """
    head / medium / tail 그룹별:
      - micro accuracy
      - macro accuracy
      - macro F1
    """
    model.eval()

    all_preds = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        preds = outputs.argmax(dim=1)

        all_preds.append(preds.cpu())
        all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    num_classes = int(all_targets.max().item()) + 1

    tp = torch.zeros(num_classes, dtype=torch.long)
    fp = torch.zeros(num_classes, dtype=torch.long)
    fn = torch.zeros(num_classes, dtype=torch.long)
    support = torch.zeros(num_classes, dtype=torch.long)

    for c in range(num_classes):
        c_targets = (all_targets == c)
        c_preds   = (all_preds == c)

        tp[c] = (c_targets & c_preds).sum()
        fp[c] = (~c_targets & c_preds).sum()
        fn[c] = (c_targets & ~c_preds).sum()
        support[c] = c_targets.sum()

    acc_per_class = torch.zeros(num_classes, dtype=torch.float)
    f1_per_class = torch.zeros(num_classes, dtype=torch.float)

    for c in range(num_classes):
        if support[c] > 0:
            acc_per_class[c] = tp[c].float() / support[c].float()
        else:
            acc_per_class[c] = float("nan")

        denom_precision = tp[c] + fp[c]
        denom_recall = tp[c] + fn[c]

        if denom_precision > 0:
            precision_c = tp[c].float() / denom_precision.float()
        else:
            precision_c = 0.0

        if denom_recall > 0:
            recall_c = tp[c].float() / denom_recall.float()
        else:
            recall_c = 0.0

        if precision_c + recall_c > 0:
            f1_per_class[c] = 2 * precision_c * recall_c / (precision_c + recall_c)
        else:
            f1_per_class[c] = 0.0

    # Add total group for evaluation
    hmt_groups["total"] = set(range(num_classes))

    results = {}

    for name, group in hmt_groups.items():
        group = sorted(list(group))
        if len(group) == 0:
            continue

        # Macro F1
        group_f1s = f1_per_class[group]
        macro_f1 = group_f1s.mean().item()

        # Micro Metrics (Acc, F1)
        # For a group, we sum TP, FP, FN over classes in that group
        g_tp = tp[group].sum().float()
        g_fp = fp[group].sum().float()
        g_fn = fn[group].sum().float()

        # Micro Accuracy = Recall = TP / (TP + FN)
        # (This is equivalent to correct / total_samples_in_group)
        denom_acc = g_tp + g_fn
        if denom_acc > 0:
            micro_acc = (g_tp / denom_acc).item()
        else:
            micro_acc = 0.0

        # Micro Precision = TP / (TP + FP)
        denom_prec = g_tp + g_fp
        if denom_prec > 0:
            micro_prec = (g_tp / denom_prec).item()
        else:
            micro_prec = 0.0
        
        # Micro F1
        if micro_prec + micro_acc > 0:
            micro_f1 = 2 * micro_prec * micro_acc / (micro_prec + micro_acc)
        else:
            micro_f1 = 0.0

        results[name] = {
            "acc": micro_acc,
            "micro_f1": micro_f1,
            "macro_f1": macro_f1,
        }

    print("\n[Group-wise Evaluation]")
    # Print total first, then head/medium/tail
    for name in ["total", "head", "medium", "tail"]:
        if name not in results:
            continue
        r = results[name]
        print(
            f"  {name:6s} | "
            f"Acc: {r['acc']*100:6.2f}% | "
            f"Micro-F1: {r['micro_f1']*100:6.2f}% | "
            f"Macro-F1: {r['macro_f1']*100:6.2f}%"
        )

    return results


# =========================
# 3. Main
# =========================
# =========================
# 3. Main
# =========================

def main():
    parser = argparse.ArgumentParser(description="CIFAR100-LT Training")

    # --- 기본 경로들 ---
    parser.add_argument("--data_dir", type=str, default="data/CIFAR100_LT")
    parser.add_argument("--train_txt", type=str, default="./data_txt/CIFAR100_LT/CIFAR100_LT_train.txt")
    parser.add_argument("--test_txt", type=str, default="./data_txt/CIFAR100_LT/CIFAR100_LT_test.txt")

    # --- 학습 관련 하이퍼파라미터 ---
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    # milestones는 리스트로 받기 위해 nargs='+' 사용
    parser.add_argument("--milestones", type=int, nargs='+', default=[100, 150])
    parser.add_argument("--lr_gamma", type=float, default=0.1)

    # --- Loss ---
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal", "asl", "ldam", "bsce"])

    # --- Generated 데이터 ---
    parser.add_argument("--use_generated", action="store_true")
    parser.add_argument("--gen_dir", type=str, default=None, help="Path to generated data (e.g. data/CIFAR100_LT/gen_train)")

    # --- Balanced sampler ---
    parser.add_argument("--balanced", action="store_true")

    # --- 저장 파일 이름 ---
    parser.add_argument("--save_path", type=str, default="baseline_ce_cifar100lt_resnet18.pth")

    args = parser.parse_args()

    # gen_dir 기본값 처리 (argparse default로 처리하기 애매할 경우)
    if args.use_generated and args.gen_dir is None:
        args.gen_dir = os.path.join(args.data_dir, "gen_train")

    # ================= 실제 학습 로직은 그대로 =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] data_dir   = {args.data_dir}")
    print(f"[INFO] train_txt  = {args.train_txt}")
    print(f"[INFO] test_txt   = {args.test_txt}")
    print(f"[INFO] use_generated = {args.use_generated}")
    print(f"[INFO] gen_dir    = {args.gen_dir}")
    print(f"[INFO] loss       = {args.loss}")
    print(f"[INFO] save_path  = {args.save_path}")

    # DataLoaders (CIFAR100-LT)
    train_loader = CIFAR100LTDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        training=True,
        balanced=args.balanced,
        retain_epoch_size=True,
        train_txt=args.train_txt,
        test_txt=args.test_txt,
        use_generated=args.use_generated,
        gen_dir=args.gen_dir,
    )

    test_loader = CIFAR100LTDataLoader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        training=False,
        balanced=False,
        retain_epoch_size=True,
        train_txt=args.train_txt,
        test_txt=args.test_txt,
        use_generated=False,
        gen_dir=None,
    )

    print(f"[INFO] #train={len(train_loader.dataset)}, #test={len(test_loader.dataset)}")
    print(f"[INFO] #classes={train_loader.num_classes}")
    print(f"[INFO] cls_num_list (first 10): {train_loader.cls_num_list[:10]}")

    hmt_groups = compute_hmt_groups_cifar100()

    # Model
    model = build_model(num_classes=train_loader.num_classes).to(device)

    # Loss
    cls_counts = torch.tensor(train_loader.cls_num_list, dtype=torch.float32).to(device)

    if args.loss == "ce":
        criterion = nn.CrossEntropyLoss()
    elif args.loss == "focal":
        criterion = FocalLoss()
    elif args.loss == "asl":
        criterion = AsymmetricLoss()
    else:
        raise ValueError(f"Unknown loss: {args.loss}")

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        current_lr = adjust_learning_rate(
            optimizer, epoch, args.lr, args.milestones, gamma=args.lr_gamma
        )
        print(f"\n=== Epoch {epoch}/{args.epochs} | lr = {current_lr:.5f} ===")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        print(f"[Train] Loss: {train_loss:.4f} | Acc@1: {train_acc:.2f}%")

        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        print(f"[Test ] Loss: {val_loss:.4f} | Acc@1: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_acc": best_acc,
                    # argparse는 안 쓰지만, 형식 맞추려고 dict 형태로 저장
                    "args": {
                        "data_dir": args.data_dir,
                        "train_txt": args.train_txt,
                        "test_txt": args.test_txt,
                        "batch_size": args.batch_size,
                        "epochs": args.epochs,
                        "lr": args.lr,
                        "momentum": args.momentum,
                        "weight_decay": args.weight_decay,
                        "num_workers": args.num_workers,
                        "milestones": args.milestones,
                        "lr_gamma": args.lr_gamma,
                        "loss": args.loss,
                        "use_generated": args.use_generated,
                        "gen_dir": args.gen_dir,
                        "balanced": args.balanced,
                        "save_path": args.save_path,
                    },
                },
                args.save_path,
            )
            print(f"[INFO] New best Acc@1: {best_acc:.2f}% → checkpoint saved to {args.save_path}")

    print(f"\n[RESULT] Best Test Acc@1: {best_acc:.2f}%")

    # === Best checkpoint 로드해서 H/M/T 평가 ===
    print("\n[INFO] Loading best checkpoint for group-wise evaluation...")
    ckpt = torch.load(args.save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    _ = evaluate_groupwise(model, test_loader, device, hmt_groups)

    print(f"\n[RESULT] Best Test Acc@1: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
