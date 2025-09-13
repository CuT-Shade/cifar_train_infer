import io
import json
import time
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18
from PIL import Image
import pandas as pd
import contextlib

DATASETS_INFO = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "num_classes": 10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
    }
}

def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic

def get_device(prefer_cpu: bool = False):
    if prefer_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")

def build_cifar_resnet18(num_classes: int):
    m = resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m

def load_state_safely(model: nn.Module, state_dict: Dict[str, Any]):
    clean = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean[k[7:]] = v
        else:
            clean[k] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    return missing, unexpected

def load_weights_from_path(model: nn.Module, path: Path, device):
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError("无法解析权重文件。")
    return load_state_safely(model, state_dict)

def load_weights_from_bytes(model: nn.Module, b: bytes, device):
    buf = io.BytesIO(b)
    obj = torch.load(buf, map_location=device)
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict):
        state_dict = obj
    else:
        raise ValueError("无法解析权重文件。")
    return load_state_safely(model, state_dict)

def build_transforms(mean, std, use_randaugment: bool, is_train: bool):
    tfs = []
    if is_train:
        tfs += [
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
        ]
        if use_randaugment:
            tfs.append(T.RandAugment(num_ops=2, magnitude=9))
    tfs += [T.ToTensor(), T.Normalize(mean, std)]
    return T.Compose(tfs)

def evaluate(model: nn.Module, loader, device, amp_dtype=None):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    preds_all = []
    targets_all = []

    use_amp = (amp_dtype is not None and device.type == "cuda")
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if use_amp else contextlib.nullcontext()

    with torch.inference_mode(), amp_ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x)
            loss = criterion(out, y)
            loss_sum += loss.item() * x.size(0)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            preds_all.append(pred.cpu())
            targets_all.append(y.cpu())

    preds_all = torch.cat(preds_all) if preds_all else torch.empty(0, dtype=torch.long)
    targets_all = torch.cat(targets_all) if targets_all else torch.empty(0, dtype=torch.long)
    return {
        "loss": loss_sum / max(1, total),
        "acc": 100.0 * correct / max(1, total),
        "preds": preds_all,
        "targets": targets_all
    }

def predict_single(model: nn.Module, img_tensor: torch.Tensor, device, topk=5):
    model.eval()
    with torch.inference_mode():
        logits = model(img_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0]
        k = min(topk, probs.shape[0])
        topv, topi = probs.topk(k)
        return [(topi[i].item(), topv[i].item()) for i in range(k)]

def denorm_to_pil(t: torch.Tensor, mean, std):
    mean_t = torch.tensor(mean)[:, None, None]
    std_t = torch.tensor(std)[:, None, None]
    x = (t.cpu() * std_t + mean_t).clamp(0,1)
    arr = (x.permute(1,2,0).numpy()*255).astype("uint8")
    return Image.fromarray(arr)

def compute_confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    k = num_classes
    with torch.inference_mode():
        indices = targets * k + preds
        cm = torch.bincount(indices, minlength=k*k).reshape(k, k)
    return cm

def metrics_from_confusion(cm: torch.Tensor) -> Dict[str, float]:
    with torch.inference_mode():
        tp = cm.diag()
        support = cm.sum(dim=1)
        pred_sum = cm.sum(dim=0)
        recall = torch.where(support > 0, tp / support, torch.zeros_like(tp))
        precision = torch.where(pred_sum > 0, tp / pred_sum, torch.zeros_like(tp))
        f1 = torch.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), torch.zeros_like(tp))
        macro_recall = recall.mean().item() * 100
        macro_precision = precision.mean().item() * 100
        macro_f1 = f1.mean().item() * 100
        overall_acc = tp.sum().item() / cm.sum().item() * 100 if cm.sum() > 0 else 0.0
    return {
        "overall_acc": overall_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }

def top_misclass_pairs(cm: torch.Tensor, classes: List[str], top_n: int = 15) -> List[Tuple[str, str, int]]:
    k = cm.size(0)
    pairs = []
    for t in range(k):
        for p in range(k):
            if t == p:
                continue
            c = int(cm[t, p].item())
            if c > 0:
                pairs.append((classes[t] if t < len(classes) else str(t),
                              classes[p] if p < len(classes) else str(p), c))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]

def train_loop(
    dataset_name: str,
    data_dir: str,
    out_root: str,
    epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    use_randaugment: bool,
    use_amp: bool,
    amp_mode: str,
    label_smoothing: float,
    workers: int,
    seed: int,
    channels_last: bool,
    progress_cb=None,
):
    set_seed(seed)
    info = DATASETS_INFO[dataset_name]
    mean, std = info["mean"], info["std"]
    num_classes = info["num_classes"]

    device = get_device()
    amp_dtype = None
    if use_amp and device.type == "cuda":
        if amp_mode == "bf16":
            amp_dtype = torch.bfloat16
        elif amp_mode == "fp16":
            amp_dtype = torch.float16
        elif amp_mode == "auto":
            amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if dataset_name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                                 transform=build_transforms(mean, std, use_randaugment, True))
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                                transform=build_transforms(mean, std, False, False))
    else:
        train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True,
                                                  transform=build_transforms(mean, std, use_randaugment, True))
        test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True,
                                                 transform=build_transforms(mean, std, False, False))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=True, persistent_workers=workers>0
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=512, shuffle=False, num_workers=workers,
        pin_memory=True, persistent_workers=workers>0
    )

    model = build_cifar_resnet18(num_classes=num_classes).to(device)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing if label_smoothing > 0 else 0.0)
    scaler = torch.cuda.amp.GradScaler() if (amp_dtype == torch.float16) else None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(out_root) / dataset_name / f"run-{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "dataset": dataset_name,
        "mean": mean,
        "std": std,
        "num_classes": num_classes,
        "classes": train_set.classes,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "randaugment": use_randaugment,
        "amp_dtype": (str(amp_dtype) if amp_dtype else "None"),
        "label_smoothing": label_smoothing,
        "seed": seed,
        "channels_last": channels_last,
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_acc = -1.0
    best_path = out_dir / "best.pth"

    for epoch in range(epochs):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        t0 = time.time()

        use_amp_ctx = (amp_dtype is not None and device.type == "cuda")
        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if use_amp_ctx else contextlib.nullcontext()

        for images, targets in train_loader:
            if channels_last:
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                outputs = model(images)
                loss = criterion(outputs, targets)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            with torch.inference_mode():
                preds = outputs.argmax(1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                loss_sum += loss.item() * targets.size(0)

        scheduler.step()
        train_acc = 100.0 * correct / max(1, total)
        train_loss = loss_sum / max(1, total)

        val_metrics = evaluate(model, test_loader, device, amp_dtype=amp_dtype)
        val_acc = val_metrics["acc"]
        val_loss = val_metrics["loss"]
        cur_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(cur_lr)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)

        with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": cur_lr,
                "time_sec": time.time() - t0
            }) + "\n")

        if progress_cb:
            progress_cb(epoch + 1, epochs, train_loss, train_acc, val_loss, val_acc, cur_lr, best_acc, str(out_dir))

    return {
        "out_dir": str(out_dir),
        "history": history,
        "best_acc": best_acc,
        "best_path": str(best_path),
        "meta_path": str(out_dir / "meta.json"),
        "classes": train_set.classes,
        "mean": mean,
        "std": std,
        "dataset": dataset_name
    }

st.set_page_config(page_title="CIFAR Train & Inference", page_icon="🧠", layout="wide")
st.title("CIFAR-10 / CIFAR-100 训练 + 推理")

tabs = st.tabs(["训练", "推理"])

default_state = {
    "last_train_result": None,
    "loaded_model": None,
    "loaded_meta": None,
    "loaded_classes": None,
    "loaded_mean": None,
    "loaded_std": None,
    "loaded_dataset": None,
    "model_ready": False,
    "confusion_cache": None
}
for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

with tabs[0]:
    st.subheader("模型训练")
    dataset_choice = st.selectbox("数据集", ["cifar10", "cifar100"])
    data_dir = st.text_input("数据根目录", "./data")
    out_root = st.text_input("输出根目录", "./outputs")
    epochs = st.number_input("训练轮数", 1, 400, 50, 10)
    batch_size = st.number_input("Batch Size", 32, 1024, 256, 32)
    auto_lr = st.checkbox("自动学习率 (0.1 * batch/128)", value=True)
    if auto_lr:
        lr = 0.1 * (batch_size / 128.0)
    else:
        lr = st.number_input("学习率", 1e-4, 1.0, 0.2, 0.05, format="%.4f")
    weight_decay = st.select_slider("Weight Decay", options=[0.0, 5e-4, 1e-3, 5e-3], value=5e-4)
    momentum = st.select_slider("Momentum", options=[0.0, 0.5, 0.9], value=0.9)
    label_smoothing = st.select_slider("Label Smoothing", options=[0.0, 0.05, 0.1], value=0.1)
    use_randaugment = st.checkbox("RandAugment", value=True)
    amp_flag = st.checkbox("启用 AMP", value=True)
    amp_mode = st.selectbox("AMP 模式", ["auto", "bf16", "fp16"], index=0)
    channels_last = st.checkbox("channels_last", value=True)
    workers = st.number_input("num_workers", 0, 16, 4, 1)
    seed = st.number_input("随机种子", 0, 9999, 42, 1)

    train_btn = st.button("开始训练", type="primary")
    progress_bar = st.progress(0)
    train_status = st.empty()
    chart_acc = st.empty()
    chart_loss = st.empty()
    best_box = st.empty()
    hist_cache = {"epoch": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}

    def on_progress(ep_now, ep_total, tr_loss, tr_acc, val_loss, val_acc, cur_lr, best_acc, out_dir):
        pct = int(ep_now / ep_total * 100)
        progress_bar.progress(pct, text=f"Epoch {ep_now}/{ep_total}")
        train_status.info(
            f"Epoch {ep_now}/{ep_total} | "
            f"Train {tr_acc:.2f}%/{tr_loss:.4f} | Val {val_acc:.2f}%/{val_loss:.4f} | "
            f"LR {cur_lr:.5f} | Best {best_acc:.2f}%"
        )
        hist_cache["epoch"].append(ep_now)
        hist_cache["train_acc"].append(tr_acc)
        hist_cache["val_acc"].append(val_acc)
        hist_cache["train_loss"].append(tr_loss)
        hist_cache["val_loss"].append(val_loss)
        chart_acc.line_chart({"Train": hist_cache["train_acc"], "Val": hist_cache["val_acc"]}, height=220)
        chart_loss.line_chart({"Train": hist_cache["train_loss"], "Val": hist_cache["val_loss"]}, height=220)
        best_box.success(f"当前最佳验证准确率：{best_acc:.2f}%\n输出目录：{out_dir}")

    if train_btn:
        try:
            st.session_state.last_train_result = train_loop(
                dataset_name=dataset_choice,
                data_dir=data_dir,
                out_root=out_root,
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
                use_randaugment=bool(use_randaugment),
                use_amp=bool(amp_flag),
                amp_mode=amp_mode,
                label_smoothing=float(label_smoothing),
                workers=int(workers),
                seed=int(seed),
                channels_last=bool(channels_last),
                progress_cb=on_progress
            )
            progress_bar.progress(100, text="训练完成 ✓")
            train_status.success(f"训练完成，最佳准确率：{st.session_state.last_train_result['best_acc']:.2f}%")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                train_status.error("显存不足：减小 batch_size 或启用 AMP / 关闭 RandAugment。")
            else:
                train_status.error(f"训练失败：{e}")
        except Exception as e:
            train_status.error(f"训练异常：{e}")

    if st.session_state.last_train_result:
        res = st.session_state.last_train_result
        dl1, dl2 = st.columns(2)
        with dl1:
            if Path(res["best_path"]).exists():
                with open(res["best_path"], "rb") as f:
                    st.download_button("下载 best.pth", f, file_name="best.pth")
        with dl2:
            if Path(res["meta_path"]).exists():
                with open(res["meta_path"], "rb") as f:
                    st.download_button("下载 meta.json", f, file_name="meta.json")
        st.caption(f"输出目录：{res['out_dir']}")

with tabs[1]:
    st.subheader("推理 / 混淆矩阵")
    infer_left, infer_right = st.columns([1,1])

    with infer_left:
        auto_from_train = st.checkbox("使用最近训练的权重", value=True)
        weight_path_input = st.text_input("权重路径（优先）", "")
        meta_path_input = st.text_input("meta.json 路径（可选）", "")
        weight_dir = st.text_input("递归扫描权重目录", "./outputs")
        refresh_weight_dir = st.button("刷新权重列表")
        if "infer_pth_list" not in st.session_state or refresh_weight_dir:
            def list_all_pth(root):
                root_p = Path(root)
                if not root_p.exists():
                    return []
                return [str(p) for p in root_p.rglob("*.pth")]
            st.session_state.infer_pth_list = list_all_pth(weight_dir)
        weight_select = st.selectbox("从目录选择权重", ["(未选择)"] + st.session_state.infer_pth_list, index=0)
        uploaded_w = st.file_uploader("上传权重（可选）", type=["pth"])
        uploaded_meta = st.file_uploader("上传 meta.json（可选）", type=["json"])
        device_pref = st.selectbox("设备", ["auto", "cpu"], index=0)
        topk = st.slider("Top-K", 1, 10, 5)

    with infer_right:
        infer_dataset_choice = st.selectbox("测试集数据集", ["cifar10", "cifar100"])
        data_dir_infer = st.text_input("测试集根目录", "./data")
        prepare_test_btn = st.button("准备 / 下载测试集")
        eval_batch_size = st.number_input("评估 batch_size", 32, 1024, 512, 32)

    meta_json_obj = None
    if uploaded_meta:
        try:
            meta_json_obj = json.loads(uploaded_meta.read().decode("utf-8"))
        except Exception as e:
            st.error(f"上传 meta.json 解析失败：{e}")

    weight_mode = "none"
    weight_bytes = None
    weight_file_path = None

    if weight_path_input.strip():
        p = Path(weight_path_input.strip())
        if p.is_file():
            weight_mode = "path"
            weight_file_path = p
        else:
            st.warning("权重路径不存在。")
    elif weight_select != "(未选择)":
        p = Path(weight_select)
        if p.is_file():
            weight_mode = "path"
            weight_file_path = p
    elif uploaded_w is not None:
        weight_mode = "upload"
        weight_bytes = uploaded_w.read()
    elif auto_from_train and st.session_state.last_train_result:
        p = Path(st.session_state.last_train_result["best_path"])
        if p.exists():
            weight_mode = "path"
            weight_file_path = p
            auto_meta_path = Path(st.session_state.last_train_result["meta_path"])
            if auto_meta_path.exists() and not meta_path_input.strip():
                meta_path_input = str(auto_meta_path)

    meta_obj = None
    if meta_path_input.strip():
        mp = Path(meta_path_input.strip())
        if mp.is_file():
            try:
                with open(mp, "r", encoding="utf-8") as f:
                    meta_obj = json.load(f)
            except Exception as e:
                st.error(f"读取 meta.json 失败：{e}")
    elif meta_json_obj:
        meta_obj = meta_json_obj

    if meta_obj:
        chosen_mean = tuple(meta_obj.get("mean", DATASETS_INFO["cifar10"]["mean"]))
        chosen_std = tuple(meta_obj.get("std", DATASETS_INFO["cifar10"]["std"]))
        chosen_dataset_name = meta_obj.get("dataset", infer_dataset_choice)
        chosen_classes = meta_obj.get("classes", None)
    else:
        info = DATASETS_INFO[infer_dataset_choice]
        chosen_mean, chosen_std = info["mean"], info["std"]
        chosen_dataset_name = infer_dataset_choice
        chosen_classes = None

    st.caption(f"Mean: {chosen_mean} | Std: {chosen_std} | 数据集(推断): {chosen_dataset_name}")

    @st.cache_resource(show_spinner=False)
    def load_test_dataset(ds_name: str, data_dir: str, mean, std):
        tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
        if ds_name == "cifar10":
            ds = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=tf)
        else:
            ds = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=tf)
        return ds

    test_dataset = None
    if prepare_test_btn:
        try:
            test_dataset = load_test_dataset(infer_dataset_choice, data_dir_infer, chosen_mean, chosen_std)
            if not chosen_classes:
                chosen_classes = test_dataset.classes
            st.success(f"测试集已准备：{infer_dataset_choice} ({len(test_dataset)})")
        except Exception as e:
            st.error(f"加载测试集失败：{e}")
    else:
        try:
            test_dataset = load_test_dataset(infer_dataset_choice, data_dir_infer, chosen_mean, chosen_std)
            if not chosen_classes:
                chosen_classes = test_dataset.classes
        except Exception:
            pass

    device = get_device(prefer_cpu=(device_pref == "cpu"))

    load_model_btn = st.button("加载 / 应用模型")
    if load_model_btn:
        if weight_mode == "none":
            st.error("未提供任何权重。")
        else:
            try:
                if not chosen_classes:
                    info = DATASETS_INFO[chosen_dataset_name]
                    chosen_classes = [str(i) for i in range(info["num_classes"])]
                model = build_cifar_resnet18(num_classes=len(chosen_classes))
                if weight_mode == "path":
                    missing, unexpected = load_weights_from_path(model, weight_file_path, device)
                    st.success(f"加载成功 (路径) {weight_file_path.name} | Missing:{len(missing)} Unexpected:{len(unexpected)}")
                else:
                    missing, unexpected = load_weights_from_bytes(model, weight_bytes, device)
                    st.success(f"加载成功 (上传) | Missing:{len(missing)} Unexpected:{len(unexpected)}")
                model.to(device)
                st.session_state.loaded_model = model
                st.session_state.loaded_meta = meta_obj
                st.session_state.loaded_classes = chosen_classes
                st.session_state.loaded_mean = chosen_mean
                st.session_state.loaded_std = chosen_std
                st.session_state.loaded_dataset = chosen_dataset_name
                st.session_state.model_ready = True
                st.session_state.confusion_cache = None
            except Exception as e:
                st.error(f"模型加载失败：{e}")

    infer_tabs = st.tabs(["自定义图片预测", "测试集浏览", "混淆矩阵 / 误分类分析"])

    with infer_tabs[0]:
        if not st.session_state.model_ready:
            st.info("请先加载模型。")
        else:
            up_img = st.file_uploader("上传图片（自动缩放到 32x32）", type=["png","jpg","jpeg"])
            if up_img:
                try:
                    img = Image.open(up_img).convert("RGB")
                    c1, c2 = st.columns(2)
                    c1.image(img, caption="原图", use_container_width=True)
                    tfm = T.Compose([
                        T.Resize((32,32)),
                        T.ToTensor(),
                        T.Normalize(st.session_state.loaded_mean, st.session_state.loaded_std)
                    ])
                    t = tfm(img)
                    tk = predict_single(st.session_state.loaded_model, t.to(device), device, topk=topk)
                    lines = []
                    for idx_, prob in tk:
                        cname = st.session_state.loaded_classes[idx_] if idx_ < len(st.session_state.loaded_classes) else str(idx_)
                        lines.append(f"{cname}: {prob*100:.2f}%")
                    c2.markdown("**Top-K 结果**\n" + "\n".join(lines))
                except Exception as e:
                    st.error(f"处理失败：{e}")
            else:
                st.info("请上传图片。")

    with infer_tabs[1]:
        if not st.session_state.model_ready:
            st.info("请先加载模型。")
        elif test_dataset is None:
            st.info("测试集未准备。")
        else:
            mode = st.radio("模式", ["单张查看", "批量随机"], horizontal=True)
            classes = st.session_state.loaded_classes
            if mode == "单张查看":
                idx = st.slider("索引", 0, len(test_dataset)-1, 0, 1)
                img_t, gt = test_dataset[idx]
                pil = denorm_to_pil(img_t, st.session_state.loaded_mean, st.session_state.loaded_std)
                c1, c2 = st.columns(2)
                c1.image(pil, caption=f"Index {idx}", use_container_width=True)
                tk = predict_single(st.session_state.loaded_model, img_t.to(device), device, topk=topk)
                top1_idx = tk[0][0]
                top1_name = classes[top1_idx]
                gt_name = classes[gt]
                correct = (top1_idx == gt)
                c2.markdown(f"**GT:** {gt_name}")
                c2.markdown(f"**预测:** {'True' if correct else 'False'} {top1_name}")
                lines = []
                for i_, p_ in tk:
                    cname = classes[i_]
                    tag = " (GT)" if i_ == gt else ""
                    lines.append(f"{cname}: {p_*100:.2f}%{tag}")
                c2.markdown("**Top-K 概率**\n" + "\n".join(lines))
            else:
                n = st.slider("随机数量", 4, 40, 12, 4)
                reroll = st.button("重新抽取")
                if reroll or "rand_vis_indices" not in st.session_state:
                    st.session_state.rand_vis_indices = random.sample(range(len(test_dataset)), n)
                cols_per_row = 6
                rows = (n + cols_per_row - 1) // cols_per_row
                with torch.inference_mode():
                    for r in range(rows):
                        cols = st.columns(cols_per_row)
                        for c in range(cols_per_row):
                            pos = r * cols_per_row + c
                            if pos >= n:
                                break
                            ds_idx = st.session_state.rand_vis_indices[pos]
                            img_t, gt = test_dataset[ds_idx]
                            tk = predict_single(st.session_state.loaded_model, img_t.to(device), device, topk=1)
                            pred_idx = tk[0][0]
                            correct = (pred_idx == gt)
                            pil = denorm_to_pil(img_t, st.session_state.loaded_mean, st.session_state.loaded_std)
                            caption = f"{'True' if correct else 'False'} {classes[pred_idx]} / GT:{classes[gt]}"
                            cols[c].image(pil, caption=caption, use_container_width=True)

    with infer_tabs[2]:
        st.markdown("### 混淆矩阵 & 误分类分析 (T=真实类, P=预测类)")
        if not st.session_state.model_ready:
            st.info("请先加载模型。")
        elif test_dataset is None:
            st.info("测试集未准备。")
        else:
            compute_btn = st.button("计算 / 刷新 混淆矩阵")
            if compute_btn:
                dl = torch.utils.data.DataLoader(
                    test_dataset, batch_size=int(eval_batch_size), shuffle=False,
                    num_workers=2, pin_memory=True
                )
                st.info("计算中...")
                prog = st.progress(0)
                preds_all, targets_all = [], []
                total_batches = len(dl)
                with torch.inference_mode():
                    for i, (x, y) in enumerate(dl):
                        x = x.to(device, non_blocking=True)
                        out = st.session_state.loaded_model(x)
                        pred = out.argmax(1).cpu()
                        preds_all.append(pred)
                        targets_all.append(y)
                        prog.progress(int((i+1)/total_batches*100))
                preds_cat = torch.cat(preds_all)
                targets_cat = torch.cat(targets_all)
                num_classes = len(st.session_state.loaded_classes)
                cm = compute_confusion_matrix(preds_cat, targets_cat, num_classes)
                metrics = metrics_from_confusion(cm)
                st.session_state.confusion_cache = {
                    "cm": cm,
                    "metrics": metrics,
                    "classes": st.session_state.loaded_classes,
                    "dataset": st.session_state.loaded_dataset,
                    "total": int(cm.sum().item())
                }
                st.success("混淆矩阵完成。")

            if st.session_state.confusion_cache:
                cache = st.session_state.confusion_cache
                cm = cache["cm"]
                classes = cache["classes"]
                metrics = cache["metrics"]
                st.markdown(
                    f"**整体准确率**: {metrics['overall_acc']:.2f}% | "
                    f"宏平均 Precision: {metrics['macro_precision']:.2f}% | "
                    f"宏平均 Recall: {metrics['macro_recall']:.2f}% | "
                    f"宏平均 F1: {metrics['macro_f1']:.2f}%"
                )

                num_classes = len(classes)
                st.markdown("#### 混淆矩阵显示")
                normalize = st.checkbox("行归一化（每真实类行和=1）", value=False)
                if num_classes > 20:
                    st.warning("类别 >20，建议筛选子集查看。")
                    selected_classes = st.multiselect(
                        "选择要显示的类别（<=20）",
                        options=list(range(num_classes)),
                        default=list(range(min(10, num_classes))),
                        format_func=lambda i: f"{i}:{classes[i]}"
                    )
                    if selected_classes:
                        sub_cm = cm[selected_classes][:, selected_classes].float()
                        if normalize:
                            sub_cm = sub_cm / sub_cm.sum(dim=1, keepdim=True).clamp(min=1)
                        df = pd.DataFrame(
                            sub_cm.numpy(),
                            index=[f"T:{classes[i]}" for i in selected_classes],
                            columns=[f"P:{classes[i]}" for i in selected_classes]
                        )
                        st.dataframe(df, use_container_width=True, height=400)
                else:
                    show_cm = cm.float()
                    if normalize:
                        show_cm = show_cm / show_cm.sum(dim=1, keepdim=True).clamp(min=1)
                    df_full = pd.DataFrame(
                        show_cm.numpy(),
                        index=[f"T:{c}" for c in classes],
                        columns=[f"P:{c}" for c in classes]
                    )
                    st.dataframe(df_full, use_container_width=True, height=500)

                st.markdown("#### Top 误分类对")
                top_n = st.slider("显示 Top N", 5, 50, 15, 5)
                mis_top = top_misclass_pairs(cm, classes, top_n=top_n)
                if mis_top:
                    mis_df = pd.DataFrame(mis_top, columns=["真实类", "被误判为", "次数"])
                    st.table(mis_df)
                else:
                    st.info("没有误分类。")

                st.markdown("#### 下载混淆矩阵 CSV")
                csv_raw = pd.DataFrame(
                    cm.numpy(),
                    index=[f"T:{c}" for c in classes],
                    columns=[f"P:{c}" for c in classes]
                ).to_csv().encode("utf-8")
                st.download_button("下载原始矩阵", data=csv_raw, file_name="confusion_matrix_raw.csv")

                norm_cm = cm.float()
                norm_cm = norm_cm / norm_cm.sum(dim=1, keepdim=True).clamp(min=1)
                csv_norm = pd.DataFrame(
                    norm_cm.numpy(),
                    index=[f"T:{c}" for c in classes],
                    columns=[f"P:{c}" for c in classes]
                ).to_csv().encode("utf-8")
                st.download_button("下载行归一化矩阵", data=csv_norm, file_name="confusion_matrix_row_norm.csv")