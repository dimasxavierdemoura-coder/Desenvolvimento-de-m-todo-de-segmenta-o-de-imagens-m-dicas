import argparse
import csv
from datetime import datetime
from pathlib import Path
import numpy as np
import cv2
import nibabel as nib
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.data import Dataset
from monai.losses import DiceLoss
from monai.networks.nets import UNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Lambda,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    Resized,
    ScaleIntensityd,
    EnsureType,
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensity,
    Resize,
)
import matplotlib.pyplot as plt

def get_image_mask_pairs(data_dir: Path):
    image_dir = data_dir / "images"
    mask_dir = data_dir / "masks"
    if not image_dir.exists() or not mask_dir.exists():
        raise FileNotFoundError(
            f"Dataset layout não encontrado. Crie '{image_dir}' e '{mask_dir}' com imagens e máscaras."
        )

    image_paths = sorted([p for p in image_dir.iterdir() if p.is_file()])
    mask_paths = sorted([p for p in mask_dir.iterdir() if p.is_file()])

    masks_by_stem = {p.stem: p for p in mask_paths}
    pairs = []
    for image_path in image_paths:
        mask_path = masks_by_stem.get(image_path.stem)
        if mask_path is None:
            continue
        pairs.append({"image": str(image_path), "mask": str(mask_path)})

    if len(pairs) == 0:
        raise ValueError("Nenhuma imagem e máscara casaram. Verifique os nomes dos arquivos.")
    return pairs


class BinarizeMaskd:
    def __call__(self, data):
        data["mask"] = (data["mask"] > 0).astype(np.float32)
        return data


def make_transforms(image_size=(128, 128), is_train=True):
    transforms = [
        LoadImaged(keys=["image", "mask"], image_only=True),
        EnsureChannelFirstd(keys=["image", "mask"]),
        ScaleIntensityd(keys=["image"]),
        BinarizeMaskd(),
        Resized(keys=["image", "mask"], spatial_size=image_size),
    ]
    if is_train:
        transforms += [
            RandRotate90d(keys=["image", "mask"], prob=0.5, max_k=3),
            RandFlipd(keys=["image", "mask"], spatial_axis=0, prob=0.5),
        ]
    transforms += [EnsureTyped(keys=["image", "mask"])]
    return Compose(transforms)


def create_dataloader(data_dir: Path, batch_size=8, image_size=(128, 128), is_train=True):
    validate_data_dir(data_dir)
    data = get_image_mask_pairs(data_dir)
    transforms = make_transforms(image_size=image_size, is_train=is_train)
    dataset = Dataset(data=data, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=4, pin_memory=torch.cuda.is_available())


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    pred = pred.float()
    target = target.float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target)
    return (2.0 * intersection + eps) / (union + eps)


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps=1e-6):
    pred = pred.float()
    target = target.float()
    intersection = torch.sum(pred * target)
    union = torch.sum(pred + target - pred * target)
    return (intersection + eps) / (union + eps)


def build_model(device, in_channels=4):
    model = UNet(
        spatial_dims=2,
        in_channels=in_channels,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    return model.to(device)


def post_process_mask(mask: np.ndarray):
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cleaned


def validate_data_dir(data_dir: Path):
    if not data_dir.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {data_dir}")
    if not (data_dir / "images").exists() or not (data_dir / "masks").exists():
        raise FileNotFoundError(
            f"Estrutura de dataset inválida. Esperado: {data_dir}/images e {data_dir}/masks"
        )


def validate_checkpoint(path: Path):
    if path is None or not path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {path}")


def save_metrics_csv(path: Path, row: dict):
    header = ["epoch", "train_loss", "val_dice", "val_iou"]
    file_exists = path.exists()
    with path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def append_log(path: Path, message: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def ensure_input_channels(tensor: torch.Tensor, desired_channels: int):
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[0] == 1 and desired_channels > 1:
        tensor = tensor.repeat(desired_channels, 1, 1)
    elif tensor.ndim == 3 and tensor.shape[0] != desired_channels:
        if tensor.shape[0] > desired_channels:
            tensor = tensor[:desired_channels]
        else:
            padding = desired_channels - tensor.shape[0]
            pad_shape = (padding, tensor.shape[1], tensor.shape[2])
            tensor = torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype)], dim=0)
    return tensor


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, alpha=0.4, color=(255, 0, 0)):
    image = image.astype(np.float32)
    image -= image.min()
    if image.max() > 0:
        image = image / image.max()
    image_rgb = (image * 255).astype(np.uint8)
    image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
    color_mask = np.zeros_like(image_rgb)
    color_mask[mask > 0] = color
    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, color_mask, alpha, 0)
    return overlay


def find_brats_modalities(volume_dir: Path):
    candidates = {"t1c": None, "t1n": None, "t2w": None, "t2f": None}
    for path in sorted(volume_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name.lower()
        if not name.endswith(".nii") and not name.endswith(".nii.gz"):
            continue
        for key in candidates:
            if key in name and candidates[key] is None:
                candidates[key] = path
    if any(v is None for v in candidates.values()):
        missing = [k for k, v in candidates.items() if v is None]
        raise ValueError(
            f"Não foram encontradas todas as modalidades BraTS em {volume_dir}. Faltando: {missing}."
        )
    return [candidates[key] for key in ["t1c", "t1n", "t2w", "t2f"]]


def make_infer_transforms(image_size=(128, 128)):
    return Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize(image_size),
        EnsureType(),
    ])


def make_array_infer_transforms(image_size=(128, 128)):
    return Compose([
        ScaleIntensity(),
        Resize(image_size),
        EnsureType(),
    ])


def predict_mask(image_tensor, model, device):
    if isinstance(image_tensor, np.ndarray):
        image_tensor = torch.from_numpy(image_tensor)
    image_tensor = image_tensor.float()
    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0)
    desired_channels = getattr(model, "in_channels", image_tensor.shape[0])
    image_tensor = ensure_input_channels(image_tensor, desired_channels)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = torch.sigmoid(model(image_tensor))
        mask = (output[0, 0] > 0.5).cpu().numpy().astype(np.uint8)
    return post_process_mask(mask)


def show_prediction(image_np: np.ndarray, mask: np.ndarray, title=""):
    if image_np.ndim == 3 and image_np.shape[0] > 1:
        image_np = np.mean(image_np, axis=0)
    if image_np.shape != mask.shape:
        image_np = cv2.resize(image_np.astype(np.float32), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = overlay_mask_on_image(image_np, mask)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np, cmap="gray")
    plt.title("Imagem")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Máscara")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def infer_single_image(image_path: Path, model, device, args):
    transforms = make_infer_transforms(image_size=args.image_size)
    image = transforms(str(image_path))
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image)
    mask = predict_mask(image, model, device)
    image_np = image.cpu().numpy()
    if image_np.ndim == 3 and image_np.shape[0] > 1:
        image_np = np.mean(image_np, axis=0)
    else:
        image_np = image_np[0]
    return image_np, mask


def infer_directory(args, model, device):
    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_files = sorted([p for p in args.image.iterdir() if p.suffix.lower() in valid_exts])
    if len(image_files) == 0:
        raise ValueError(f"Nenhuma imagem encontrada no diretório {args.image}")

    output_dir = None
    if args.output_mask:
        if args.output_mask.exists() and args.output_mask.is_dir():
            output_dir = args.output_mask
        elif not args.output_mask.suffix:
            output_dir = args.output_mask
        else:
            raise ValueError("Para inferência em lote, --output-mask deve ser um diretório ou não especificado.")
    if output_dir is None:
        output_dir = args.output_dir / "batch_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, image_path in enumerate(image_files):
        image_np, mask = infer_single_image(image_path, model, device, args)
        mask_path = output_dir / f"{image_path.stem}_mask.png"
        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))
        print(f"Predição salva: {mask_path}")
        if idx < args.display_count:
            show_prediction(image_np, mask, title=f"Batch {idx + 1}: {image_path.name}")


def infer_volume(args, model, device):
    if args.image.is_dir():
        modality_files = find_brats_modalities(args.image)
        volume = np.stack(
            [nib.load(str(path)).get_fdata(dtype=np.float32) for path in modality_files],
            axis=0,
        )
    else:
        volume = nib.load(str(args.image)).get_fdata(dtype=np.float32)
        if volume.ndim == 3:
            volume = np.expand_dims(volume, 0)

    if volume.ndim != 4 or volume.shape[0] not in {1, 4}:
        raise ValueError(
            "Inferência 3D exige volume com shape (C,H,W,D) onde C=1 ou C=4 para BraTS." 
        )

    transforms = make_array_infer_transforms(image_size=args.image_size)
    mask_slices = []
    for z in range(volume.shape[-1]):
        slice_image = volume[..., z].astype(np.float32)
        image_tensor = transforms(slice_image)
        mask = predict_mask(image_tensor, model, device)
        mask_slices.append(mask)

    mask_volume = np.stack(mask_slices, axis=-1).astype(np.uint8)
    display_slices = [0, volume.shape[-1] // 2, volume.shape[-1] - 1]
    for idx in display_slices:
        image_np = volume[..., idx]
        mask = mask_volume[..., idx]
        show_prediction(image_np, mask, title=f"Slice {idx + 1}/{volume.shape[-1]}")

    if args.output_mask:
        output_path = args.output_mask
    else:
        output_path = args.output_dir / f"{args.image.stem}_pred.nii.gz"
    if output_path.suffix.lower() in {".nii", ".gz"}:
        if args.image.is_dir():
            affine = nib.load(str(modality_files[0])).affine
        else:
            affine = nib.load(str(args.image)).affine
        nib.save(nib.Nifti1Image(mask_volume, affine), str(output_path))
        print(f"Volume de máscara salvo em: {output_path}")
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(output_path), mask_volume)
        print(f"Volume de máscara salvo em: {output_path}.npy")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dir = args.data_dir / "train"
    val_dir = args.data_dir / "val"
    train_loader = create_dataloader(train_dir, batch_size=args.batch_size, image_size=args.image_size, is_train=True)
    val_loader = create_dataloader(val_dir, batch_size=args.batch_size, image_size=args.image_size, is_train=False)

    model = build_model(device, in_channels=args.in_channels)
    loss_function = DiceLoss(sigmoid=True)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = args.output_dir / "unet_tumor.pth"
    latest_checkpoint = args.output_dir / "latest.pth"
    metrics_path = args.output_dir / "train_metrics.csv"
    log_path = args.log_file or args.output_dir / "train.log"

    best_val_dice = 0.0
    start_epoch = 0
    if args.resume:
        if not args.resume.exists():
            raise FileNotFoundError(f"Checkpoint de retreinamento não encontrado: {args.resume}")
        checkpoint = torch.load(str(args.resume), map_location=device)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", optimizer.state_dict()))
        start_epoch = checkpoint.get("epoch", 0)
        best_val_dice = checkpoint.get("best_val_dice", 0.0)
        print(f"Retomando treino do checkpoint {args.resume} a partir da época {start_epoch}")

    if start_epoch == 0 and metrics_path.exists():
        metrics_path.unlink()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch_data in train_loader:
            inputs, labels = batch_data["image"].to(device), batch_data["mask"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch + 1}/{args.epochs} - Train loss: {epoch_loss:.4f}")
        append_log(log_path, f"{datetime.now()} - Epoch {epoch + 1}: Train loss {epoch_loss:.4f}")

        model.eval()
        val_dice = 0.0
        val_iou = 0.0
        with torch.no_grad():
            for batch_data in val_loader:
                inputs, labels = batch_data["image"].to(device), batch_data["mask"].to(device)
                outputs = torch.sigmoid(model(inputs))
                preds = (outputs > 0.5).float()
                val_dice += dice_score(preds, labels)
                val_iou += iou_score(preds, labels)

            val_dice /= len(val_loader)
            val_iou /= len(val_loader)

        print(f"Validation Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        append_log(log_path, f"{datetime.now()} - Epoch {epoch + 1}: Val Dice {val_dice:.4f}, IoU {val_iou:.4f}")

        save_metrics_csv(metrics_path, {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_dice": float(val_dice),
            "val_iou": float(val_iou),
        })

        checkpoint_data = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_dice": best_val_dice,
        }
        torch.save(checkpoint_data, latest_checkpoint)

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            checkpoint_data["best_val_dice"] = best_val_dice
            torch.save(checkpoint_data, best_model_path)
            print(f"Novo melhor modelo salvo em: {best_model_path}")

    print(f"Treinamento finalizado. Melhor Dice de validação: {best_val_dice:.4f}")


def infer(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device, in_channels=args.in_channels)
    checkpoint_path = args.checkpoint or args.output_dir / "unet_tumor.pth"
    validate_checkpoint(checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if args.image is None:
        raise ValueError("Para inferência, use --image <imagem ou diretório> e --checkpoint <arquivo>")
    if not args.image.exists():
        raise FileNotFoundError(f"Arquivo ou diretório para inferência não encontrado: {args.image}")

    if args.image.is_dir():
        nifti_files = [p for p in args.image.iterdir() if p.suffix.lower() in {".nii", ".gz"}]
        if len(nifti_files) > 0:
            infer_volume(args, model, device)
        else:
            infer_directory(args, model, device)
        return

    if args.image.suffix.lower() in {".nii", ".gz"}:
        infer_volume(args, model, device)
        return

    image_np, mask = infer_single_image(args.image, model, device, args)
    show_prediction(image_np, mask)

    if args.output_mask:
        output_path = Path(args.output_mask)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), (mask * 255).astype(np.uint8))
        print(f"Máscara salva em: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline de segmentação médica com MONAI e PyTorch")
    parser.add_argument("--mode", choices=["train", "infer"], default="train", help="Modo de execução")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Diretório raiz com train/ e val/")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Diretório para checkpoints e outputs")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas de treinamento")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamanho do batch")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Taxa de aprendizado")
    parser.add_argument("--image-size", nargs=2, type=int, default=[128, 128], help="Tamanho fixo para redimensionamento")
    parser.add_argument("--in-channels", type=int, default=4, help="Número de canais de entrada para o modelo")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint para inferência")
    parser.add_argument("--resume", type=Path, help="Checkpoint para retomar treino")
    parser.add_argument("--log-file", type=Path, help="Arquivo de log de treino")
    parser.add_argument("--image", type=Path, help="Imagem ou diretório para inferência")
    parser.add_argument("--output-mask", type=Path, help="Caminho para salvar máscara de inferência; em lote deve ser diretório")
    parser.add_argument("--display-count", type=int, default=4, help="Número de predições em lote exibidas")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        if args.image is None:
            raise ValueError("Para inferência, use --image <imagem> e opcionalmente --checkpoint <arquivo>")
        infer(args)
