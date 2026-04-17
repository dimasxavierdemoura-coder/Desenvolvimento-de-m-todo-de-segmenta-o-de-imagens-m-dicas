import argparse
import csv
from pathlib import Path

import torch

from segmentation_pipeline import build_model, create_dataloader, dice_score, iou_score


def save_report(report_path: Path, rows):
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["image", "mask", "dice", "iou"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = args.checkpoint or args.output_dir / "unet_tumor.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint_path}")

    model = build_model(device, in_channels=args.in_channels)
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    split_dir = args.data_dir / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Diretório de avaliação não encontrado: {split_dir}")

    loader = create_dataloader(split_dir, batch_size=1, image_size=args.image_size, is_train=False)
    rows = []
    total_dice = 0.0
    total_iou = 0.0
    count = 0

    with torch.no_grad():
        for idx, sample in enumerate(loader.dataset):
            image = sample["image"].unsqueeze(0).to(device)
            label = sample["mask"].unsqueeze(0)
            output = torch.sigmoid(model(image))
            pred = (output > 0.5).float().cpu()
            dice = dice_score(pred, label).item()
            iou = iou_score(pred, label).item()
            image_path = loader.dataset.data[idx]["image"]
            mask_path = loader.dataset.data[idx]["mask"]
            rows.append({
                "image": Path(image_path).name,
                "mask": Path(mask_path).name,
                "dice": f"{dice:.6f}",
                "iou": f"{iou:.6f}",
            })
            total_dice += dice
            total_iou += iou
            count += 1

    if count == 0:
        raise ValueError(f"Nenhuma imagem encontrada em {split_dir}")

    save_report(args.output_report, rows)
    mean_dice = total_dice / count
    mean_iou = total_iou / count
    print(f"Avaliação concluída: {count} amostras")
    print(f"Dice médio: {mean_dice:.4f}, IoU médio: {mean_iou:.4f}")
    print(f"Relatório salvo em: {args.output_report}")


def parse_args():
    parser = argparse.ArgumentParser(description="Avaliação de modelo de segmentação médica")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Diretório raiz do dataset")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Conjunto para avaliação")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Diretório para salvar relatório")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint do modelo")
    parser.add_argument("--in-channels", type=int, default=4, help="Número de canais de entrada")
    parser.add_argument("--image-size", nargs=2, type=int, default=[128, 128], help="Tamanho para redimensionamento")
    parser.add_argument("--output-report", type=Path, default=Path("output/evaluation_report.csv"), help="Caminho CSV de relatório")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
