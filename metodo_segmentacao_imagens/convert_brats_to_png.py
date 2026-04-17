import argparse
from pathlib import Path
import numpy as np
import imageio
import nibabel as nib

MODALITY_MAP = {
    "flair": "t2f",
    "t2f": "t2f",
    "t2w": "t2w",
    "t1c": "t1c",
    "t1n": "t1n",
}


def normalize_slice(slice_data: np.ndarray) -> np.ndarray:
    slice_data = np.nan_to_num(slice_data.astype(np.float32))
    min_val = float(np.min(slice_data))
    max_val = float(np.max(slice_data))
    if max_val <= min_val:
        return np.zeros_like(slice_data, dtype=np.uint8)
    normalized = (slice_data - min_val) / (max_val - min_val)
    return (normalized * 255.0).astype(np.uint8)


def save_image(image_slice: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(output_path), image_slice)


def get_case_ids(data_root: Path):
    return sorted([p.name for p in data_root.iterdir() if p.is_dir()])


def parse_modalities(modality_arg: str):
    mods = [m.strip().lower() for m in modality_arg.split(",") if m.strip()]
    mapped = []
    for mod in mods:
        if mod not in MODALITY_MAP:
            raise ValueError(f"Modalidade desconhecida: {mod}")
        mapped.append(MODALITY_MAP[mod])
    return mapped


def build_image_path(case_id: str, modality: str, source_root: Path):
    return source_root / "data" / case_id / f"{case_id}-{modality}.nii.gz"


def build_label_path(case_id: str, source_root: Path):
    return source_root / "labels" / f"{case_id}-seg.nii.gz"


def load_volume(path: Path):
    img = nib.load(str(path))
    return img.get_fdata(dtype=np.float32)


def convert_case(case_id: str, source_root: Path, output_root: Path, modalities, split_set: str, min_mask_pixels: int):
    image_volumes = []
    for modality in modalities:
        image_path = build_image_path(case_id, modality, source_root)
        if not image_path.exists():
            print(f"Aviso: arquivo de imagem ausente para {case_id} ({modality}), pulando")
            return 0, 0
        image_volumes.append(load_volume(image_path))

    label_path = build_label_path(case_id, source_root)
    if not label_path.exists():
        print(f"Aviso: arquivo de máscara ausente para {case_id}, pulando")
        return 0, 0
    label_volume = load_volume(label_path)

    if any(vol.ndim != 3 for vol in image_volumes) or label_volume.ndim != 3:
        raise ValueError("Todos os volumes devem ser 3D.")
    if any(vol.shape != label_volume.shape for vol in image_volumes):
        raise ValueError(f"Dimensões inconsistentes para {case_id}")

    count_saved = 0
    total_slices = label_volume.shape[2]
    for slice_idx in range(total_slices):
        mask_slice = label_volume[:, :, slice_idx]
        if np.count_nonzero(mask_slice) < min_mask_pixels:
            continue

        image_slices = [normalize_slice(vol[:, :, slice_idx]) for vol in image_volumes]
        image_array = np.stack(image_slices, axis=-1)
        mask_slice = (mask_slice > 0).astype(np.uint8)

        image_name = f"{case_id}_{slice_idx:03d}.png"
        save_image(image_array, output_root / split_set / "images" / image_name)
        save_image(mask_slice, output_root / split_set / "masks" / image_name)
        count_saved += 1

    print(f"{case_id}: {count_saved} fatias salvas ({total_slices} totais)")
    return count_saved, total_slices


def split_case_ids(case_ids, val_ratio):
    split_idx = int(len(case_ids) * (1.0 - val_ratio))
    return case_ids[:split_idx], case_ids[split_idx:]


def parse_args():
    parser = argparse.ArgumentParser(description="Converte BraTS NIfTI para PNG 2D no layout do pipeline")
    parser.add_argument("--source-root", type=Path, default=Path("SYNAPSE/BraTS-GLI-fastlane"), help="Raiz do pacote BraTS extraído")
    parser.add_argument("--output-root", type=Path, default=Path("data"), help="Diretório de saída para train/val")
    parser.add_argument("--modalities", type=str, default="t1c,t1n,t2f,t2w", help="Modalidades a serem empilhadas como canais (ex: t1c,t1n,t2f,t2w)")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Porcentagem de casos para validação")
    parser.add_argument("--min-mask-pixels", type=int, default=1, help="Número mínimo de pixels de máscara para salvar a fatia")
    parser.add_argument("--no-split", action="store_true", help="Gerar todos os arquivos em data/all em vez de separar train/val")
    return parser.parse_args()


def main():
    args = parse_args()
    source_root = args.source_root
    output_root = args.output_root
    modalities = parse_modalities(args.modalities)

    if not source_root.exists():
        raise FileNotFoundError(f"Diretório de origem não encontrado: {source_root}")

    case_ids = get_case_ids(source_root / "data")
    if not case_ids:
        raise ValueError("Nenhum caso encontrado em source_root/data")

    if args.no_split:
        val_ratio = 0.0
    else:
        val_ratio = args.val_ratio

    train_ids, val_ids = split_case_ids(case_ids, val_ratio)
    if args.no_split:
        val_ids = []

    print(f"Modalidades: {modalities}")
    print(f"Casos totais: {len(case_ids)}, train: {len(train_ids)}, val: {len(val_ids)}")

    for case_id in train_ids:
        convert_case(case_id, source_root, output_root, modalities, "train", args.min_mask_pixels)

    for case_id in val_ids:
        convert_case(case_id, source_root, output_root, modalities, "val", args.min_mask_pixels)

    print("Conversão concluída.")


if __name__ == "__main__":
    main()
