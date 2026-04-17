import numpy as np
import nibabel as nib
from pathlib import Path

from convert_brats_to_png import convert_case


def test_convert_case_creates_png(tmp_path):
    source_root = tmp_path / "SYNAPSE"
    data_dir = source_root / "data"
    labels_dir = source_root / "labels"
    case_id = "case001"
    case_dir = data_dir / case_id
    case_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    volume_shape = (2, 2, 3)
    for modality in ["t1c", "t1n", "t2w", "t2f"]:
        arr = np.arange(np.product(volume_shape), dtype=np.float32).reshape(volume_shape)
        nib.save(nib.Nifti1Image(arr, np.eye(4)), case_dir / f"{case_id}-{modality}.nii.gz")

    seg = np.zeros(volume_shape, dtype=np.float32)
    seg[:, :, 1] = 1.0
    nib.save(nib.Nifti1Image(seg, np.eye(4)), labels_dir / f"{case_id}-seg.nii.gz")

    output_root = tmp_path / "data_out"
    count_saved, total_slices = convert_case(case_id, source_root, output_root, ["t1c", "t1n", "t2w", "t2f"], "train", min_mask_pixels=1)

    assert count_saved > 0
    assert total_slices == 3
    assert (output_root / "train" / "images").exists()
    assert (output_root / "train" / "masks").exists()
    saved_images = list((output_root / "train" / "images").glob("*.png"))
    saved_masks = list((output_root / "train" / "masks").glob("*.png"))
    assert len(saved_images) == len(saved_masks) == count_saved
