# Pipeline de Segmentação Médica

Pipeline completo para segmentação médica usando **PyTorch** e **MONAI**, com suporte a:

- conversão de BraTS NIfTI para dataset 2D PNG
- treino de U-Net 2D multicanal
- inferência em imagens 2D únicas e em lote
- inferência 3D a partir de volumes NIfTI
- inferência multicanal BraTS 3D
- avaliação com geração de relatório CSV
- retreinamento a partir de checkpoint

## Estrutura do projeto

- `segmentation_pipeline.py` — treino e inferência
- `convert_brats_to_png.py` — conversão BraTS `.nii.gz` para PNG 2D
- `evaluate.py` — avaliação com métricas Dice/IoU
- `requirements.txt` — dependências Python
- `environment.yml` — ambiente Conda
- `data/` — dataset de entrada
- `output/` — checkpoints, máscaras e relatórios
- `tests/` — testes unitários

## Instalação

### Usando `pip`

```bash
pip install -r requirements.txt
```

### Usando `conda`

```bash
conda env create -f environment.yml
conda activate segmentation-env
pip install -r requirements.txt
```

## Layout de dataset esperado

O pipeline de treino requer:

```
data/
  train/
    images/
    masks/
  val/
    images/
    masks/
```

- Imagens e máscaras devem ter o mesmo nome base.
- Imagens podem ser multicanais (por exemplo, 4 canais para BraTS).
- Máscaras devem ser binárias ou binarizáveis.

## Conversão de BraTS NIfTI para PNG 2D

Use `convert_brats_to_png.py` para converter o dataset BraTS para o formato 2D do pipeline.

### Exemplo

```bash
python convert_brats_to_png.py \
  --source-root SYNAPSE/BraTS-GLI-fastlane \
  --output-root data \
  --modalities t1c,t1n,t2f,t2w \
  --val-ratio 0.2
```

### O script realiza

- leitura de casos em `source-root/data`
- carregamento de modalidades BraTS em `t1c`, `t1n`, `t2w`, `t2f`
- normalização de fatias para 0-255
- salvamento de imagens PNG multicanais em `output-root/<train|val>/images`
- salvamento de máscaras binárias em `output-root/<train|val>/masks`
- separação de casos entre `train` e `val`

### Modalidades suportadas

- `t1c`
- `t1n`
- `t2w`
- `t2f`
- `flair` (alias para `t2f`)

### Gerar sem divisão train/val

```bash
python convert_brats_to_png.py --no-split --output-root data/all
```

## Treinamento

```bash
python segmentation_pipeline.py \
  --mode train \
  --data-dir data \
  --output-dir output \
  --in-channels 4 \
  --epochs 50 \
  --batch-size 8 \
  --image-size 128 128
```

### Principais parâmetros

- `--mode train` — modo de treino
- `--data-dir` — diretório raiz com `train/` e `val/`
- `--output-dir` — diretório de saída
- `--in-channels` — canais de entrada do modelo
- `--epochs` — número de épocas
- `--batch-size` — tamanho do batch
- `--learning-rate` — taxa de aprendizado
- `--image-size` — redimensionamento `HxW`
- `--resume` — checkpoint para retomar treino
- `--log-file` — arquivo de log de treino

### Modelo

- UNet 2D do MONAI
- `in_channels` configurável
- `out_channels=1`
- perda `DiceLoss(sigmoid=True)`
- métricas de validação: Dice e IoU

### Saídas de treino

- `output/unet_tumor.pth` — melhor checkpoint salvo
- `output/latest.pth` — último checkpoint
- `output/train_metrics.csv` — métricas por época
- `output/train.log` — log de treino

## Inferência

### 1) Imagem 2D única

```bash
python segmentation_pipeline.py \
  --mode infer \
  --checkpoint output/unet_tumor.pth \
  --image data/val/images/BraTS-GLI-00015-000_010.png \
  --output-mask output/predicted_mask.png
```

### 2) Lote de imagens 2D

```bash
python segmentation_pipeline.py \
  --mode infer \
  --checkpoint output/unet_tumor.pth \
  --image data/val/images \
  --output-mask output/batch_masks \
  --display-count 6
```

### 3) Volume NIfTI 3D único

```bash
python segmentation_pipeline.py \
  --mode infer \
  --checkpoint output/unet_tumor.pth \
  --image SYNAPSE/BraTS-GLI-fastlane/data/BraTS-GLI-00015-000/BraTS-GLI-00015-000-t1c.nii.gz \
  --output-mask output/volume_pred.nii.gz
```

### 4) Inferência multicanal BraTS 3D

```bash
python segmentation_pipeline.py \
  --mode infer \
  --checkpoint output/unet_tumor.pth \
  --image SYNAPSE/BraTS-GLI-fastlane/data/BraTS-GLI-00015-000 \
  --output-mask output/volume_pred_multimodal.nii.gz
```

- Quando `--image` é diretório de caso BraTS, o pipeline detecta automaticamente os arquivos `t1c`, `t1n`, `t2w` e `t2f`.
- O script exibe slices e sobreposições durante a inferência de volume.

## Avaliação

```bash
python evaluate.py \
  --data-dir data \
  --split val \
  --checkpoint output/unet_tumor.pth \
  --output-report output/evaluation_report.csv
```

### Relatório gerado

- `image`
- `mask`
- `dice`
- `iou`

## Testes

```bash
pytest
```

Cobertura principal:

- conversão de BraTS para PNG
- expansão correta de canais
- predição de máscara a partir de NumPy

## Dependências chave

- `torch==2.1.0`
- `torchvision==0.16.0`
- `monai==1.2.0`
- `opencv-python==4.9.0.80`
- `numpy==1.26.0`
- `matplotlib==3.8.0`
- `imageio==2.31.1`
- `nibabel==5.0.1`
- `pytest==8.4.0`
- `tensorboard==2.15.0`
- `pandas==2.2.2`

## Dicas e observações

- Ajuste `--batch-size` e `--image-size` de acordo com a memória disponível.
- Use `--resume output/latest.pth` para continuar treino de onde parou.
- Verifique `output/train_metrics.csv` para acompanhar desempenho por época.
- Confirme `data/train` e `data/val` antes de iniciar o treino.
- O pipeline faz pós-processamento simples de máscara para reduzir ruído.

---

Este README documenta todos os modos atuais do projeto: preparo de dados, treino, inferência, avaliação e testes.
