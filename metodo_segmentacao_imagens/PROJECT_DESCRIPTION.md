# Descrição do Projeto

Este projeto é um pipeline completo de segmentação médica que transforma dados de ressonância magnética multicanais do formato BraTS (NIfTI) em um fluxo de trabalho pronto para treino, inferência e avaliação.

Ele cobre toda a jornada:

1. converte volumes BraTS em imagens 2D PNG multicanais;
2. organiza os dados em `train` e `val`;
3. treina uma U-Net 2D em PyTorch/MONAI;
4. faz inferência em imagens individuais, lotes de imagens e volumes 3D;
5. gera relatórios de avaliação com métricas Dice e IoU;
6. suporta retreinamento a partir de checkpoints e salva logs/métricas.

## O que faz

- Lê modalidades BraTS como `t1c`, `t1n`, `t2w` e `t2f` e normaliza cada fatia.
- Gera dataset 2D em `data/train` e `data/val` com imagens multicanais e máscaras binárias.
- Treina uma rede U-Net 2D configurável para segmentação binária de lesão/máscara.
- Avalia o modelo em um conjunto de validação, salvando relatório CSV com métricas por imagem.
- Faz inferência:
  - em imagem 2D única,
  - em diretório de imagens,
  - em volume NIfTI único,
  - em volume BraTS multicanal a partir de diretório de caso.
- Mantém checkpoints:
  - melhor modelo (`output/unet_tumor.pth`)
  - último checkpoint (`output/latest.pth`)
  - métricas de treino (`output/train_metrics.csv`)
  - log de treino (`output/train.log`)

## Bibliotecas utilizadas

Principais bibliotecas:

- `torch` / `torchvision`
- `monai`
- `opencv-python`
- `numpy`
- `matplotlib`
- `imageio`
- `nibabel`
- `pytest`
- `pandas`
- `tensorboard`

Essas bibliotecas permitem:

- processamento de imagens médicas,
- criação e treino de redes neurais,
- leitura/gravação de NIfTI e PNG,
- transformações de dados e visualização.

## Caminho de execução

1. Instalar dependências:
   - `pip install -r requirements.txt`
   - ou via Conda com `environment.yml`
2. Converter BraTS NIfTI para dataset 2D:
   - `python convert_brats_to_png.py --source-root SYNAPSE/BraTS-GLI-fastlane --output-root data --modalities t1c,t1n,t2f,t2w --val-ratio 0.2`
3. Treinar o modelo:
   - `python segmentation_pipeline.py --mode train --data-dir data --output-dir output --in-channels 4 --epochs 50 --batch-size 8 --image-size 128 128`
4. Fazer inferência:
   - `python segmentation_pipeline.py --mode infer --checkpoint output/unet_tumor.pth --image data/val/images/<imagem>.png --output-mask output/predicted_mask.png`
5. Avaliar:
   - `python evaluate.py --data-dir data --split val --checkpoint output/unet_tumor.pth --output-report output/evaluation_report.csv`

## Resumo para post

Este projeto demonstra um pipeline robusto de segmentação clínica que vai desde a preparação de dados BraTS até a inferência em 2D e 3D. Ele mostra como dados volumétricos médicos podem ser usados em uma arquitetura 2D multicanal, com etapas claras de pré-processamento, treino, visualização e avaliação. O pipeline é organizado para oferecer reprodutibilidade e experimentação rápida.