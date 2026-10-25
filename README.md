# 기억의 지진계 (Seismograph of Remembering)

같은 시대에 나온 두 기억 이론, Frederic Bartlett 의 『Remembering』(1932) 과 Aby Warburg 의 『므네모시네 아틀라스』(1925~1929) 를 fMRI 로 학습한 인공 뇌 위에서 마주 세우는 이중 작품의 코드다. 1부 〈뇌를 지나간 그림〉 은 Bartlett 의 연쇄 재생산을 그대로 옮겨, 그림 한 장이 인공 뇌를 거듭 지나가며 어떻게 다시 지어지는지 기록한다. 2부 〈므네모시네〉 는 그 사슬 속에서 감정이 형태보다 오래 살아남는지 재고, 그 생존 길이로 Warburg 의 패널 46 (님프) 을 다시 배열한다.

## 무엇을 계산하는가

1부의 한 단계는 다음과 같다.

```
x(t+1) = G( D( E(x(t)) + eps ) ),   eps ~ N(0, (sigma * s_v)^2)
```

E 는 그림을 nsdgeneral 복셀 반응으로 바꾸는 인코더, D 는 복셀 반응을 생성 모델의 조건으로 바꾸는 디코더, G 는 Brain-Diffuser 의 2단계 생성(VDVAE 로 저해상 복원, 이어서 Versatile Diffusion 으로 확산)이다. sigma 는 망각의 속도에 해당하는 복셀 잡음의 크기이고, s_v 는 인코더가 설명하지 못한 실제 뇌의 잔차 표준편차다. 그래서 sigma = 1 은 "인공 뇌가 실제 뇌만큼 흔들린다" 는 뜻이 된다.

같은 그림에 대해 뇌를 거치지 않는 대조 사슬도 함께 돌린다. 대조 사슬은 그림에서 같은 모델의 특징(VDVAE 잠재, CLIP 이미지 토큰)을 바로 뽑아 G 에 넣고, 뇌가 맡던 텍스트 조건 자리는 CLIP 이미지에서 CLIP 텍스트로 가는 직접 선형 사상이 채운다. 대조 사슬이 보여 주는 것은 생성 모델의 사전지식만으로 생기는 표류이고, 두 사슬의 차이가 뇌라는 병목의 몫이다. 이 대조가 있어야 "사슬이 수렴한 곳이 뇌의 스키마인가, 확산 모델의 사전지식인가" 를 가를 수 있다.

매 단계마다 원본과의 CLIP 유사도, SSIM, 화소 상관, 인공 뇌 반응의 원본 대비 상관을 기록하고, CLIP 임베딩과 복셀 반응 궤적을 저장한다. 2부는 이 궤적에 감정 모델(CLIP 임베딩에서 Cowen & Keltner 34 범주 점수로 가는 릿지)을 적용해 형태의 생존 길이와 감정의 생존 길이를 비교한다.

## 단일 모델 원칙

처음부터 끝까지 Brain-Diffuser (Ozcelik & VanRullen, 2023) 계열 하나만 쓴다. 생성은 VDVAE, CLIP ViT-L/14, Versatile Diffusion 이고, 인코더 입력과 감정 모델의 특징도 같은 VDVAE 와 같은 CLIP 에서 뽑는다. 피험자는 NSD subj01 하나다.

인코더와 디코더는 모두 선형 릿지다. 디코더 알파는 원 논문 값(VDVAE 50,000, CLIP-V 60,000, CLIP-T 100,000)을 그대로 쓰고, 인코더는 복셀마다 학습셋 안의 검증셋으로 알파를 고른다. 그러므로 뇌를 한 번 지나가는 일 자체는 복셀 공간을 지나는 선형 병목이며, 사슬의 비선형성은 생성 모델과 생성된 그림을 다시 보는 단계에서 생긴다. 이 점은 해석과 Art Paper 서술에서 분명히 밝혀 둘 필요가 있다.

원 논문은 디코더 출력을 테스트 예측 전체의 배치 통계로 재정규화했다. 사슬은 그림을 한 장씩 처리하므로, 여기서는 NSD 테스트 982장을 E, 잡음, D 순서로 통과시켜 얻은 고정 통계(보정 통계)로 대신하고, sigma 마다 따로 구한다 (`scripts/07_calibrate.py`).

## 저장소 구조

```
configs/        default.yaml (모든 경로와 설정), 그림 목록과 패널 예시
seismo/         라이브러리
  ridge.py        쌍대 형태 릿지 (조각 단위 계산, 복셀별 알파)
  nsd.py          NSD 준비 (Brain-Diffuser 와 같은 분할)
  features.py     특징 추출, 인코더 입력 블록
  brain.py        인공 뇌 (인코더, 디코더, 보정, 실행 시점 클래스)
  chain.py        1부 사슬
  emotion.py      감정 모델
  mnemosyne.py    2부 생존 길이, 패널 배열, 빈자리 채우기
  backends/       braindiffuser (실제), mock (GPU 없이 점검용)
scripts/        00 ~ 10 단계 스크립트, mock 점검
jobs/           랩서버용 실행 스크립트, Slurm 틀
tests/          단위 테스트 (릿지가 sklearn.Ridge 와 같은 해를 내는지 등)
```

## 랩서버 준비

환경은 Brain-Diffuser 저자가 공개한 conda 환경을 그대로 쓰는 것이 가장 안전하다 (Python 3.8, PyTorch 1.12.1, CUDA 11.3, transformers 4.19). Versatile Diffusion 의 옛 코드가 이 조합에서 검증되었기 때문이다. CUDA 11.3 빌드는 A100 에서는 돌지만 H100 은 지원하지 않으므로, H100 서버라면 PyTorch 와 CUDA 를 올린 환경을 따로 만들어 확인해야 한다.

```bash
git clone https://github.com/dametime14/seismograph-of-remembering.git
cd seismograph-of-remembering

# 1) 코드와 가중치 (약 8.5GB). 가중치는 큰 디스크로
WEIGHTS_DIR=/path/to/storage/weights bash scripts/00_setup_third_party.sh

# 2) 환경
conda env create -n seismograph -f third_party/brain-diffuser/environment.yml
conda activate seismograph
pip install -r requirements.txt

# 3) 설정: 경로를 서버에 맞게
cp configs/default.yaml configs/local.yaml   # paths.data_root 등을 고친다

# 4) GPU 없이 파이프라인 점검 (1~2분)
bash scripts/smoke_test_mock.sh
python -m pytest -q tests

# 5) NSD subj01 (Data Access Agreement 작성 후)
NSD_ROOT=/path/to/storage/nsd bash scripts/01_download_nsd.sh 1
```

계산 노드에 인터넷이 없다면 00 단계를 로그인 노드에서 먼저 돌려 CLIP 캐시를 만들어 두어야 한다. `HF_HOME` 을 공유 디스크로 두면 편하다.

## 실행 순서

| 단계 | 스크립트 | 장치 | 결과 |
|---|---|---|---|
| 준비 | `02_prepare_nsd.py` | CPU | 그림 단위 평균 beta, 자극(uint8), 캡션 |
| 특징 | `03_extract_features.py` | GPU | VDVAE 잠재, CLIP-V 토큰, CLIP-T 토큰, 인코더 블록 |
| 디코더 | `04_train_decoders.py` | GPU 또는 CPU | fMRI 에서 세 조건으로 가는 릿지, 대조용 사상 |
| 인코더 | `05_train_encoder.py` | GPU 또는 CPU | 그림 특징에서 복셀로 가는 릿지, 잔차 잡음 척도 |
| 점검 | `06_sanity_reconstruct.py` | GPU | 실제 fMRI 로 Brain-Diffuser 재구성 재현 |
| 보정 | `07_calibrate.py` | CPU | sigma 별 디코더 출력 통계 |
| 1부 | `08_run_chains.py` | GPU | 사슬 그림, 지표 로그, 궤적, 밀착 인화 |
| 감정 | `09_train_emotion.py` | GPU | 감정 모델 |
| 2부 | `10_mnemosyne.py` | GPU (빈자리만) | 생존표, 패널 순서, 빈자리 후보 |

모든 스크립트는 `--config configs/local.yaml` 과 `--set key=value` 덮어쓰기를 받는다. 1부부터 순서대로 돌리려면 `CONFIG=configs/local.yaml bash jobs/run_pipeline.sh` 를 쓰고, Slurm 이 있으면 `jobs/slurm_gpu.sbatch` 를 고쳐 쓴다. 사슬은 `--shard i --num-shards k` 로 나눠 여러 GPU 에 돌릴 수 있다.

06 단계는 건너뛰지 않는 편이 좋다. 실제 fMRI 로 만든 재구성이 원 논문 그림과 비슷하게 나와야 그 위에 세운 사슬을 믿을 수 있다.

## 그림 넣기

`data/artworks/` 에 그림 파일과 `artworks.csv` 를 둔다 (형식은 `configs/artworks.example.csv`). 퍼블릭 도메인 작품과, 사진 자체의 권리도 확인된 이미지만 쓴다. 정사각형이 아닌 그림은 기본적으로 가운데를 잘라 쓰며 `chain.crop: pad` 로 바꾸면 회색 여백을 둔다.

## 감정 모델 자료

라벨 표의 기본 형식은 EmoBrain 저장소의 `cowen_horikawa_labels.csv` (정수 자극 번호 `stim_num_int`, `score_0` ~ `score_33`) 와 Cowen & Keltner 영상 파일 이름(`0001.mp4` ...) 을 따른다. 이미 다른 서버에 자료가 있다면 `configs/local.yaml` 의 `emotion.labels_csv` 와 `emotion.videos_dir` 만 그쪽으로 가리키면 된다. 범주 이름은 표의 열 순서를 확인한 뒤 `emotion.category_names` 에 채운다. 영상 프레임으로 학습한 감정 모델을 정지 회화에 적용할 때의 영역 차이는 아직 열린 질문이다.

## 자원 추정 (subj01)

처리된 자료는 약 6GB (학습 자극 8,859장 uint8 약 4.8GB 포함), 특징 파일은 약 14GB, 학습된 인공 뇌는 약 32GB (디코더 약 25GB, 인코더 약 6.5GB) 다. 디코더와 인코더 학습은 RAM 32GB 이상이면 충분하도록 조각 단위로 계산한다. 사슬을 돌릴 때는 인공 뇌 행렬을 RAM 에 올려 CPU 에서 곱하는 것이 기본이고 (`brain.runtime`), GPU 에는 VDVAE 와 Versatile Diffusion 만 올라간다. 원 논문은 12GB GPU 두 장을 가정했으며 여기서는 한 장 (A100 40GB 이상) 에 모두 올린다. 실제 시간은 처음 실행할 때 재서 이 절에 적어 둘 것.

## 아직 정하지 않은 것

감정 생존의 기준값 (`mnemosyne.emotion_threshold`) 과 형태 기준값, 사슬 길이와 sigma 격자, 빈자리 후보를 고를 때의 목표 감정 (지금은 패널 그림들의 평균 감정) 은 모두 첫 결과를 본 뒤 정한다. 인코더 입력 블록 구성 (`brain.encoder.blocks`) 도 06 단계와 인코더 평가 (`models/subj01/eval/encoder.json`) 를 보고 조정할 수 있다.

## 출처와 권리

Brain-Diffuser 코드 (ozcelikfu/brain-diffuser, MIT) 는 `third_party/` 에 원본 그대로 두고 불러 쓴다. Versatile Diffusion (Xu et al., 2023), VDVAE (Child, 2021), CLIP (Radford et al., 2021) 의 가중치는 각 배포처의 조건을 따른다. NSD (Allen et al., 2022) 는 NSD Data Access Agreement 와 이용 조건을 따른다. 감정 라벨은 Cowen & Keltner (2017) 의 평정과 Horikawa et al. (2020) 의 자극 구성을 따른다.

이 저장소의 코드는 MIT 라이선스를 따른다 (`LICENSE`). NSD 자료, 사전학습 가중치, 그림 이미지는 이 라이선스에 포함되지 않으며 각자의 조건을 따른다.
