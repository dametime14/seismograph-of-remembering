#!/usr/bin/env bash
# GPU, NSD, 가중치 없이 전체 파이프라인이 끝까지 도는지 확인한다 (CPU 1~2분).
# 랩서버에서 환경을 만든 직후 한 번 돌려 볼 것.
#
#   bash scripts/smoke_test_mock.sh
set -euo pipefail
set -f   # --set 값의 대괄호가 파일 이름 패턴으로 해석되지 않도록
cd "$(dirname "$0")/.."

D=outputs/mock_data
O=outputs/mock
rm -rf "$D" "$O"
python scripts/make_synthetic_nsd.py --out "$D"

SET="backend=mock device=cpu brain.solver_device=cpu paths.data_root=$D paths.outputs=$O \
chain.n_iters=3 chain.n_replicates=1 chain.sigmas=[0.0,0.5] features.batch_size=16 \
brain.encoder.clip_patch_pool_grid=2 mnemosyne.panel=$D/panel.yaml \
mnemosyne.vacancy.n_candidates=4 mnemosyne.vacancy.keep_top=2 \
emotion.alphas=[0.1,1.0,10.0] brain.encoder.alphas=[1.0,100.0,10000.0] \
brain.control_text_alphas=[0.1,10.0,1000.0]"

python scripts/03_extract_features.py --set $SET
python scripts/04_train_decoders.py   --set $SET
python scripts/05_train_encoder.py    --set $SET
python scripts/07_calibrate.py        --set $SET
python scripts/08_run_chains.py       --set $SET
python scripts/09_train_emotion.py    --set $SET --skip-embed
python scripts/10_mnemosyne.py        --set $SET --vacancy

echo
echo "[smoke] OK. 결과: $O/chains/*/contact_sheet.png, $O/chains/survival.csv"
echo "[smoke] mock 자료는 작고 알파 격자를 일부러 좁혀 두어서, 알파 경고가 나와도 정상이다."
