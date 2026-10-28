# E14 on Colab (L4): Paste-Ready Cells

> **Superseded (2026-09-29) by `experiments/colab_run_all.py`**: one cell that does all of this, resumes after a disconnect, and adds the M2 detector runs. Kept for reference.

E14 = E12's data plus 1,004 mined hard-negative tiles, same schedule (40 epochs max, patience 15, batch 16, seed 0, imgsz 640). It is trained **fresh** on Colab rather than resumed from the paused local run (epoch 10): the local checkpoint stores Windows data paths, and one uninterrupted run is cleaner. Record the GPU in 16. E12 was trained on the local RTX 4060, so a small E12-vs-E14 difference should be confirmed with a second seed or an E12 rerun on the same GPU.

**Before cell 1:** build the zips locally (`python experiments/pack_for_colab.py`), then upload `C:\cuas_data\colab\cuas_data_e14.zip` (1.1 GB) and `cuas_code.zip` to **`MyDrive/cuas/`** in the browser. Runtime → Change runtime type → **L4 GPU**.

### Cell 1: setup (Drive, data on local disk, runs on Drive)
```python
from google.colab import drive
drive.mount('/content/drive')
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install ultralytics==8.4.162
!mkdir -p /content/repo /content/cuas_data /content/drive/MyDrive/cuas_runs
!unzip -q -o /content/drive/MyDrive/cuas/cuas_code.zip -d /content/repo
!unzip -q -o /content/drive/MyDrive/cuas/cuas_data_e14.zip -d /content/cuas_data
!ln -sfn /content/drive/MyDrive/cuas_runs /content/cuas_data/runs     # weights survive a disconnect
!sed 's#C:/cuas_data#/content/cuas_data#g' /content/cuas_data/e14/data_e14.yaml > /content/cuas_data/e14/data_e14_colab.yaml
import os; os.environ['CUAS_DATA'] = '/content/cuas_data'; os.chdir('/content/repo')
!cat /content/cuas_data/e14/data_e14_colab.yaml | head -3
!find /content/cuas_data -name '*.jpg' | wc -l
```
**Send back:** the GPU name, the first yaml lines, and the image count (expected ~46,000).

### Cell 2: smoke test (1 epoch on 5% of the data, ~2 min)
```python
!CUAS_DATA=/content/cuas_data python experiments/train_local.py --data /content/cuas_data/e14/data_e14_colab.yaml --name E14_colab_smoke --epochs 1 --fraction 0.05 2>&1 | tr '\r' '\n' | grep -E "python |it/s|^ +all " | tail -6
```
**Send back:** the output, especially the **it/s** figure (local RTX 4060: ~3.2 it/s).

### Cell 3: full E14 run (~2-2.5 h on an L4 if the speed-up holds)
```python
!CUAS_DATA=/content/cuas_data python experiments/train_local.py --data /content/cuas_data/e14/data_e14_colab.yaml --name E14_hardneg_yolo26s_L4 2>&1 | tee /content/drive/MyDrive/cuas_runs/E14_L4_train.log | tr '\r' '\n' | grep -E "^ +all |EarlyStopping|epochs completed|Results saved"
```
**If Colab disconnects:** rerun cell 1, then:
```python
!CUAS_DATA=/content/cuas_data python experiments/train_local.py --resume /content/cuas_data/runs/E14_hardneg_yolo26s_L4/weights/last.pt 2>&1 | tee -a /content/drive/MyDrive/cuas_runs/E14_L4_train.log | tr '\r' '\n' | grep -E "^ +all |EarlyStopping|epochs completed"
```
**Send back:** the last ~10 lines (best epoch, early stop, final per-class table).

### Cell 4: evaluation (same protocol as E12, ~15 min)
```python
!CUAS_DATA=/content/cuas_data python experiments/e12_eval.py E14_hardneg_yolo26s_L4 2>&1 | tee /content/drive/MyDrive/cuas_runs/eval_E14_hardneg_yolo26s_L4.txt | tail -60
```
**Send back:** the whole printed output.

### Afterwards (local, CPU only)
Download `MyDrive/cuas_runs/E14_hardneg_yolo26s_L4/` (at least `weights/best.pt`, `e12_eval.json`, `dets_native.json`, `results.csv`, `args.yaml`) into `C:\cuas_data\runs\E14_hardneg_yolo26s_L4\`, then:
```bash
python experiments/compare_runs.py E12_multidata_yolo26s E14_hardneg_yolo26s_L4
```
