# STRATA-final — 실험 재현 상세 가이드

이 문서는 `STRATA-final.zip` 패키지로 STRATA 논문의 **시간 도달성(temporal reachability)**
실험 결과를 **처음부터** 재현하는 절차와, **대규모 공개 그래프를 직접 내려받아 로컬에서
실행**하는 절차를 단계별로 설명합니다.

> 패키지 구성 개요
> - **원본 STRATA (정적 APSP)** — `02_Implementations/`, `02_Implementation_BRANDES/`,
>   `03_Experiments/`, `docs/` : 초기 공유 저장소(https://dknife.github.io/STRATA/)의 코드.
>   이 가이드의 대상이 아니며, 해당 디렉토리의 자체 문서(`docs/howto.html`)를 참고하십시오.
> - **시간 도달성 확장** — `05_Temporal_STRATA/` : 본 가이드가 재현하는 논문 실험의 코드.
> - **데이터** — `05_Temporal_STRATA/datasets_real_temporal/static/facebook_combined.txt` (실제 SNAP ego-Facebook; STRATA-v19부터, 레거시 `04_Datasets/real-world/` 파일의 사본).

표기: 셸 명령은 `$`로 시작합니다. 별도 언급이 없으면 모든 명령은 **`05_Temporal_STRATA/`
디렉토리 안에서** 실행합니다.

---

## 0. 빠른 요약 (TL;DR)

```bash
# (1) 압축 해제 후 시간 도달성 확장 디렉토리로 이동
unzip STRATA-final.zip
cd STRATA-final/05_Temporal_STRATA

# (2) 가상환경 + 의존성 (pyenv + virtualenv — 자세한 절차는 2.2 참조)
pyenv install 3.12.3 && pyenv local 3.12.3
pip install --upgrade pip virtualenv
virtualenv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt          # CPU/맥
# 컴파일 커널까지: pip install -r requirements-ubuntu-cuda.txt
# GraphBLAS 비교까지: pip install python-graphblas

# (3) 정확성 자가 검증 (가장 먼저)
python3 -m pytest tests/ -q

# (4) 핵심 실험 재현 (논문 §6)
python3 experiments/make_real_temporal.py                                   # 실제 그래프 시간화
python3 experiments/ablation_real_temporal.py                               # 실제 ego-Facebook (§6.5)
python3 experiments/baseline_ablation.py --edges /tmp/fb_full_temporal.txt --mode strict \
        --out results/ablation/baseline_ablation.csv                        # 가속 분해 (§6.5)
python3 experiments/ablation_graphblas.py                                   # GraphBLAS 공정 비교 (§6.5)
python3 experiments/ablation_deletion.py                                    # 삭제 수리 (§6.4)
python3 experiments/ablation_throughput_curve.py                            # 처리량 1/W (§6.6)
python3 experiments/hit_prioritization.py --synthetic --out-dir results/hitsel  # 사례 연구 (§6.7)

# (5) v21 신규 — 실제 타임스탬프 평가 (§"실제 타임스탬프 데이터셋")
#     실제 SNAP 시계열 그래프 2종이 datasets_real_temporal/ 에 동봉됨.
#     표준화 후 정확 검증 전체 쌍 벤치마크 실행:
python3 experiments/real_temporal_benchmark.py        # 시간 대 정적 + 가속, 실제 3개
python3 experiments/real_dynamic_and_blocking.py      # 실제 삽입/윈도/삭제 + 블로킹
python3 experiments/numba_kernel.py --edges datasets_real_temporal/standard/CollegeMsg.txt \
       --out results/real/numba_collegemsg.csv        # 컴파일 커널, 실제 데이터 비트 검증
#     제약 없는 머신에서 실제 >=10^7 간선 점 재현:
python3 experiments/fetch_real_temporal.py --which large   # sx-stackoverflow(~63M 간선) 등 다운로드

# (6) v22 — 헤드라인 표를 현재 코어로 재생성 + strict 프루닝 절제
python3 experiments/perf_table.py                     # table_honest_perf.tex 재작성(양쪽 LaTeX 트리)
python3 experiments/ablation_strict_prune.py          # strict np.any 검사는 순수 오버헤드(2.3-2.5x)

# (7) v23 — 컴파일-대-컴파일, 인덱스 베이스라인, 통합 메인 표
python3 experiments/compiled_baselines.py             # STRATA-numba vs 컴파일 per-source BFS (기하평균 4.36x); table_maincompare.tex 재작성
python3 experiments/index_crossover.py                # STRATA 인덱스(구축+O(1)) vs 온디맨드 BFS; 크로스오버 ~2 소스

# (8) v24 — 스냅숏 필요성(C1 뒷받침)
python3 experiments/ablation_snapshot_necessity.py    # no-snapshot 순서 의존 + strict 과다계상 최대 17.7%

# (9) v25 — 실제 데이터 AML/출처 사례(합성 드럭 사례 대체)
python3 experiments/aml_tracing.py                    # Bitcoin-OTC: 허위경로 44.8%, 상위20 중첩 0
```

생성된 수치는 `results/ablation/*.csv`에 기록되며 논문 본문과 대조합니다.

---

## 1. 사전 요구 사항

| 항목 | 권장 |
|------|------|
| 운영체제 | macOS 13+ (Apple Silicon 포함) 또는 Ubuntu 22.04+ |
| Python | 3.10 ~ 3.12 |
| 메모리 | 16GB 이상 (실제 그래프 n=4039 전체 쌍에 충분) |
| 디스크 | 2GB (코드+결과). 대규모 그래프 추가 시 +수 GB |
| 네트워크 | (선택) 대규모 공개 데이터 다운로드 시에만 필요 |

핵심 의존성은 `numpy`, `pandas`, `scipy`, `matplotlib`, `pytest`입니다(순수 CPU 재현).
GraphBLAS 비교에는 `python-graphblas`, 컴파일 커널·처리량 곡선에는 `numba`가 추가로 필요합니다.

---

## 2. 1단계 — 설치

### 2.1 압축 해제와 디렉토리 구조

```bash
$ unzip STRATA-final.zip
$ cd STRATA-final/05_Temporal_STRATA
```

`05_Temporal_STRATA/`가 **시간 도달성 실험의 작업 디렉토리**입니다.

```
STRATA-final/
├── 02_Implementations/         # (원본) 정적 APSP 12개 방법 — 본 가이드 대상 아님
├── 02_Implementation_BRANDES/  # (원본) Matrix Brandes 중심성
├── 03_Experiments/             # (원본) 정적 실험
├── docs/                       # (원본) dknife.github.io 사이트
├── 04_Datasets/
│   └── real-world/facebook_combined.txt   # 실제 SNAP ego-Facebook (n=4039)
└── 05_Temporal_STRATA/         # ← 본 가이드의 재현 대상
    ├── temporal_strata/          #   핵심 라이브러리
    │   ├── packed_reachability.py#     temporal_strata_packed() 본체
    │   ├── io.py                 #     표준 간선 파일 로더
    │   ├── baseline.py           #     정확 시간 BFS 기준
    │   └── ...
    ├── experiments/              #   실험별 재현 스크립트
    ├── tests/                    #   정확성 자가 검증 (pytest)
    ├── requirements-mac.txt
    └── requirements-ubuntu-cuda.txt
```

실제 데이터(`04_Datasets/`)는 `05_Temporal_STRATA/`의 **한 단계 위**에 있으며, 스크립트는
상대경로(`../04_Datasets/...`)로 자동 참조합니다.

### 2.2 가상환경과 의존성 (pyenv + virtualenv)

재현 환경의 Python 버전을 고정하기 위해 `pyenv`로 Python을 설치하고, `virtualenv`로
격리된 환경을 만듭니다. (시스템 Python을 건드리지 않아 안전하며, 버전 불일치로 인한
재현 실패를 예방합니다.)

#### (1) pyenv 설치

```bash
# macOS (Homebrew)
$ brew install pyenv

# Ubuntu/Debian — 빌드 의존성 먼저 설치
$ sudo apt-get update && sudo apt-get install -y \
    make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev \
    libxmlsec1-dev libffi-dev liblzma-dev
$ curl -fsSL https://pyenv.run | bash
```

설치 후, 셸 설정에 pyenv를 추가하고 셸을 다시 엽니다(또는 `source`).

```bash
# ~/.bashrc (또는 ~/.zshrc) 에 추가
$ echo 'export PYENV_ROOT="$HOME/.pyenv"'              >> ~/.bashrc
$ echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
$ echo 'eval "$(pyenv init -)"'                        >> ~/.bashrc
$ exec "$SHELL"                                        # 셸 재시작
$ pyenv --version                                      # 설치 확인
```

#### (2) Python 설치 및 로컬 버전 고정

```bash
$ pyenv install 3.12.3          # 재현 권장 버전 (3.10~3.12 호환)
# 이 디렉토리에서 항상 3.12.3을 쓰도록 .python-version 고정
$ cd STRATA-final/05_Temporal_STRATA
$ pyenv local 3.12.3
$ python --version              # Python 3.12.3 확인
```

#### (3) virtualenv로 격리 환경 생성

```bash
# virtualenv 설치 (pyenv가 고른 Python에 설치됨)
$ pip install --upgrade pip virtualenv

# 프로젝트 전용 가상환경 생성 & 활성화
$ virtualenv .venv
$ source .venv/bin/activate          # Windows: .venv\Scripts\activate
$ python --version                   # 가상환경의 Python 3.12.3 확인
```

> 참고: `pyenv-virtualenv` 플러그인을 쓰면 다음과 같이 한 번에 만들 수도 있습니다.
> ```bash
> $ pyenv virtualenv 3.12.3 strata-final
> $ pyenv local strata-final          # 디렉토리 진입 시 자동 활성화
> ```

#### (4) 의존성 설치

```bash
$ pip install --upgrade pip
$ pip install -r requirements-mac.txt          # 맥/CPU 기본
$ pip install -r requirements-ubuntu-cuda.txt  # (선택) numba 포함 — 컴파일 커널/처리량
$ pip install python-graphblas                 # (선택) GraphBLAS 공정 비교용
```

작업이 끝나면 `deactivate`로 가상환경을 빠져나옵니다.

> `python-graphblas`는 SuiteSparse:GraphBLAS 바이너리를 함께 설치합니다. 설치가 어려우면
> GraphBLAS 비교(7단계)만 건너뛰면 나머지는 모두 재현됩니다.

---

## 3. 2단계 — 정확성 자가 검증 (가장 먼저)

STRATA의 모든 성능 수치는 **"정확 시간 BFS 기준과 비트 단위로 동일"**하다는 전제 위에 있습니다.

```bash
$ python3 -m pytest tests/ -q
```

검증 내용:
- `test_packed_reachability.py` — STRATA 묶음 결과가 정확 BFS 기준과 **도달 쌍 수·전체 비트셋**에서 일치 (엄격/폐포)
- `test_temporal_strata.py` — 시각 순서·동시각 의미론(strict vs closure) 경계 사례
- `test_original_baselines.py` — 베이스라인 구현 정합성

> 라이브러리만 빠르게 확인:
> ```bash
> $ python3 -c "
> from temporal_strata.packed_reachability import temporal_strata_packed, reachable_pairs
> edges=[(1,0,1),(2,1,2),(3,2,3)]   # (t,u,v)
> R=temporal_strata_packed(4, edges, same_time_mode='strict')
> print('reachable pairs =', reachable_pairs(R,4))
> "
> ```

---

## 4. 3단계 — 표준 입력 형식

라이브러리 로더(`temporal_strata.io.load_standard_edges`)와 대부분의 스크립트는
**표준 시간 간선 형식**을 읽습니다. 한 줄에 공백으로 구분된 세 정수:

```
src dst timestamp
```

- `src`, `dst`: 0부터 시작하는 정점 ID
- `timestamp`: 정수 시각 표식 (작을수록 이른 시각)
- `#`로 시작하는 줄은 주석. `# meta: {"n": 4039}` 형태면 JSON 메타데이터로 해석
- gzip(`.gz`)도 직접 로드

```python
from temporal_strata.io import load_standard_edges
n, edges, meta = load_standard_edges("graph.edges")   # edges: [(t,u,v), ...]
```

> **주의 — 스크립트별 입력 인자 방식이 두 가지입니다.**
> - `load_standard_edges`를 쓰는 스크립트(`ablation_real_temporal.py` 등)는 위 형식을 읽습니다.
> - 일부 절제 스크립트(`ablation_closure_detector.py`, `baseline_ablation.py`)는
>   명령행 인자로 경로를 받습니다. 각 단계의 명령을 그대로 따르면 됩니다.

---

## 5. 4단계 — 실제 그래프(ego-Facebook n=4039) 재현 (§6.5)

논문 §6.5의 "실제 위상" 결과부터 재현합니다(가장 대표적이고 빠름).

### 5.1 시간 스트림 생성

```bash
$ python3 experiments/make_real_temporal.py
# → /tmp/fb_full_temporal.txt 생성 (n=4039, 88,234 간선, 발견 순서로 200 그룹)
```

> **정직성 주의:** 이 그래프의 *위상*은 실제(SNAP ego-Facebook)지만, *타임스탬프*는
> BFS 발견 순서로 부여한 **제어된** 값입니다("실제 토폴로지 + 제어된 시각"). 진짜 시간
> 동역학(원본 타임스탬프) 데이터는 부록 A에서 직접 내려받아 실행합니다.

### 5.2 블로킹 스케일링 측정

```bash
$ python3 experiments/ablation_real_temporal.py
```
**대조 (논문 §6.5):**

| β | 패스 | 패스당 워드 | 도달 쌍 |
|---|------|-----------|--------|
| 4039 | 1 | 258,496 | 1,535,364 |
| 2019 | 3 | 129,248 | 1,535,364 |
| 1009 | 5 | 64,624 | 1,535,364 |
| 504 | 9 | 32,312 | 1,535,364 |

모든 β에서 도달 쌍이 **1,535,364로 동일**해야 합니다(블로킹은 정확성 불변).
출력: `results/ablation/real_temporal.csv`. (절대 실행 시간은 하드웨어마다 다릅니다.)

---

## 6. 5단계 — 가속 분해 (§6.5)

무엇이 가속을 만드는지 4개 구현을 같은 그래프에서 비교합니다.

```bash
$ python3 experiments/baseline_ablation.py \
    --edges /tmp/fb_full_temporal.txt --mode strict \
    --out results/ablation/baseline_ablation.csv
```
**대조 (실제 ego-Facebook 예시):**

```
exact_bfs_ref       pairs=1535364   (가장 느림, 정확 BFS 기준)
numpy_dense_group   pairs=1535364
strata_packed       pairs=1535364   (워드 패킹)
bitset_ms_no_prune  pairs=1535364   (가지치기 없는 비트셋 — 가장 빠름)
ALL METHODS AGREE: True
```
4개 구현이 **모두 1,535,364쌍으로 동일**해야 합니다(워드 패킹이 지배적 가속, 가지치기는
strict에서 무용함을 보임). `--mode closure`로 폐포 모드도 측정 가능합니다.

---

## 7. 6단계 — GraphBLAS 공정 비교 (§6.5)

```bash
$ pip install python-graphblas        # 아직 설치 안 했다면
$ python3 experiments/ablation_graphblas.py
```

웜업 + best-of-3를 STRATA와 GraphBLAS **양쪽에 동일하게** 적용하고, GraphBLAS 폐포를
마스크 mxm으로 계산합니다.

**대조 (논문 §6.5):**

| n | 도달 쌍 | 비율(STRATA/GraphBLAS) |
|---|--------|----------------------|
| 256 | 5,870 | 2.33× |
| 512 | 30,501 | 2.78× |
| 1024 | 93,715 | 2.02× |
| 2048 | 744,796 | 0.93× |
| 4039 | 2,512,141 | 0.66× |

모든 n에서 도달 쌍이 **정확히 일치**해야 하고, 비율은 작은 그래프에서 2–2.8×(느림),
n=2048 동급, n=4039에서 0.66×(빠름)의 추세를 보입니다. 출력:
`results/ablation/graphblas_static.csv`.

---

## 8. 7단계 — 그 외 실험

각 스크립트의 입력 방식에 유의하십시오.

### 8.1 삭제 수리 비용 (§6.4)
```bash
$ python3 experiments/ablation_deletion.py        # 기본값 /tmp/fb_full_temporal.txt 사용
```
**대조:** 접미 1k 간선당 ≈0.008초(상수), 초기/중간/후기 삭제가 접미 크기에 비례.

### 8.2 처리량 1/W 곡선 (§6.6) — numba 필요
```bash
$ python3 experiments/ablation_throughput_curve.py
```
**대조:** n=2k/4k/8k/16k에서 27/5.3/1.5/0.43 M edges/s (~1/W).

### 8.3 비자명성 — naive 베이스라인 대비 (§2.2/§6.5)
```bash
$ python3 experiments/ablation_naive_temporal.py
```
**대조:** 16–28× (비트 동일).

### 8.4 블로킹 스케일링 (합성, §6.5)
```bash
$ python3 experiments/ablation_blocking.py
```
**대조:** β를 줄이면 패스↑·패스당 워드↓, 모든 β에서 도달 쌍 동일.

### 8.5 폐포 검출기 (§6.5) — 입력 그래프를 인자로
```bash
# 그룹 구조가 뚜렷한 스트림을 인자로 전달 (u v t 형식)
$ python3 experiments/ablation_closure_detector.py <graph1.txt> [graph2.txt ...]
```
**대조:** 변화 검출 폐포가 무조건 스윕 대비 23–290× 빠름(비트 동일). 입력이 없으면 빈
결과가 되므로 반드시 그래프 경로를 전달하십시오.

### 8.6 사례 연구 — 전향적 hit 우선순위화 (§6.7)
```bash
$ python3 experiments/hit_prioritization.py --synthetic --out-dir results/hitsel
```
**대조:** EF strict 6.1 > closure 4.8 ≫ 인기도 0.9; static 상위 hit의 100%가 누설.

---

## 9. 결과 대조 체크리스트

1. **정확성:** `pytest tests/` 전부 통과 → STRATA가 정확 BFS와 비트 동일
2. **쌍 수 일치:** 각 실험의 "도달 쌍" 수가 위 표와 동일 (하드웨어 무관)
3. **추세 일치:** 가속/비율의 *방향과 크기 대역*이 일치 (절대 초는 기기마다 다름)
4. **CSV 생성:** `results/ablation/`에 해당 `*.csv`가 기록됨

> 논문은 **동일 기계 상대 비교**와 **비트 단위 쌍 수 일치**를 주장합니다. 재현 판정도 이
> 두 가지를 기준으로 하십시오(절대 실행 시간은 CPU·메모리 대역폭에 따라 달라집니다).

---

# 부록 A. 대규모 공개 그래프 직접 다운로드 & 로컬 실행

**진짜 시간 동역학(원본 타임스탬프)을 가진 대규모 공개 그래프**를 직접 내려받아 STRATA로
실행하는 절차입니다(5단계의 제어된 타임스탬프와 다른, 실제 동역학 데이터).

## A.1 권장 데이터셋

| 데이터셋 | 규모(대략) | 출처 | 형식 |
|---------|-----------|------|------|
| **CollegeMsg** | 1,899 정점 / 59,835 간선 | SNAP | `src dst unixts` |
| **email-Eu-core-temporal** | 986 / 332,334 | SNAP | `src dst time` |
| **sx-stackoverflow** (대규모) | 2.6M / 63M | SNAP | `src dst time` |
| **tgbl-wiki-v2** | ~10⁵ / ~10⁶ | TGB2 | py-tgb API |

## A.2 경로 1 — SNAP 그래프 (가장 간단, 권장)

SNAP 시간 그래프는 `src dst timestamp` 3열이라 STRATA 표준 형식과 거의 동일합니다.

### (1) 다운로드
```bash
$ cd STRATA-final/05_Temporal_STRATA
$ mkdir -p data/real

$ curl -L -o data/real/CollegeMsg.txt.gz \
    https://snap.stanford.edu/data/CollegeMsg.txt.gz
$ gunzip -f data/real/CollegeMsg.txt.gz
```
> 방화벽으로 막히면 브라우저로 https://snap.stanford.edu/data/ 에서 받아 `data/real/`에 두고 `gunzip`.

### (2) 표준 형식으로 정규화
SNAP CollegeMsg 한 줄은 `SRC DST UNIXTS`입니다. 정점을 0부터 조밀화하고 타임스탬프를 정렬:

```bash
$ cat > to_standard.py << 'EOF'
import sys
src_path, out_path = sys.argv[1], sys.argv[2]
edges = []; nodes = {}
def nid(x):
    if x not in nodes: nodes[x] = len(nodes)
    return nodes[x]
for line in open(src_path):
    line = line.strip()
    if not line or line.startswith("#"): continue
    a = line.replace(",", " ").split()
    u, v, t = int(a[0]), int(a[1]), int(float(a[2]))
    edges.append((t, nid(u), nid(v)))
edges.sort()
times = sorted(set(t for t,_,_ in edges))
tmap = {t:i for i,t in enumerate(times)}
with open(out_path, "w") as f:
    f.write(f'# meta: {{"n": {len(nodes)}}}\n')
    for t,u,v in edges:
        f.write(f"{u} {v} {tmap[t]}\n")
print(f"n={len(nodes)} edges={len(edges)} groups={len(times)} -> {out_path}")
EOF

$ python3 to_standard.py data/real/CollegeMsg.txt data/real/collegemsg.edges
```

### (3) STRATA로 실행
표준 형식이므로 실제 그래프 스크립트에 경로 인자를 전달하거나, 라이브러리를 직접 호출:

```bash
# 블로킹 스케일링 스크립트 (첫 인자로 경로)
$ python3 experiments/ablation_real_temporal.py data/real/collegemsg.edges

# 또는 라이브러리 직접 호출
$ python3 << 'EOF'
import time, math
from temporal_strata.io import load_standard_edges
from temporal_strata.packed_reachability import (
    temporal_strata_packed, reachable_pairs, words_for)

n, edges, meta = load_standard_edges("data/real/collegemsg.edges")
print(f"loaded n={n} edges={len(edges)}")

t0 = time.perf_counter()
R = temporal_strata_packed(n, edges, same_time_mode="strict")
dt = time.perf_counter() - t0
print(f"strict: pairs={reachable_pairs(R,n)}  time={dt:.3f}s  words/row={words_for(n)}")

beta = max(64, n // 4)
t0 = time.perf_counter()
Rb = temporal_strata_packed(n, edges, same_time_mode="strict", block_bits=beta)
dt = time.perf_counter() - t0
assert reachable_pairs(Rb, n) == reachable_pairs(R, n)   # 블로킹은 정확성 불변
print(f"blocked(beta={beta}, passes={math.ceil(n/beta)}): time={dt:.3f}s  (pairs match: OK)")

Rc = temporal_strata_packed(n, edges, same_time_mode="closure")
print(f"closure: pairs={reachable_pairs(Rc,n)}")
EOF
```

이로써 **실제 타임스탬프를 가진 공개 그래프**에서 정확성(블로킹 불변)과 성능(단일 패스 vs
블록)을 직접 확인할 수 있습니다.

## A.3 경로 2 — TGB2 (tgbl-wiki-v2)

### (1) py-tgb 설치 및 캐시
```bash
$ pip install py-tgb
$ python3 -c "
from tgb.linkproppred.dataset import LinkPropPredDataset
LinkPropPredDataset(name='tgbl-wiki-v2', root='data/raw/tgb')
"   # 최초 1회 다운로드(캐시)
```

### (2) 표준 형식으로 내보내기 (동봉 스크립트)
```bash
$ python3 experiments/export_tgb_wiki.py \
    --root data/raw/tgb \
    --out  data/standard/tgbl-wiki.edges
# (선택) 규모 제한: --max-nodes 20000 --max-edges 1000000
```

### (3) 실행 (정점이 많으면 블로킹 권장)
```bash
$ python3 << 'EOF'
import time, math
from temporal_strata.io import load_standard_edges
from temporal_strata.packed_reachability import temporal_strata_packed, reachable_pairs, packed_state_bytes

n, edges, meta = load_standard_edges("data/standard/tgbl-wiki.edges")
print(f"n={n} edges={len(edges)}  full-state≈{packed_state_bytes(n)/1e6:.1f} MB")

beta = 2048
t0 = time.perf_counter()
R = temporal_strata_packed(n, edges, same_time_mode="strict", block_bits=beta)
dt = time.perf_counter() - t0
print(f"blocked(beta={beta}, passes={math.ceil(n/beta)}): pairs={reachable_pairs(R,n)} time={dt:.2f}s")
EOF
```

> 메모리 가늠: 전체 상태는 `n·⌈n/64⌉·8` 바이트(`packed_state_bytes(n)`로 확인).
> 예: n=10만이면 ~1.25GB. 블로킹 β로 패스당 점유를 `n·⌈β/64⌉·8`로 제한하십시오.

## A.4 대규모 실행 팁

- **메모리 우선:** `packed_state_bytes(n)`로 전체 상태 크기 확인 → 예산 초과 시 `block_bits=β` 설정.
  패스당 점유 ∝ β, 패스 수 = ⌈n/β⌉.
- **정확성 점검:** 작은 β와 큰 β의 `reachable_pairs`가 같은지 항상 대조(블로킹 불변).
- **컴파일 가속:** `numba` 설치 시 `experiments/numba_kernel.py`의 커널로 큰 스트림을 가속.
- **공정 측정:** 시간 보고 시 **웜업 후 best-of-3** 사용(논문 전역 규율). 단발 측정은
  JIT·할당 오버헤드가 섞여 왜곡됩니다.
- **윈도 분석:** 회전 윈도가 필요하면 간선 리스트를 시각 구간으로 슬라이스해 입력.

---

## 문의 / 재현 이슈

- 정확성 테스트 실패: Python·numpy 버전을 `requirements`에 맞추고 재설치 후 재시도.
- GraphBLAS 설치 실패: 6단계만 건너뛰면 나머지 전부 재현됩니다.
- 대규모 OOM: `block_bits`를 더 작게(예: 1024→512) 설정해 패스당 메모리를 낮추십시오.
- 빈 결과 CSV: `ablation_closure_detector.py`는 입력 그래프 경로를 **인자로** 받습니다.


## 9. 멀티코어 병렬 검증과 강스케일링 (pre-v1 이력에서 추가)

열 분할(column-partitioned) 스레드 병렬 코어(`temporal_strata/parallel.py`)는 어떤 스레드 수·청크 분할·인터리빙에서도 순차 코어와 비트 동일함이 증명되어 있고(논문 Theorem 3), 테스트로도 검증된다.

```bash
# 병렬 포함 전체 테스트 (STRATA-v36부터 331개; torch/PyG 부재 시 사유 명시 스킵)
python3 -m pytest tests/ -q

# 강스케일링 하네스: 커널/엔드투엔드 분리, 병렬 per-source BFS 대비, 표 자동 방출
NUMBA_NUM_THREADS=8 python3 experiments/parallel_scaling.py
```

하네스 규칙: (1) 타이밍 전에 모든 병렬 구성을 순차 결과와 비트 대조하고 불일치 시 중단한다. (2) 단일 코어 호스트에서는 speedup을 출력하지 않고 PENDING 표를 방출한다. (3) 결과는 `results/real/parallel_scaling.json`과 `table_parallel_scaling.tex`로 기록되며 머신 문자열이 캡션에 포함된다. 산출된 표를 `STRATA-latex/STRATA-paper-EN/table_parallel_scaling.tex`에 복사하고 main.tex의 두 `%% TODO(M5)` 문장을 채운 뒤 재빌드한다(14쪽 이하 확인).

## 10. Apple M5 Max 실측 세션 (STRATA-v4에서 1회 실행 완료, 재실측은 10.1(v5) / 10.2(v6) / 10.3(v7), 논문 수치는 10.4(v8), v9 재측정은 10.5)

기계: Apple M5 Max, super 코어 6개 + performance 코어 12개, 128 GB. 다른 앱을
닫고 전원을 연결한 상태에서 `STRATA-v7/STRATA-code/05_Temporal_STRATA`에서 모든
단계를 한 세션에 실행한다. 모든 스크립트는 타이밍 전에 비트 동일을 검증하고,
기계 정보(`hw.perflevel*` 코어 수, 감지된 캐시라인 크기 포함)를 JSON에 기록한다.
macOS는 스레드가 어느 코어에서 도는지 보장하지 않으므로 코어 고정이나 배치 해석은
하지 않는다.


**명령 하나로 실행 (백로그 R15).** `python3 tools/run_m5_session.py`(또는
`tools/run_m5_session.sh`)가 아래 명령을 같은 순서로 그대로 실행하고, 명령마다
로그를 `results/real/logs/`에 남기며, 기존 결과 파일은 덮어쓰지 않고 시각을 붙인
이름으로 보존한다. 1단계 검사가 실패하거나 계획 확인에서 놀게 되는 워커가 보이면
측정 전에 멈춘다. `--dry-run`으로 계획을 미리 보고, `--from-step N` /
`--only-step N`으로 중단된 곳부터 다시 실행한다. 가상환경은 먼저 만들고 활성화한다
(1단계의 처음 두 줄).

**1단계 — 환경, 정확성, 실행 계획 확인 (기대: 331 passed)**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt -r requirements-optional.txt
python3 -m pytest tests/ -q
NUMBA_NUM_THREADS=18 python3 -m pytest tests/test_parallel.py tests/test_parallel_layout.py -q
python3 tools/check_numbers.py
NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py --plan-only \
  --threads 1,2,4,6,8,12,16,18 --layouts rowmajor,tiled,auto --schedules static,dynamic \
  --chunk-policy threads --large 50000:1000000:400 --large 100000:2000000:400 \
  --block-bits 65536 --plan-out results/real/plan_check.json
```
계획 출력에서 p ≤ W인 실데이터에 `<-- workers idle`이 보이면 멈추고(R8),
측정 대신 그 출력을 전달한다.

**2단계 — 강스케일링: 메모리 배치 × 청크 분배 (백로그 B, R4, R6, R8)**
```bash
NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py \
  --threads 1,2,4,6,8,12,16,18 --repeats 5 \
  --layouts rowmajor,tiled,auto --schedules static,dynamic --oversub 4 \
  --chunk-policy threads --table-config tiled:static \
  --out results/real/parallel_scaling.json \
  --tex-out results/real/table_parallel_scaling.tex
```

**3단계 — 큰 그래프와 순차 비율 (R12)**
```bash
NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py \
  --skip-real --skip-synthetic \
  --large 50000:1000000:400 --large 100000:2000000:400 --block-bits 65536 \
  --reference compiled --baseline-max-work 1e10 \
  --threads 1,2,4,6,8,12,16,18 --repeats 3 \
  --layouts rowmajor,tiled,auto --schedules static,dynamic --chunk-policy threads \
  --out results/real/parallel_scaling_large.json \
  --tex-out results/real/table_parallel_scaling_large.tex
```

**4단계 — 메모리 대역폭 상한 (R5)**
```bash
NUMBA_NUM_THREADS=18 python3 experiments/bandwidth_roofline.py \
  --threads 1,2,4,6,8,12,16,18 --array-mib 2048 --repeats 5 \
  --layouts rowmajor,tiled --chunk-policy threads --synthetic \
  --large 100000:2000000:400 \
  --out results/real/bandwidth_roofline.json
```

**5단계 — `layout="auto"` 규칙의 row-major / tiled 교차점 (R6)**
```bash
NUMBA_NUM_THREADS=18 python3 experiments/layout_crossover.py \
  --threads 18 --points 9 --repeats 5 --out results/real/layout_crossover_p18.json
NUMBA_NUM_THREADS=18 python3 experiments/layout_crossover.py \
  --threads 6 --points 9 --repeats 5 --out results/real/layout_crossover_p6.json
```

**6단계 — 실데이터 컴파일 대 컴파일과 외부 기준선 (R9, R13)**
```bash
python3 experiments/compiled_real.py --repeats 5 --out results/real/compiled_real.json
pip install raphtory
python3 experiments/external_baselines.py --adapter raphtory --check-only --diagnose \
  --out results/real/external_check.json
# 의미론이 일치한다고 나온 경우에만:
python3 experiments/external_baselines.py --adapter raphtory \
  --datasets CollegeMsg,BitcoinOTC,BitcoinOTCpos --sources all --repeats 3 \
  --out results/real/external_baselines.json
```
Raphtory 어댑터가 `ADAPTER UNAVAILABLE`이나 의미론 불일치를 보고하면 콘솔 출력을
그대로 두고, 세션 중에 어댑터를 고치지 않는다.

**7단계 — MPS 프로브와 미리보기 그림**
```bash
python3 experiments/backend_bench.py --device mps
python3 experiments/plot_parallel.py scaling \
  --json results/real/parallel_scaling.json --out results/real/fig_parallel_scaling.pdf
python3 experiments/plot_parallel.py scaling \
  --json results/real/parallel_scaling_large.json --out results/real/fig_parallel_scaling_large.pdf
python3 experiments/plot_parallel.py roofline \
  --json results/real/bandwidth_roofline.json --out results/real/fig_roofline.pdf
```

`results/real/` 폴더 전체(plan_check, parallel_scaling, parallel_scaling_large,
bandwidth_roofline, layout_crossover_p18 / _p6, compiled_real,
external_check / external_baselines, backend_bench JSON, 방출된 표, 미리보기 PDF)와
콘솔 로그를 전달한다. 다음 원고 버전의 성능 수치는 이 파일들에서만 가져오며, 원고에 손으로
옮겨 적지 않는다.

### 10.1 STRATA-v5 재실측 (백로그 R17, R13; 2026-11-15 완료)

첫 세션(STRATA-v4, 2026-11-15)은 끝났다. STRATA-v5는 배열 기반 전처리 경로(R17)를
추가하고 기본 배치를 tiled로 바꿨으므로 1 / 2 / 3 / 6 / 7단계를 다시 실행한다.
4 / 5단계(커널 대역폭, 배치 교차점)는 전처리와 무관하므로 반복하지 않는다. 기존
결과 파일은 시각을 붙인 이름으로 자동 보존된다.

**A. 주 환경 (첫 세션과 같은 Python)**
```bash
cd ~/Research/STRATA/STRATA-v5/STRATA-code/05_Temporal_STRATA
python3 -m venv .venv && source .venv/bin/activate
python3 tools/run_m5_session.py --dry-run --steps 1,2,3,6,7
python3 tools/run_m5_session.py --steps 1,2,3,6,7
deactivate
```

**B. Python 3.13 환경에서 외부 기준선 (R13)**
첫 세션에서는 Python 3.15용 Raphtory 배포 파일이 없었다. Python 3.13 가상환경에서
6단계의 외부 기준선 부분만 실행한다:
```bash
cd ~/Research/STRATA/STRATA-v5/STRATA-code/05_Temporal_STRATA
python3.13 -m venv .venv313 && source .venv313/bin/activate
pip install -r requirements-mac.txt
python3 tools/run_m5_session.py --only-external
deactivate
```
`python3.13`이 없으면 먼저 설치한다(예: `brew install python@3.13` 또는 python.org
설치 파일). 그래도 어댑터가 `ADAPTER UNAVAILABLE`이나 의미론 불일치를 보고하면
로그를 그대로 두고, 세션 중에 어댑터를 고치지 않는다.

### 10.2 STRATA-v6 재실측 (백로그 R19, R13 추가; 2026-11-15 완료)

STRATA-v6는 터치된 행 계획의 전체 정렬을 선형 탐색으로 바꾸고(R19; 전처리는 이제
워밍업 + best-of-R로 재며, JIT 컴파일이 들어가는 첫 호출 시간은 따로 기록한다),
외부 기준선 진단(`--diagnose`, R13 추가)을 더했다. 10.1과 같은 두 부분을 v6
폴더에서 실행한다:

**A. 주 환경**
```bash
cd ~/Research/STRATA/STRATA-v6/STRATA-code/05_Temporal_STRATA
python3 -m venv .venv && source .venv/bin/activate
python3 tools/run_m5_session.py --steps 1,2,3,6,7
deactivate
```

**B. Python 3.13 환경 (진단을 포함한 외부 기준선)**
```bash
cd ~/Research/STRATA/STRATA-v6/STRATA-code/05_Temporal_STRATA
python3.13 -m venv .venv313 && source .venv313/bin/activate
pip install -r requirements-mac.txt
python3 tools/run_m5_session.py --only-external
deactivate
```
진단 결과는 `results/real/external_check.json`(`toy_diagnosis`, 데이터셋별
`diagnosis` / `raw_sample`)에 남는다. 시간 측정은 여전히 의미론이 일치할 때만 한다.

### 10.3 STRATA-v7: 외부 기준선만 (백로그 R13, 어댑터 수정)

v6 진단에서 Raphtory 어댑터가 모든 정점을 도달로 셌음이 드러났다(`reachable_nodes`
목록이 비어 있어도 사전 자체는 참으로 판정됨). STRATA-v7은 어댑터를 고치고,
"모든 출발점이 모든 정점에 도달" 결과를 경고하는 검사를 더했다. 다른 변경은 없으므로
B만 다시 실행한다:

```bash
cd ~/Research/STRATA/STRATA-v7/STRATA-code/05_Temporal_STRATA
python3.13 -m venv .venv313 && source .venv313/bin/activate
pip install -r requirements-mac.txt
python3 tools/run_m5_session.py --only-external
deactivate
```
의미론 검사가 strict나 closure와 일치하면 같은 명령이 시간 측정까지 실행한다
(`results/real/external_baselines.json`).

결과(2026-11-15): Raphtory 0.17.0은 작은 예제 5개 모두에서 closure 의미론을 따르고,
표본 출발점 192개 중 189개에서 STRATA closure 관계와 같다. 나머지 3개에서는 도달 가능한
정점을 1개씩 빠뜨린다. 백로그 R13 결정 (가)에 따라 **시간은 측정하지 않는다**.

### 10.4 STRATA-v8: M5 결과 파일에서 논문 수치 / 표 / 그림 생성

STRATA-v8은 새로 측정하지 않는다. 논문의 M5 수치는 모두 결과 파일에서 계산한 LaTeX
매크로이고, supplementary 표(ms 단위)와 멀티코어 그림 3개도 같은 파일에서 그린다.
앞으로 세션을 다시 실행하면 아래로 다시 만든다(시간 측정 없음):

```bash
cd ~/Research/STRATA/STRATA-v8/STRATA-code/05_Temporal_STRATA
python3 experiments/external_witness.py          # 외부 도구 불일치의 반증 경로
python3 experiments/m5_numbers.py                # m5_numbers.tex, 표 8, 표 S1-S2
P=../../STRATA-latex/STRATA-paper-EN/figures
python3 experiments/plot_parallel.py paper --json results/real/parallel_scaling.json \
    --large results/real/parallel_scaling_large.json --out $P/fig_scaling.pdf
python3 experiments/plot_parallel.py breakdown --json results/real/parallel_scaling.json \
    --large results/real/parallel_scaling_large.json --out $P/fig_breakdown.pdf
python3 experiments/plot_parallel.py roofline --json results/real/bandwidth_roofline.json \
    --out $P/fig_roofline.pdf
python3 tools/check_numbers.py                   # 논문 사본이 낡으면 실패
python3 tools/make_submission_bundle.py --scan-only   # 제출 묶음 익명 스캔
```
`check_numbers.py`는 이제 초록도 검사한다(150~190단어, 수식 없음, 수치가 매크로와 같음).
논문을 현재 결과 파일과 비교하므로, 새 측정 직후 `m5_numbers.py`를 실행하고 논문을 다시
빌드한 뒤 쪽수(12쪽)를 확인한다.

### 10.5 STRATA-v9: 반복값 저장 / 병렬 전처리 / 새 단계를 포함한 재측정

STRATA-v9는 v8 모의 심사가 요구한 측정(백로그 R20, R25, R26, R27, R29, R31)을
준비만 하고 실행하지 않았다.

- 시간을 재는 모든 구성은 이제 **반복 측정값 전체**를 저장한다(R26). 2 / 3단계는
  순차 계획과 배열이 같은지 확인한 뒤 스레드 수마다 **병렬 전처리** 시간도 잰다
  (R20, `--prep-parallel compare`).
- 8단계는 **단일 스레드 실험**을 M5 Max에서 다시 잰다(R25). `--no-tex`로 원고
  표를 직접 쓰지 않게 하고, `tools/machine_stamp.py`가 기계 정보와 파일 해시를 남긴다.
- 9단계는 **n ≥ 10^5 공개 시간 그래프** 두 개(SNAP sx-askubuntu, n = 159,316,
  간선 964,437; sx-superuser, n = 194,085, 간선 1,443,339)를 내려받아
  **표본 기준**(순차 컴파일 코어, 표본 출발점 64개 열을 독립 출발점별 BFS와 대조)으로
  스케일링을 잰다. 내려받기에 실패하면 기록하고 나머지를 건너뛴다(대체 데이터 없음).
- 10단계는 시간 / 일 / 주 단위로 묶은 실데이터에서 **closure**를 잰다(sweep 수,
  Lemma 4 확인, strict / closure 시간)(R29).
- 11단계는 `python-graphblas`를 설치해 **정적** closure를 참고점으로만 잰다(R31).
  패키지를 쓸 수 없으면 건너뛴다.

주 환경에서 실행한다(약 1~2시간. 논문의 모든 시간에 반복값이 남도록 4~6단계도 다시 잰다):

```bash
cd ~/Research/STRATA/STRATA-v9/STRATA-code/05_Temporal_STRATA
source .venv/bin/activate
python3 tools/run_m5_session.py --dry-run --steps 1,2,3,4,5,6,8,9,10,11
python3 tools/run_m5_session.py --steps 1,2,3,4,5,6,8,9,10,11
```

바뀐 단계와 새 단계에서 실행되는 명령:

```bash
# step 2: strong scaling: layout x schedule
NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py --threads 1,2,4,6,8,12,16,18 --repeats 5 --layouts rowmajor,tiled,auto --schedules static,dynamic --oversub 4 --chunk-policy threads --table-config tiled:static --prep-parallel compare --out results/real/parallel_scaling.json --tex-out results/real/table_parallel_scaling.tex
# step 3: large graphs and the sequential fraction
NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py --skip-real --skip-synthetic --large 50000:1000000:400 --large 100000:2000000:400 --block-bits 65536 --reference compiled --baseline-max-work 1e10 --threads 1,2,4,6,8,12,16,18 --repeats 3 --layouts rowmajor,tiled,auto --schedules static,dynamic --chunk-policy threads --prep-parallel compare --out results/real/parallel_scaling_large.json --tex-out results/real/table_parallel_scaling_large.tex
# step 8: single-thread experiments on the evaluation machine (R25)
NUMBA_NUM_THREADS=1 python3 experiments/perf_table.py --no-tex
NUMBA_NUM_THREADS=1 python3 experiments/compiled_baselines.py --no-tex
NUMBA_NUM_THREADS=1 python3 experiments/real_temporal_benchmark.py --reps 5
NUMBA_NUM_THREADS=1 python3 experiments/ablation_snapshot_necessity.py
NUMBA_NUM_THREADS=1 python3 experiments/ablation_strict_prune.py
NUMBA_NUM_THREADS=1 python3 experiments/index_crossover.py
NUMBA_NUM_THREADS=1 python3 experiments/real_dynamic_and_blocking.py
python3 tools/machine_stamp.py --label single_thread --out results/real/single_thread_machine.json results/perf/perf_table.json results/perf/compiled_baselines.json results/real/real_temporal_results.json results/real/snapshot_necessity.json results/real/ablation_strict_prune.json results/real/index_crossover.json results/real/real_dynamic_blocking.json
# step 9: large public temporal graphs (R27)
python3 experiments/fetch_real_temporal.py --names sx-askubuntu,sx-superuser --log results/real/fetch_large.json
NUMBA_NUM_THREADS=18 python3 experiments/parallel_scaling.py --skip-real --skip-synthetic --extra-real sx-askubuntu --extra-real sx-superuser --reference sampled --baseline-max-work 1e10 --threads 1,2,4,6,8,12,16,18 --repeats 3 --layouts tiled --schedules static,dynamic --chunk-policy threads --prep-parallel compare --out results/real/parallel_scaling_realbig.json --tex-out results/real/table_parallel_scaling_realbig.tex
# step 10: closure semantics on bucketed real timestamps (R29)
NUMBA_NUM_THREADS=18 python3 experiments/closure_grouped.py --threads 1,2,4,6,8,12,16,18 --repeats 5 --out results/real/closure_grouped.json
# step 11: multithreaded GraphBLAS reference point, static closure (R31)
pip install python-graphblas
NUMBA_NUM_THREADS=18 python3 experiments/graphblas_reference.py --threads 1,2,4,6,8,12,16,18 --repeats 5 --out results/real/graphblas_reference.json
```

끝나면 "재실측 끝"이라고 알려 준다. 원고 수치는 다음 반영에서 `experiments/m5_numbers.py`
(10.4절)로 다시 만든다. 그 전에는 원고가 바뀌지 않는다.

### 10.6 STRATA-v10: 1 / 8 / 9 / 11단계 재실행 (백로그 R32 / R33)

v9 세션(2026-11-16)은 1~6 / 8~10단계를 마쳤고, 11단계는 `experiments/graphblas_reference.py`의 하네스 결함으로 멈췄다(v10에서 수정, R32). 8단계는 python-graphblas 설치 전에 실행되어 GraphBLAS 열이 비었다. v10은 시각 범위가 넓은 스트림의 정렬도 바꿨으므로(R33) 9단계를 다시 잰다. STRATA-v10 폴더에서 실행한다.

```bash
cd STRATA-v10/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 1,8,9,11
```

- 복사된 `.venv313/bin/activate`는 아직 STRATA-v9 폴더를 가리키므로, 위처럼 v10 인터프리터를 경로로 호출한다. 실행기는 모든 `python3` / `pip` 명령에 이 인터프리터를 쓴다.
- 1단계: `requirements-optional.txt`로 python-graphblas를 설치하고, 실제 python-graphblas 경로를 포함한 테스트 227개를 실행한다.
- 8단계: GraphBLAS 열을 포함해 단일 스레드 묶음을 다시 잰다.
- 9단계: 순위 압축 정렬로 sx-askubuntu / sx-superuser를 다시 잰다.
- 11단계: GraphBLAS 정적 closure 참고점을 잰다.

약 15분. 끝나면 "재실측 끝"을 보내 주세요. `experiments/m5_numbers.py`로 논문 수치를 다시 만들고, 9 / 11단계에 의존하는 문장은 새 결과 파일로 작성한다.

### 10.7 STRATA-v11: 스레드 수 기록 / GraphBLAS 1스레드 / 적응형 정렬 (백로그 R35 / R36 / R37)

v11은 단일 스레드 하네스가 런타임별 실제 스레드 수(Numba, GraphBLAS, OpenMP 풀)를 결과 파일에 기록하게 하고(R37), `compiled_baselines.py`의 GraphBLAS를 1스레드로 고정하며(R36), 거의 시간순인 입력은 적응형 안정 정렬로 정렬한다(R35). `experiments/m5_numbers.py`는 스레드 기록이 없는 단일 스레드 결과 파일을 거부하므로, 테스트(1단계)는 8단계가 새 파일을 쓴 뒤에야 통과한다. 아래 순서로 실행한다.

```bash
cd STRATA-v11/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 8,11
.venv313/bin/python3 tools/run_m5_session.py --steps 1,9
```

- 8단계: 스레드 기록과 GraphBLAS 1스레드 열을 포함한 단일 스레드 묶음
- 11단계: 행마다 GraphBLAS 풀 크기를 읽어 기록하는 참고점
- 1단계: 환경과 테스트 235개(스레드 기록 검사 포함)
- 9단계: 적응형 정렬로 sx-askubuntu / sx-superuser (기록에 `sort_path` 표시)

모두 약 20분. 끝나면 "재실측 끝"을 보내 주세요.

정정(2026-11-16, v11 실행에서 발견): 1단계는 원고의 생성 파일이 결과 파일에서 새로 만든 것과 같은지도 확인하므로, 8단계가 새 측정값을 쓰면 그 파일들을 다시 만들기 전까지 실패한다. 두 명령 사이에 `.venv313/bin/python3 experiments/m5_numbers.py`를 실행하거나, v11처럼 다음 반영에서 생성 파일을 다시 만든 뒤 `--steps 1,9`를 다시 실행한다. 스레드 기록 검사 자체는 첫 실행에서 통과했다.

### 10.8 STRATA-v12: v11 전처리 경로로 합성 스트림 재측정 (백로그 R40)

v11의 합성 스트림 전처리 수치는 적응형 정렬(R35)이 들어가기 전에 측정했다. 합성 스케일링 두 단계만 다시 실행한다. 원고의 생성 파일은 새 결과로 나중에 다시 만들므로 같은 명령에 1단계를 넣지 않는다(10.7 정정 참조).

```bash
cd STRATA-v12/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 2,3
```

약 12분. 끝나면 "재실측 끝"을 보내 주세요. 커널도 다시 측정하므로 초록 수치의 끝자리가 바뀔 수 있으며, `tools/check_numbers.py`가 확인한다. `tools/check_figures.py`(R38)를 포함한 6개 게이트는 `python3 tools/run_gates.py`로 한 번에 실행한다.

### 10.9 STRATA-v13: Temporal Ligra 기준선과 자동 재생성 (백로그 R39 / R42)

새 단계가 둘이다. 12단계는 이 기계에서 Temporal Ligra(MIT, github.com/jshun/ligra, `temporal` 분기, 고정 커밋 `75eb1a29`)를 빌드하고, earliest arrival 탐색이 STRATA와 같은 관계를 계산하는지(지속 시간 1 = strict, 0 = closure; 장난감 스트림 + 표본 출발점 64개) 확인한 뒤에만, 1 / 18스레드에서 모든 출발점을 시간 측정하며 도달 집합을 모두 검증한다.

```bash
python3 experiments/temporal_ligra.py fetch --dest external/temporal_ligra
python3 experiments/temporal_ligra.py build --src external/temporal_ligra
python3 experiments/temporal_ligra.py check --src external/temporal_ligra --out results/real/temporal_ligra_check.json
python3 experiments/temporal_ligra.py time --src external/temporal_ligra --threads 1,18 --out results/real/temporal_ligra_time.json
```

복제본은 `external/`에 두며 패키지에 넣지 않는다. Apple clang에서 OpenMP를 쓰려면 Homebrew `libomp`가 필요하다(`brew install libomp`). 없으면 순차 변형을 빌드하고 그 사실을 기록하므로, 가능하면 실행 전에 설치한다.

13단계(R42)는 측정 단계를 실행하면 자동으로 붙고(`--no-regen`으로 끔), 새 결과로 원고 생성 파일을 모두 다시 만든다.

```bash
python3 experiments/m5_numbers.py
python3 experiments/make_paper_figures.py
python3 experiments/plot_parallel.py paper --json results/real/parallel_scaling.json --large results/real/parallel_scaling_large.json --out ../../STRATA-latex/STRATA-paper-EN/figures/fig_scaling.pdf
python3 experiments/plot_parallel.py breakdown --json results/real/parallel_scaling.json --large results/real/parallel_scaling_large.json --out ../../STRATA-latex/STRATA-paper-EN/figures/fig_breakdown.pdf
python3 experiments/plot_parallel.py roofline --json results/real/bandwidth_roofline.json --out ../../STRATA-latex/STRATA-paper-EN/figures/fig_roofline.pdf
python3 tools/check_numbers.py
```

세션 기록에 다시 만든 파일과 sha256이 남는다. v13 실행:

```bash
cd STRATA-v13/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 12
```

(13단계는 자동으로 붙음). 약 15~30분(대부분 출발점별 Temporal Ligra 실행). 끝나면 "재실측 끝"을 보내 주세요.

### 10.10 STRATA-v14: 출발점 묶음 병렬 기준선 (백로그 R43)

Temporal Ligra는 탐색 하나 안에서 병렬화하므로 관측 시각 그래프에서 오히려 느려졌다(v13). 12단계는 이제 1스레드 실행 파일을 출발점 p개 묶음으로 나눠 p개 프로세스로 동시에 실행하는 방식(출발점별 도구에 가장 유리한 병렬화)도 측정한다. 경과 시간(프로세스 시작 / 그래프 읽기 포함)과 임계 경로 계산 시간(묶음별 도구 계산 시간 합의 최댓값)을 기록하고, 모든 출발점을 검증한다.

```bash
python3 experiments/temporal_ligra.py time-batch --src external/temporal_ligra --threads 1,6,12,18 --out results/real/temporal_ligra_batch.json
```

실행(13단계는 자동으로 붙음):

```bash
cd STRATA-v14/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 12
```

약 20~40분. 끝나면 "재실측 끝"을 보내 주세요.

### 10.11 STRATA-v16: 투고 직전 참고문헌 검사 (백로그 R46)

`tools/check_bib.py`는 인용된 refs.bib 항목을 Crossref 기록(DOI로, DOI가 없으면 제목으로 찾은 뒤 DBLP까지, 후보로)과 대조한다. 대조 항목은 저자 성 / 이름 / 순서 / 수, 제목, 연도, 권, 쪽이다. 클라우드 작업 공간과 Mac VM은 Crossref / DBLP에 접속하지 못하므로, macOS에서 직접 한 번 실행한다(네트워크 필요, 측정 아님, 1분 이내).

```bash
cd STRATA-v16/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/check_bib.py --fetch
```

결과는 `reports/bib_check.md`(항목별 일치 / 불일치 / 미확인)와 `reports/bib_cache.json`(받아 온 기록, 이후에는 오프라인 대조)에 저장된다. 종료 코드 1은 불일치가 하나 이상 있다는 뜻이다. 끝나면 "bib 검사 끝"을 보내 주세요.

### 10.12 STRATA-v17: 참고문헌 검사 재실행 (백로그 R46 / R47)

v17은 잘못된 응답을 읽는 방식을 고쳤고(DBLP 봇 차단 페이지가 더는 실행을 멈추지 않음), 이름 앞 소사(von 등)를 무시하며, 같은 논문 / 같은 연도의 제목 후보를 고른다. refs.bib에는 DOI 15개가 추가됐다. macOS에서 같은 검사를 다시 실행한다(DOI로 받은 기록은 캐시를 재사용하고, v16 규칙으로 고른 제목 후보는 다시 받는다).

```bash
cd STRATA-v17/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/check_bib.py --fetch
```

보고서에 "fetch incomplete" 항목이 남으면 몇 분 뒤 같은 명령을 한 번 더 실행한다. 끝나면 "bib 검사 끝"을 보내 주세요.

v18 폴더에서도 같은 검사를 실행할 수 있다(`cd STRATA-v18/STRATA-code/05_Temporal_STRATA`). v18은 refs.bib와 도구를 바꾸지 않았다.

### 10.13 STRATA-v19: 정정 후 참고문헌 재검사 (백로그 R50)

v19는 참고문헌 4개(tang2010temporalreachability, restless2020, steer2024raphtory, wen2020spanreach — 옛 키 wen2020topkreach)를 고치고 DOI 3개를 더했다. 도구도 바뀌었다: Crossref 부제를 제목에 합치고, DBLP 앞에 OpenAlex를 조회하며, 제목 검색으로 찾은 기록이 다르면 "mismatch"가 아니라 "check"(확인 필요)로 표시하고, 기록 자체의 알려진 오류는 사유와 함께 허용할 수 있다(`reports/bib_manual.json`). 캐시는 모두 한 번 다시 받는다.

```bash
cd STRATA-v19/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/check_bib.py --fetch
```

끝나면 "bib 검사 끝"을 보내 주세요.

### 10.14 STRATA-v21: 강한 기준선과 10^7 간선 이상 실데이터 (백로그 R54)

v20 모의 심사(`reports/mock_review_v20.md`)에 답하는 세션 단계 3개를 추가했다. 14단계는 단어 묶음 시간 MS-BFS(단어당 출발점 64개, 비어 있는 출발점 단어는 건너뜀, 묶음을 스레드에 분배)를 같은 전처리 스트림에서 STRATA와 비교한다(먼저 같은 관계인지 확인). 15단계는 공개 시간 색인 도구 두 개(tgindex, TGLib)를 받아 라이선스 / README / 빌드 파일을 기록한다(어댑터는 다음 일괄 반영에서 이 기록으로 작성). 16단계는 SNAP sx-stackoverflow(n = 2,601,977, m = 63,497,050)를 받아 폭 4096의 출발점 블록 앞 8개를 1 / 18스레드로 실행하고, 표본 출발점 64개를 독립 per-source BFS와 비트 대조한다. 전체는 636블록이며, 기록에는 측정한 블록과, 외삽임을 밝힌 "평균 시간 × 636"이 함께 남는다.

```bash
NUMBA_NUM_THREADS=18 python3 experiments/temporal_msbfs.py bench --datasets CollegeMsg,BitcoinOTC,BitcoinOTCpos --synthetic --threads 1,18 --repeats 5 --out results/real/temporal_msbfs.json
python3 experiments/temporal_index_baseline.py probe --dest external --out results/real/index_baseline_probe.json
python3 experiments/fetch_real_temporal.py --names sx-stackoverflow --log results/real/fetch_stackoverflow.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 4096 --blocks 8 --threads 1,18 --schedule dynamic --state rows --verify-sources 64 --out results/real/large_real_blocked.json
```

실행(13단계는 자동으로 붙고 마지막에 실행됨):

```bash
cd STRATA-v21/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 14,15,16
```

sx-stackoverflow 다운로드와 변환은 약 6,300만 줄을 처리한다(소요 시간은 아직 재지 않음). 실행 중 블록마다 시간이 출력된다. 15 / 16단계는 네트워크(gitlab.com, snap.stanford.edu)가 필요하며, 다운로드가 실패하면 기록하고 그 단계의 나머지를 건너뛴다. 끝나면 "재실측 끝"을 보내 주세요.

### 10.15 STRATA-v22: sx-stackoverflow 넓은 블록과 TGLib 기준선 (백로그 R55 / R56 / R57)

v21 실행에서 sx-stackoverflow의 정확성은 확인했지만, 블록 4096개 출발점(64단어)에서는 18스레드가 1스레드보다 1.84배 느렸다. 17단계는 블록 폭(4096 / 65536개 출발점, 즉 64 / 1,024단어; 65536이면 상주 약 21 GB)과 스레드(1 / 6 / 12 / 18, static)를 바꿔 재고, 이어서 이 측정에서 가장 빨랐던 스레드 수로 모든 블록(65536이면 40개)을 실행해 총시간을 외삽 없이 잰다. 패킹 키가 맞지 않을 때의 전처리는 이제 시각 안정 정렬 + 여러 간선이 있는 시각 그룹 안의 정렬을 쓴다(R56). 기록에 `prep_sort`와 단계별 시간이 남는다. 18단계는 TGLib(MIT, PyPI `temporalgraphlib==1.7.1`)를 설치해 earliest-arrival 집합이 STRATA와 같은지(strict: 간선 지연 1) 먼저 확인하고, 같을 때만 모든 출발점에서 1 / 18프로세스로 시간을 잰다. tgindex는 저자 허락 전에는 쓰지 않는다(문의 초안: `reports/tgindex_permission_request.md`).

```bash
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 4096 --blocks 2 --threads 1,6,12,18 --schedule static --state rows --verify-sources 64 --out results/real/large_real_b4096.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 65536 --blocks 2 --threads 1,6,12,18 --schedule static --state rows --verify-sources 64 --out results/real/large_real_b65536.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 65536 --blocks all --threads best:results/real/large_real_b65536.json --schedule static --state rows --verify-sources 64 --out results/real/large_real_full.json
pip install temporalgraphlib==1.7.1
python3 experiments/tglib_baseline.py check --out results/real/tglib_check.json
NUMBA_NUM_THREADS=1 python3 experiments/tglib_baseline.py time --check results/real/tglib_check.json --procs 1,18 --out results/real/tglib_time.json
```

실행(13단계는 자동으로 붙고 마지막에 실행됨):

```bash
cd STRATA-v22/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 17,18
```

세션 `20261117-155538`에서 두 단계를 실행했다. 17단계 전체 실행은 6스레드로 40블록 커널 시간 708.9초였고, 18단계는 PyPI에 macOS arm64 휠이 없어 설치에서 멈췄다(10.16 참조).

### 10.16 STRATA-v23: TGLib 소스 빌드와 6스레드 최적 (백로그 R58 / R60)

PyPI의 temporalgraphlib는 Linux x86_64 휠만 있으므로, 18단계는 이제 고정 커밋(e56c3dd, PyBind11 서브모듈까지 재귀 clone)의 MIT 소스를 빌드한다. 먼저 받은 소스를 `pip install`하고, 실패하면 README의 CMake Release 빌드를 이 인터프리터에 연결한 뒤 `.pth` 파일로 등록한다. `results/real/tglib_build.json`에 경로 / 커밋 / 도구 버전 / 로그 끝부분이 남고, 빌드가 실패하면 기록만 하고 check / time은 건너뛴다. 빌드에는 `git`, `cmake`, C++ 컴파일러(Xcode 명령줄 도구)가 필요하다.

v22 측정에서 두 블록 폭 모두 6스레드가 가장 빨랐고, M5 Max는 "Super" 6코어와 "Performance" 12코어로 보고된다. 19단계는 블록 폭마다 블록 1개에서 두 설명을 가른다. 1 / 6 / 12 / 18스레드에서 static과 dynamic 스케줄(dynamic은 빠른 코어가 열 묶음을 더 가져감)을 재고, 접근 바이트 모델 + 묶음마다 읽는 스트림 바이트를 커널 시간으로 나눠 같은 스레드 수의 STREAM-triad 대역폭과 나란히 기록한다. 모든 구성은 표본 출발점에서 출발점별 BFS와 대조한다.

```bash
python3 experiments/tglib_baseline.py build --dest external --out results/real/tglib_build.json
python3 experiments/tglib_baseline.py check --out results/real/tglib_check.json
NUMBA_NUM_THREADS=1 python3 experiments/tglib_baseline.py time --check results/real/tglib_check.json --procs 1,18 --out results/real/tglib_time.json
NUMBA_NUM_THREADS=18 python3 experiments/thread_diagnosis.py --dataset sx-stackoverflow --block-bits 4096,65536 --threads 1,6,12,18 --schedules static,dynamic --repeats 3 --out results/real/thread_diagnosis.json
```

실행(13단계는 자동으로 붙고 마지막에 실행됨):

```bash
cd STRATA-v23/STRATA-code/05_Temporal_STRATA
xcode-select -p && cmake --version     # 18단계 빌드 도구 확인
.venv313/bin/python3 tools/run_m5_session.py --steps 18,19
```

세션 `20261117-191557`에서 두 단계를 실행했다. 18단계는 CMake의 OpenMP 확인에서 멈췄고(Apple clang에는 OpenMP 런타임이 없음), 19단계는 스레드 스윕을 측정했다(10.17 참조).

### 10.17 STRATA-v24: libomp로 TGLib 빌드, 열 묶음 수 대 스레드 수 (백로그 R63 / R64)

Apple clang에는 OpenMP 런타임이 없어 v23 세션에서 TGLib의 `find_package(OpenMP REQUIRED)`가 실패했다. Homebrew의 libomp를 한 번 설치하면 `build`가 `brew --prefix libomp`로 찾아 CMake에 넘긴다(pip 경로는 `CMAKE_ARGS`로). CMake 빌드가 그래도 실패하면 TGLib의 `-Werror`를 CMake include로 걸러(소스 수정 없음) 한 번 더 시도하고 `werror_relaxed`로 기록한다. check / time은 Numba의 `workqueue` 스레드 계층으로 실행해 OpenMP 런타임은 TGLib만 읽는다.

v23 스레드 진단에서, 같은 스레드 수에서도 열 묶음만 늘리면 느려졌고 static 스케줄에서는 묶음 수가 곧 스레드 수였다. 20단계는 65,536개 출발점 블록 1개에서 둘을 가른다. 6스레드로 묶음 6 / 12 / 18(스트림 통과 횟수 증가), 이어서 묶음 18개로 6 / 12 / 18스레드(동시 스레드 증가).

```bash
python3 experiments/tglib_baseline.py build --dest external --out results/real/tglib_build.json
NUMBA_THREADING_LAYER=workqueue python3 experiments/tglib_baseline.py check --out results/real/tglib_check.json
NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=1 python3 experiments/tglib_baseline.py time --check results/real/tglib_check.json --procs 1,18 --out results/real/tglib_time.json
NUMBA_NUM_THREADS=18 python3 experiments/thread_diagnosis.py --dataset sx-stackoverflow --block-bits 65536 --pairs 6:6,6:12,6:18,12:18,18:18 --repeats 3 --out results/real/thread_chunks.json
```

실행(13단계는 자동으로 붙고 마지막에 실행됨):

```bash
brew install libomp                    # 18단계용, 한 번만
cd STRATA-v24/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 18,20
```

세션 `20261117-220420`에서 두 단계를 실행했다. 18단계는 libomp를 찾았지만 TGLib 테스트 실행 파일 링크(pip 경로)와 pybind11의 CMake 최소 버전(CMake 경로)에서 멈췄고, 20단계는 조합을 측정했다(10.18 참조).

### 10.18 STRATA-v25: TGLib 링크 수정과 무작위 행 갱신 (백로그 R68 / R69)

TGLib은 OpenMP 플래그를 컴파일 줄에만 넣어 v24 빌드가 테스트 실행 파일 링크에서 실패했다. 이제 `build`는 모든 링크에 `-L<libomp>/lib -lomp`를 넘기고, CMake 경로에도 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`(TGLib pyproject가 pip에 넘기는 것과 같음)를 넣으며, `_pytglib` 바인딩을 빌드해 받은 소스의 `pytglib/__init__.py`와 함께 `external/tglib_py/`에 설치한다. 링크가 실패하면 링커의 "Undefined symbols" 줄을 기록한다.

v24 조합 측정에서 하락은 묶음 수가 아니라 동시 스레드 수를 따랐고 STREAM triad는 계속 올랐다. 21단계는 커널의 접근 형태만(묶음별 개인 타일에서 `tile[v] |= tile[u]`, 행당 1,024단어, 시각 / 스냅샷 없음) sx-stackoverflow 스트림 앞 1,600만 간선과 같은 수의 무작위 쌍으로 1 / 6 / 12 / 18스레드에서 잰다. 단어 체크섬은 스레드 수와 무관해야 한다.

```bash
python3 experiments/tglib_baseline.py build --dest external --out results/real/tglib_build.json
NUMBA_THREADING_LAYER=workqueue python3 experiments/tglib_baseline.py check --out results/real/tglib_check.json
NUMBA_THREADING_LAYER=workqueue NUMBA_NUM_THREADS=1 python3 experiments/tglib_baseline.py time --check results/real/tglib_check.json --procs 1,18 --out results/real/tglib_time.json
NUMBA_NUM_THREADS=18 python3 experiments/random_rows_bench.py --dataset sx-stackoverflow --words 1024 --edges 16000000 --threads 1,6,12,18 --repeats 3 --out results/real/random_rows.json
```

실행(13단계는 자동으로 붙고 마지막에 실행됨):

```bash
cd STRATA-v25/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 18,21
```

세션 `20261118-064755`와 `20261118-070247`이 STRATA-v25에서 두 단계를 실행했다. TGLib은 pip 경로로 빌드되어 strict 관계와 일치했고, 무작위 행 갱신은 스레드 수를 늘릴 때마다 빨라졌다(10.19 참조). 같은 날 앞선 두 세션은 v25가 생기기 전 STRATA-v24에서 v24 코드로 실행됐다.

### 10.19 STRATA-v26: 세션 안전장치와 쪽수 예산 (백로그 R71 / R72 / R66)

`run_m5_session.py`는 옆에 더 큰 번호의 `STRATA-vN` 폴더가 있는데 이전 폴더에서 실행하거나(`--allow-old`로만 진행), `--steps`에 없는 단계 번호가 있으면 측정 전에 멈춘다(종료 코드 4). 세션 기록에는 버전 폴더, 최신 폴더, 사용 가능 / 요청 단계, 인터프리터 / 작업 폴더 / 실행기 위치를 남기고, 서로 다른 `STRATA-vN`을 가리키면 경고한다. `run_gates.py`는 12쪽 / 4쪽 한도 안에 남은 줄 수를 출력한다(`tools/page_budget.py`, 정보이며 게이트 아님). 새 측정 단계는 없다.

### 10.20 STRATA-v27: 6스레드 이후의 커널 단계 (백로그 R73)

21단계의 무작위 행 갱신은 18스레드까지 빨라지지만, 같은 블록의 strict 커널은 그렇지 않다. 22단계는 65,536개 출발점 블록 1개에서 열 커널의 변형 4개를 잰다(static, 묶음 = 스레드, 1 / 6 / 12 / 18스레드). 실제 strict 커널, 같은 타일 / 복사에서 스냅샷만 뺀 것(strict 의미론 아님, 시간 비교용), 상태와 타일 사이 복사만 하는 것, 모든 타일을 시계 시작 전에 할당하고 접근해 둔 strict 커널. strict는 표본 출발점에서 출발점별 BFS와, 각 변형은 자기 기준 상태와 대조한다.

```bash
NUMBA_NUM_THREADS=18 python3 experiments/kernel_phases.py --dataset sx-stackoverflow --block-bits 65536 --threads 1,6,12,18 --repeats 3 --out results/real/kernel_phases.json
```

실행(13단계는 자동으로 붙고 마지막에 실행됨):

```bash
cd STRATA-v27/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 22
```

세션 `20261118-081108`과 `20261118-093215`가 22단계를 두 번 실행했고 결과가 일치했다. 복사만 하는 변형은 1스레드 약 3.3초에서 18스레드 32.6초로 늘었고, 간선 처리 부분은 계속 빨라졌다(10.21 참조).

### 10.21 STRATA-v28: 복사 단계, 타일을 상태로 쓰는 실행, compiled 기준선 (백로그 R75 / R76 / R79)

23단계는 같은 블록에서 복사 단계가 늘어나는 원인을 가른다. 새 상태의 모든 페이지를 시계 밖에서 미리 접근한 뒤의 복사, 한 번 할당해 시계 밖에서 다시 채운 상태로의 복사, 묶음 경계를 128바이트 캐시 라인 배수에 맞춘 복사, 그리고 타일을 상태로 쓰는 실행(공유 상태 / 복사 없음; 출발점별 BFS와 strict 쌍 수로 대조). 24단계는 타일을 상태로 써서 sx-stackoverflow를 실행한다(블록 2개로 1 / 6 / 12 / 18스레드, 이어서 가장 빠른 스레드 수로 전체 블록). 25단계는 1스레드 compiled 기준선을 다시 잰다. 표 5의 STRATA-numba 열은 시각 그룹마다 상태 전체를 복사하는 초기 실험용 커널이었고, 두 compiled 열 모두 간선 목록의 Python 정렬 시간을 포함했다. 이제 둘 다 같은 정렬 배열에서 시작하고 STRATA는 실제 코어를 쓴다.

```bash
NUMBA_NUM_THREADS=18 python3 experiments/kernel_phases.py --dataset sx-stackoverflow --block-bits 65536 --threads 1,6,12,18 --variants strict,copy,copy_prefault,copy_reuse,copy_aligned,tiles --repeats 3 --out results/real/kernel_copy.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 65536 --blocks 2 --threads 1,6,12,18 --schedule static --state tiles --verify-sources 64 --out results/real/large_real_tiles_b65536.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 65536 --blocks all --threads best:results/real/large_real_tiles_b65536.json --schedule static --state tiles --verify-sources 64 --out results/real/large_real_tiles_full.json
NUMBA_NUM_THREADS=1 python3 experiments/compiled_baselines.py --no-tex
```

실행(13단계는 자동으로 붙고 마지막에 실행됨):

```bash
cd STRATA-v28/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 23,24,25
```

23–25단계는 합쳐서 약 33분 걸린다(세션 20261118-153811 / 172302 / 202159, 세 번 모두 일관).

결과(STRATA-v29부터 원고에 사용): 새로 할당한 블록 상태로의 복사는 p = 1 → 18에서 3.0 → 30.9 s로 커지지만, 시계 전에 이미 써 둔 상태로의 같은 복사는 2.2 → 0.6 s로 줄어든다. 타일을 상태로 쓰면 31.3 → 6.5 s(strict 커널 최적의 2.7배 빠름)이고, 40개 블록 전체가 p = 6에서 709 s 대신 p = 18에서 257 s에 끝나며 쌍 수(2.68e12)는 같다. compiled 기준선은 이제 제품 코어를 재며, 합성 스트림에서 compiled STRATA 커널이 compiled 소스별 BFS보다 14.7–349.7배 빠르다(v27의 legacy 커널은 2.5–5.3배). v28 하네스의 copy_prefault는 copy와 같은 시간을 냈다. 페이지 접근 루프(자기 대입)를 컴파일러가 지울 수 있기 때문이며, STRATA-v29는 실행 시점의 0을 OR해 쓴다. 원고는 copy_prefault를 쓰지 않는다.

### 10.22 STRATA-v30: prefault 확인, 좁은 블록과 dynamic 묶음에서의 타일 상태 (백로그 R90 / R91)

왜: v29는 공유 상태 손실의 위치를 새로 할당한 상태로 타일을 되쓰는 단계로 보았다. 그런데 시계 전에 모든 페이지에 쓰는 변형(copy_prefault)은 v28에서 이를 검증하지 못했다. 접근 루프를 컴파일러가 지울 수 있었기 때문이다. v30은 실행 시점의 0을 OR해 쓴다. copy_prefault가 이제 copy_reuse처럼 스레드에 따라 줄면 원인은 새 페이지 첫 쓰기이고, 여전히 copy처럼 커지면 다른 원인이다. 타일 상태는 넓은 블록(beta = 65,536)과 static 스케줄에서만 쟀으므로, 27단계는 좁은 블록(beta = 4,096, 17단계의 rows 스윕과 비교)과 dynamic 스케줄(4p 묶음, 최대 W)을 더한다.

어떻게: 26단계는 블록 하나에서 복사 변형 세 개만 실행한다. 27단계는 설정마다 블록 두 개를 실행하고, 표본 64개를 소스별 BFS와 대조하며 쌍 수를 남긴다. ``large_real_blocked.py``는 v30부터 타일 상태가 기본값이다. 이전 세션 단계에는 ``--state rows``를 명시해, 같은 명령이 예전과 같은 것을 잰다.

무엇:

```bash
NUMBA_NUM_THREADS=18 python3 experiments/kernel_phases.py --dataset sx-stackoverflow --block-bits 65536 --threads 1,6,12,18 --variants copy,copy_prefault,copy_reuse --repeats 3 --out results/real/kernel_prefault.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 4096 --blocks 2 --threads 1,6,12,18 --schedule static --state tiles --verify-sources 64 --out results/real/large_real_tiles_b4096.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 4096 --blocks 2 --threads 1,6,12,18 --schedule dynamic --state tiles --verify-sources 64 --out results/real/large_real_tiles_dyn_b4096.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 65536 --blocks 2 --threads 1,6,12,18 --schedule dynamic --state tiles --verify-sources 64 --out results/real/large_real_tiles_dyn_b65536.json
```

실행(13단계는 자동으로 붙고 마지막에 실행됨):

```bash
cd STRATA-v30/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 26,27
```

26–27단계는 11분 걸렸다(세션 20261119-041615). 결과(STRATA-v31부터 원고에 사용): 시계 전에 모든 페이지를 써 두면 복사가 p = 1 → 18에서 2.5 → 0.8 s로 줄어 재사용 상태로의 복사와 같아지고, 새 상태로의 복사는 3.3 → 32.0 s로 커진다. 따라서 손실은 커널 안에서 새 페이지에 처음 쓰는 비용이다. 좁은 블록(beta = 4,096)에서 타일 상태는 모든 p에서 공유 상태의 0.98–1.09배이고 둘 다 6스레드 이후 빨라지지 않는다. dynamic 스케줄은 최적 p에서 타일 상태를 2.1배(beta = 4,096), 1.6배(beta = 65,536) 느리게 한다.

### 10.23 STRATA-v31: 블록 폭에 따른 타일 상태, 26–27단계 재실행 (백로그 R94 / R95)

왜: 타일 상태는 beta = 65,536에서 이득(p = 18에서 블록당 6.6 s, 1스레드 대비 4.7배)이지만 beta = 4,096(64단어, p = 18이면 스레드당 약 4단어)에서는 이득이 없다. 좁은 블록 여러 개를 동시에 돌리는 것은 열 묶음이 블록 경계를 따르는 더 넓은 블록 하나와 같은 계산이므로(정리 3), 두 폭 사이 어디서부터 스레드가 이득이 되는지, 그때 상주 메모리가 얼마인지가 질문이다. 26–27단계는 한 번만 돌았으므로 두 번째 실행으로 좁은 블록의 작은 차이가 실제인지 확인한다.

어떻게: 28단계는 beta = 16,384와 32,768에서 타일 상태를 실행한다(폭마다 블록 두 개, static, 표본 64개 검증). R95는 새 명령이 없고, 같은 세션에서 26·27단계를 다시 돌린다. ``large_real_blocked.py``는 이제 static이 기본 스케줄이다(dynamic은 두 폭 모두에서 타일 상태를 느리게 했다). 16단계에는 ``--schedule dynamic``을 명시해 예전과 같은 것을 잰다.

무엇:

```bash
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 16384 --blocks 2 --threads 1,6,12,18 --schedule static --state tiles --verify-sources 64 --out results/real/large_real_tiles_b16384.json
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 32768 --blocks 2 --threads 1,6,12,18 --schedule static --state tiles --verify-sources 64 --out results/real/large_real_tiles_b32768.json
```

실행(13단계는 자동으로 붙고 마지막에 실행됨):

```bash
cd STRATA-v31/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 26,27,28
```

26–28단계는 15분 걸렸다(세션 20261119-050516). 26–27단계는 첫 세션과 7% 이내로 같았다. 28단계: 최적 블록당 평균 시간은 beta = 16,384에서 4.96 s(p = 12, 블록 상태 5.3 GB), beta = 32,768에서 3.82 s(p = 18, 10.7 GB)이고, 소스당 비용은 beta = 4,096에서 65,536까지 3.5배 줄어든다. 스레드당 단어 수만으로는 최적 p를 예측하지 못한다(beta = 4,096은 스레드당 10.7단어에서 가장 빠르고, beta = 16,384는 14.2단어에서 이미 느려진다).

### 10.24 STRATA-v32: beta = 32,768에서 전체 블록 (백로그 R98)

왜: 절반 폭 블록은 블록 상태를 절반으로 줄이지만, 전체 시간은 블록 두 개에서 외삽한 값뿐이었다. 어떻게: 24단계와 같은 전체 실행을 beta = 32,768, 18스레드(28단계의 최적 p)로 한다. 결과(2026-11-19 직접 실행, 이제 세션 29단계): 80개 블록 전체 280 s, 쌍 수 2.68e12 동일, 최대 RSS 19.3 GB(beta = 65,536은 257 s, 30.0 GB). ``large_real_blocked.py --memory-gb G``는 상태가 G GB에 들어가는 가장 넓은 2의 거듭제곱 블록을 고른다(여기서 11 → 32,768, 25 → 65,536).

```bash
NUMBA_NUM_THREADS=18 python3 experiments/large_real_blocked.py --dataset sx-stackoverflow --block-bits 32768 --blocks all --threads 18 --schedule static --state tiles --verify-sources 64 --out results/real/large_real_tiles_full_b32768.json
```

```bash
cd STRATA-v32/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 29
```

### 10.25 STRATA-v34: 비블록 커널의 상태, 큰 그래프의 창, 단계 수 (백로그 R102 / R109 / R105)

왜: 모의 심사 2차(reports/mock_review_v32.md)는 sx-stackoverflow에서 찾은 첫 쓰기 비용이 큰 합성 스트림의 비블록 tiled 커널도 제한하는지(AE 필수 2), 큰 실그래프에서의 롤링 창(R4-1), compiled 배율이 64개 소스 묶음을 넘는 이유(R3-1)를 묻는다.

어떻게: 30단계는 큰 합성 스트림을 새 상태(제품 경로), 시계 전에 모든 페이지를 써 둔 상태(진단용), 타일을 상태로(폭 n인 블록 하나) 세 가지로 실행하며 세 결과의 쌍 수와 표본 열이 같아야 한다. 31단계는 sx-askubuntu의 롤링 창 8개(각각 간선의 4분의 1)를 18스레드 타일 상태로 다시 계산하고 창마다 표본 16개를 소스별 BFS와 대조하며, 절반 스트림 상태에 이후 묶음 5개를 더한 결과가 전체 재계산과 비트 단위로 같아야 한다. 32단계는 results/real/compiled_real.json의 두 compiled 커널의 기본 단계 수를 센다(시계 없음). 관측 데이터에서 BFS/STRATA 단계 비는 34–45배이므로, M5 Max에서 BFS 한 단계는 STRATA 단어 연산의 4.8–5.3배 비용이다.

무엇:

```bash
NUMBA_NUM_THREADS=18 python3 experiments/tile_state_unblocked.py --large 50000:1000000:400 --large 100000:2000000:400 --threads 1,6,12,18 --repeats 3 --out results/real/tile_state_unblocked.json
python3 experiments/fetch_real_temporal.py --names sx-askubuntu --log results/real/fetch_askubuntu_window.json
NUMBA_NUM_THREADS=18 python3 experiments/window_large.py --dataset sx-askubuntu --threads 18 --slots 8 --window-frac 0.25 --batches 5 --out results/real/window_large.json
python3 experiments/op_counts.py --out results/real/op_counts.json
```

```bash
cd STRATA-v34/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 30,31,32
```

30–32단계는 1.5분 걸렸다(세션 20261119-180704). 결과(STRATA-v35부터 원고에 사용): 큰 합성 스트림에서 타일 상태는 공유 상태 시간의 0.89–1.03배, 사전 쓰기는 0.97–1.03배여서, 미리 정한 규칙(R112: 두 스트림 모두 모든 p에서 타일이 5% 이상 빠름)을 충족하지 못했고 원고는 공유 상태 결과를 유지한다. 비블록 실행은 상태를 한 번만 쓰므로 블록 실행의 첫 쓰기 비용이 나타나지 않는다. sx-askubuntu에서 18스레드로 241k 간선 창 8개는 각 40–50 ms, 96k 간선 삽입은 102–110 ms(전체 재계산 320 ms)이다. tiled 커널이 호출마다 3.2 GB 상태 전체를 모으고 되쓰기 때문이다.

### 10.26 STRATA-v34: 두 번째 기계 (백로그 R103)

왜: 프로젝트 규칙(R101 결정 B)은 사용자가 두 번째 기계에서 직접 잰 결과를 그 기계 안의 비교로만 보고하도록 허용한다.

어떻게: ``tools/run_server_session.py --tag <이름>``은 16, 3, 30, 26, 24, 28단계를 논리 CPU 수 c에 대해 스레드 1, c/8, c/4, c/2, c로 다시 실행하고 모든 파일을 results/server/<이름>/에 쓴다. results/real/은 건드리지 않는다. 먼저 ``--dry-run``으로 계획을 확인한다.

```bash
python3 tools/run_server_session.py --tag <이름> --dry-run
python3 tools/run_server_session.py --tag <이름>
```

기계 내 수치: ``python3 experiments/server_numbers.py --tag <이름>``은 results/server/<이름>/만 읽어 STRATA-latex/STRATA-paper-EN/server_numbers_<이름>.tex에 비율, 스레드 수, 기계 설명만 쓴다. 다른 종류의 매크로가 생기면 멈춘다.

### 10.27 STRATA-v36: 제자리 삽입 묶음 (백로그 R113)

왜: 31단계에서 sx-askubuntu의 96k 간선 삽입이 102–110 ms로 전체 재계산(320 ms)의 약 3분의 1이었다. tiled 커널이 호출마다 3.2 GB 상태 전체를 모으고 되쓰기 때문이다. row-major는 묶음이 쓰는 행만 갱신하므로, 명제 5대로 삽입 비용이 묶음 크기를 따라야 한다.

어떻게: 33단계는 절반 스트림 상태의 사본 두 개를 같은 묶음 5개로 진행한다. 하나는 tiled, 하나는 row-major 제자리(static, 묶음 18개)이며, 둘 다 전체 재계산과 비트 단위로 같아야 한다. 창 계산은 건너뛴다.

```bash
NUMBA_NUM_THREADS=18 python3 experiments/window_large.py --dataset sx-askubuntu --threads 18 --slots 0 --batches 5 --insert-layouts tiled,rowmajor --out results/real/window_insert.json
```

```bash
cd STRATA-v36/STRATA-code/05_Temporal_STRATA
.venv313/bin/python3 tools/run_m5_session.py --steps 33
```

33단계는 1분 걸렸다(세션 20261120-043121). 결과(STRATA-v37부터 원고에 사용): row-major 삽입은 96k 간선 묶음당 24–26 ms로 tiled(97–105 ms)보다 4.1배 빠르고, 같은 실행의 전체 재계산은 191 ms라 삽입 비용이 묶음 크기를 따른다. 전체 재계산은 31단계(창 계산 뒤)에서 320 ms, 여기서 191 ms였으므로 원고는 한 실행 안의 수치끼리만 비교한다.

