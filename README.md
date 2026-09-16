# ♻️ Garbage Classification

**Klasifikacija vrsta otpada primenom konvolutivnih neuronskih mreža (CNN)**

 *Veštačka inteligencija sa primenama*.

---

##  Opis projekta

Projekat predstavlja kompletan sistem za automatsku klasifikaciju šest vrsta otpada na osnovu slika, koristeći konvolutivnu neuronsku mrežu (CNN) implementiranu u PyTorch-u. Sistem obuhvata:

- **Trenirani model** mašinskog učenja
- **Preprocessing pipeline** za pripremu podataka
- **Front-end aplikaciju** (Streamlit) za interakciju sa modelom
- **Docker kontejner** za jednostavno pokretanje
- **MLflow** za praćenje eksperimenata

Klase koje model prepoznaje: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

---

##  Dataset

Korišćen je **Garbage Classification** dataset sa [Kaggle](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) platforme.

| Klasa | Broj slika | Procenat |
|---|---|---|
| paper | 594 | 23.51% |
| glass | 501 | 19.83% |
| plastic | 482 | 19.07% |
| metal | 410 | 16.22% |
| cardboard | 403 | 15.95% |
| trash | 137 | 5.42% |
| **Ukupno** | **2.527** | **100%** |

**Podela podataka:**
- Trening skup: 1.768 slika (70%)
- Validacioni skup: 379 slika (15%)
- Test skup: 380 slika (15%)

> ⚠️ Dataset je nebalansiran – klasa *trash* ima znatno manje primera od ostalih.

---

##  Arhitektura modela

```
Conv2d(3 → 32, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool
Conv2d(32 → 64, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool
Conv2d(64 → 128, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool
Conv2d(128 → 256, kernel=3, padding=1) + BatchNorm + ReLU + AdaptiveAvgPool
Dropout(0.5)
Linear(256 → 6)
```

- **Ukupno parametara:** 390.918 (baseline)
- **Loss funkcija:** CrossEntropyLoss
- **Optimizator:** Adam (lr=0.001, weight_decay=0.0001)
- **Scheduler:** StepLR (step=10, gamma=0.1)
- **Batch size:** 32

---

##  Rezultati

Izvršeno je **5 eksperimenata** sa različitim konfiguracijama. Svi modeli su trenirani na istom train/val/test splitu (1768/379/380 slika).

### Poređenje eksperimenata

| Eksperiment | Best Val Acc | **Test Acc** | Parametri | Ključna razlika |
|---|---|---|---|---|
| baseline | 0.7045 | 0.6737 | 390.918 | Osnovna konfiguracija (30 epoha) |
| larger_model | 0.6781 | 0.6474 | 1.555.974 | Više filtera (64→512) |
| high_dropout | 0.6623 | 0.6184 | 390.918 | Dropout 0.7 |
| **no_augmentation**  | **0.7335** | **0.7316** | 390.918 | Bez augmentacije podataka |
| higher_lr | 0.6201 | 0.5895 | 390.918 | Learning rate 0.005 |

**Najbolji model:** `no_augmentation` sa **73.16%** test tačnosti.

### Ključni nalazi

1. **Augmentacija je škodila** na ovom datasetu – `no_augmentation` (73.16%) je bolji od `baseline` sa augmentacijom (67.37%).
2. **Veći model nije pomogao** – `larger_model` (64.74%) je lošiji od baseline-a, verovatno zbog overfitting-a na malom datasetu.
3. **Previsok learning rate škodi** – `higher_lr` (58.95%) je najgori rezultat.
4. **Dropout 0.7 je previše** – `high_dropout` (61.84%) je lošiji od baseline-a (0.5).

### Per-class metrike (baseline model)

| Klasa | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| cardboard | 0.89 | 0.82 | 0.85 | 61 |
| paper | 0.77 | 0.82 | 0.79 | 89 |
| plastic | 0.71 | 0.61 | 0.66 | 72 |
| glass | 0.50 | 0.79 | 0.61 | 75 |
| metal | 0.58 | 0.40 | 0.48 | 62 |
| trash | 0.83 | 0.24 | 0.37 | 21 |

### Praćenje eksperimenata (MLflow)

Rezultati svih eksperimenata su logovani u **MLflow**:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Zatim otvoriti `http://localhost:5000` → **Model training** tab → eksperiment **Garbage-Classification**.

### Vizualizacija

Svi grafici se nalaze u `results/plots_full/`:
- `learning_curves.png` – krive učenja
- `confusion_matrix.png` – matrica konfuzije
- `roc_curves.png` – ROC krive
- `precision_recall_curves.png` – PR krive
- `class_distribution.png` – raspodela klasa
- `error_examples.png` – primeri grešaka

---

##  Struktura projekta

```
Garbage-Classification/
├── src/
│   ├── data_pipeline.py          # Preprocessing i DataLoader
│   ├── dataset.py                # Custom Dataset klasa
│   ├── model_architecture.py     # CNN model
│   ├── train.py                  # Trening jedne konfiguracije
│   ├── evaluate.py               # Evaluacija na test skupu
│   ├── run_experiments.py        # Pokretanje svih eksperimenata
│   ├── split_dataset.py          # Podela na train/val/test
│   ├── experiment_utils.py       # Pomoćne funkcije za eksperimente
│   ├── log_to_mlflow.py          # Logovanje u MLflow
│   └── visualize_full.py         # Generisanje svih grafika
├── configs/
│   └── experiments.yaml          # Konfiguracije eksperimenata
├── data/
│   ├── raw/                      # Originalne slike (Kaggle dataset)
│   └── splits/                   # train.csv, val.csv, test.csv
├── models/
│   ├── best_model.pth            # Najbolji model (73.16%)
│   ├── no_augmentation_best.pth
│   ├── larger_model_best.pth
│   ├── high_dropout_best.pth
│   └── higher_lr_best.pth
├── results/
│   ├── all_experiments.json      # Rezultati svih eksperimenata
│   ├── evaluation_results.json   # Detaljna evaluacija
│   ├── plots/                    # confusion_matrix, metrics_per_class
│   └── plots_full/               # learning_curves, ROC, PR, error_examples
├── logs/                         # MLflow logovi po eksperimentu
├── screenshots/                  # Screenshot-ovi aplikacije i MLflow-a
├── notebooks/
│   └── 01_data_analysis.ipynb    # Eksploratorna analiza
├── app.py                        # Streamlit front-end
├── Dockerfile                    # Docker konfiguracija
├── mlflow.db                     # MLflow SQLite baza
├── requirements.txt              # Sve zavisnosti (lokalni razvoj)
├── requirements-docker.txt       # Minimalni set za Docker
├── .dockerignore
└── README.md
```

---

##  Pokretanje

### 1. Kloniranje repozitorijuma

```bash
git clone https://github.com/nikolartn-it/Garbage-Classification.git
cd Garbage-Classification
```

### 2. Kreiranje virtuelnog okruženja

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows PowerShell
# source .venv/bin/activate     # Linux/Mac
```

### 3. Instalacija zavisnosti

```bash
pip install -r requirements.txt
```

### 4. Priprema podataka

Postavi Kaggle dataset u `data/raw/Garbage classification/`, zatim:

```bash
python src/split_dataset.py
```

Ovo kreira `train.csv`, `val.csv` i `test.csv` u `data/splits/`.

### 5. Treniranje modela

```bash
python src/train.py
```

Ili pokreni sve eksperimente odjednom:

```bash
python src/run_experiments.py
```

### 6. Evaluacija

```bash
python src/evaluate.py
python src/visualize_full.py
```

### 7. MLflow UI (praćenje eksperimenata)

```bash
python src/log_to_mlflow.py
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Otvara se na `http://localhost:5000`.

### 8. Pokretanje front-end aplikacije (lokalno)

```bash
streamlit run app.py
```

Aplikacija je dostupna na: `http://localhost:8501`

---

## 🐳 Docker

### Build slike

```bash
docker build -t garbage-classifier .
```

### Pokretanje kontejnera

```bash
docker run -d -p 8502:8501 --name garbage-app garbage-classifier
```

Aplikacija je dostupna na: **http://localhost:8502**

> **Napomena:** Port 8501 je bio zauzet na razvojnoj mašini, pa je aplikacija mapirana na **8502**. Interno, Streamlit i dalje sluša na 8501.

### Zaustavljanje i brisanje

```bash
docker stop garbage-app
docker rm garbage-app
```

---

##  Tehnologije

- **Python 3.12**
- **PyTorch 2.x** + **torchvision** – neuronske mreže
- **NumPy**, **Pandas** – obrada podataka
- **scikit-learn** – metrike
- **Matplotlib**, **Seaborn** – vizualizacija
- **Streamlit** – front-end
- **MLflow** – praćenje eksperimenata
- **Docker** – kontejnerizacija

---

##  Eksperimenti

Definisano je 5 konfiguracija u `src/experiment_utils.py`:

| Eksperiment | Opis |
|---|---|
| `baseline` | Osnovna konfiguracija (30 epoha) |
| `larger_model` | Veći model sa filterima [64, 128, 256, 512] |
| `high_dropout` | Dropout = 0.7 |
| `no_augmentation` | **Bez augmentacije podataka**  |
| `higher_lr` | Learning rate = 0.005 |

Rezultati se čuvaju u `results/all_experiments.json` i MLflow bazi.

---

##  Screenshot-ovi

Screenshot-ovi aplikacije i MLflow interfejsa nalaze se u folderu `screenshots/`:
- `app_upload.png` –  početni ekran
- `app_result.png` –  rezultat predikcije
- `docker_running.png` – Docker kontejner aktivan
- `mlflow_compare.png` – MLflow poređenje eksperimenata
- `mlflow_best_run.png` – MLflow najbolji run



---

 
GitHub: [@nikolartn-it](https://github.com/nikolartn-it)

---

