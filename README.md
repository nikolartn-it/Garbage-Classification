# Garbage-Classification

Klasifikacija vrsta otpada primenom konvolutivnih neuronskih mreža

1. UVOD

Cilj ovog seminarskog rada je razvoj sistema za automatsku klasifikaciju
vrsta otpada koriscenjem konvolutivnih neuronskih mreza (CNN).

Dataset koji se koristi je Garbage Classification dataset sa Kaggle platforme,
koji sadrzi slike sest razlicitih klasa otpada:
cardboard, glass, metal, paper, plastic i trash.

Model je implementiran u Python programskom jeziku koriscenjem PyTorch
biblioteke. Rad prikazuje ceo proces od pripreme podataka, preko
treniranja modela, do evaluacije rezultata.


2. OPIS SKUPA PODATAKA

Dataset sadrzi ukupno 2.527 slika podeljenih u sest klasa.

Distribucija slika po klasama:
- paper: 594 slika (23.51%)
- glass: 501 slika (19.83%)
- plastic: 482 slike (19.07%)
- metal: 410 slika (16.22%)
- cardboard: 403 slike (15.95%)
- trash: 137 slika (5.42%)

Sve slike su u JPG formatu, RGB modu boja, dimenzija 512x384 piksela.
Dataset je nebalansiran - klasa trash ima znatno manje primera
od ostalih klasa, sto predstavlja izazov za model.

Podaci su podeljeni na:
- Trening skup: 1768 slika (70%)
- Validacioni skup: 379 slika (15%)
- Test skup: 380 slika (15%)

  3. Preprocessing pipeline
Pre treniranja modela, podaci prolaze kroz sledeće korake pripreme:

3.1 Učitavanje i transformacija slika
Sve slike se učitavaju pomoću torchvision.datasets.ImageFolder i transformišu u tenzore dimenzija 224×224 piksela.

3.2 Normalizacija
Koristi se standardna ImageNet normalizacija:
Mean: [0.485, 0.456, 0.406]
Std: [0.229, 0.224, 0.225]

3.3 Augmentacija podataka (samo trening skup)
Za trening skup korišćene su sledeće augmentacije:
Random horizontal flip (p=0.5)
Random rotation (do 10 stepeni)
Color jitter (brightness, contrast, saturation, hue)
Validacioni i test skupovi nisu augmentirani – koristi se samo resize i normalizacija.

3.4 Kreiranje DataLoader-a
Podaci se učitavaju u batch-evima veličine 32 pomoću torch.utils.data.DataLoader, sa shuffle=True za trening skup.


4. METODOLOGIJA

4.1 Arhitektura modela

Koriscen je CNN model sa sledecom arhitekturom:

- Conv2d(3 -> 32, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool
- Conv2d(32 -> 64, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool
- Conv2d(64 -> 128, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool
- Conv2d(128 -> 256, kernel=3, padding=1) + BatchNorm + ReLU + AdaptiveAvgPool
- Dropout(0.5)
- Linear(256 -> 6)

Model ima ukupno 390.918 parametara.

4.2 Hiperparametri

- Broj epoha: 30
- Batch size: 32
- Learning rate: 0.001
- Weight decay: 0.0001
- Optimizator: Adam
- Loss funkcija: CrossEntropyLoss
- Scheduler: StepLR (step=10, gamma=0.1)

4.3 Augmentacija podataka

Za trening skup koriscene su sledece augmentacije:
- Random horizontal flip (p=0.5)
- Random rotation (do 10 stepeni)
- Color jitter (brightness, contrast, saturation, hue)

Validacioni i test skupovi nisu augmentirani.

4.4 Eksperimenti

Planirano je 5 eksperimenata sa razlicitim konfiguracijama:
1. Baseline - osnovna konfiguracija
2. Veci model - vise filtera u konvolucionim slojevima
3. Veci dropout - dropout=0.7
4. Bez augmentacije
5. Visa learning rate (0.005)

4.5 Tok treniranja
Trening je praćen pomoću TensorBoard-a i tqdm progres bara. Najbolji model (po val_acc) automatski se čuva u models/best_model.pth. Krive učenja prikazane su na grafiku learning_curves.png.

5 Metrike na test skupu

Ukupna tacnost modela na test skupu: 67,37%

Per-class metrike:

**Validaciona tačnost (najbolji model):** 70.45%  

| Klasa | Precision | Recall | F1-score | Broj primera |
|---|---|---|---|---|
| cardboard | 0.89 | 0.82 | **0.85** | 61 |
| paper | 0.77 | 0.82 | **0.79** | 89 |
| plastic | 0.71 | 0.61 | **0.66** | 72 |
| glass | 0.50 | 0.79 | **0.61** | 75 |
| metal | 0.58 | 0.40 | **0.48** | 62 |
| trash | 0.83 | 0.24 | **0.37** | 21 |

### Zbirne metrike

| Metrika | Precision | Recall | F1-score |
|---|---|---|---|
| Macro avg | 0.71 | 0.61 | 0.63 |
| Weighted avg | 0.70 | 0.67 | 0.67 |

Svi grafici (learning curves, confusion matrix, ROC, PR) nalaze se u folderu `results/plots_full/`.

---

##  Analiza rezultata

**Najbolje klasifikovane klase:**
- **cardboard** – F1: 0.85 (najbolji rezultat)
- **paper** – F1: 0.79
- **plastic** – F1: 0.66

**Klase sa slabijim rezultatima:**
- **trash** – F1: 0.37 (mali broj primera, ali značajno poboljšanje u odnosu na prethodnu iteraciju)
- **metal** – F1: 0.48 (vizuelno preklapanje sa *glass* i *plastic*)

**Ključni napredak nakon 30 epoha:**
- Ukupna tačnost: **58.42% → 67.37%** (+8.95%)
- Klasa `trash` prvi put prepoznata (F1: 0.00 → 0.37)
- Klasa `cardboard` poboljšana sa F1 0.81 na 0.85


6. Front-end aplikacija
Za interakciju sa modelom razvijena je web aplikacija pomoću Streamlit biblioteke.

6.1 Funkcionalnosti
Aplikacija (app.py) omogućava:
Upload slike (JPG, JPEG, PNG) preko drag-and-drop interfejsa ili pretraživača fajlova
Prikaz uploadovane slike
Predikciju klase sa pouzdanošću (confidence)
Prikaz verovatnoća po svim klasama pomoću progress bar-ova

6.2 Kako radi
Aplikacija učitava istrenirani model (models/best_model.pth) pri pokretanju pomoću @st.cache_resource dekoratora (model se učitava samo jednom).
Uploadovana slika se transformiše (resize na 224×224, normalizacija), propušta kroz model, a izlaz se prevodi u verovatnoće pomoću softmax funkcije.

6.3 Pokretanje aplikacije lokalno
streamlit run app.py
Aplikacija je dostupna na http://localhost:8502.

7. Docker kontejner
7.1 Dockerfile
Aplikacija je pakovana u Docker sliku pomoću Dockerfile-a koji:
Koristi python:3.12-slim kao baznu sliku
Instalira sistemske zavisnosti (libgl1, libglib2.0-0) potrebne za Pillow
Instalira Python zavisnosti iz requirements-docker.txt (minimalni set – bez pywinpty, tensorflow, jupyter koji su Windows-only)
Kopira source kod (src/, models/, app.py)
Eksponira port 8501
Pokreće Streamlit aplikaciju


7.2 Build i pokretanje
Build slike:
docker build -t garbage-classifier .

Pokretanje kontejnjera:
docker run -d -p 8502:8501 --name garbage-app garbage-classifier

7.3 Prednosti kontejnerizacije
Reproduktivnost: aplikacija radi identično na svakoj mašini koja ima Docker
Izolacija: zavisnosti su izolovane od sistema domaćina
Prenosivost: slika se može lako podeliti i pokrenuti bilo gde

8 Zakljucak

U ovom radu razvijen je CNN model za klasifikaciju sest vrsta otpada.
Model postize 58.42% tacnosti na test skupu, sto je znatno iznad
slucajnog pogadjanja (16.67%).

Najbolje klasifikovane su klase cardboard (F1: 0.81) i paper (F1: 0.70).
Najveci izazov predstavlja klasa trash, koja ima mali broj primera
u dataset-u i nije prepoznata od strane modela.


