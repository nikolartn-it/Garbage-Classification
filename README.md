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


3. METODOLOGIJA

3.1 Arhitektura modela

Koriscen je CNN model sa sledecom arhitekturom:

- Conv2d(3 -> 32, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool
- Conv2d(32 -> 64, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool
- Conv2d(64 -> 128, kernel=3, padding=1) + BatchNorm + ReLU + MaxPool
- Conv2d(128 -> 256, kernel=3, padding=1) + BatchNorm + ReLU + AdaptiveAvgPool
- Dropout(0.5)
- Linear(256 -> 6)

Model ima ukupno 390.918 parametara.

3.2 Hiperparametri

- Broj epoha: 10
- Batch size: 32
- Learning rate: 0.001
- Weight decay: 0.0001
- Optimizator: Adam
- Loss funkcija: CrossEntropyLoss
- Scheduler: StepLR (step=10, gamma=0.1)

3.3 Augmentacija podataka

Za trening skup koriscene su sledece augmentacije:
- Random horizontal flip (p=0.5)
- Random rotation (do 10 stepeni)
- Color jitter (brightness, contrast, saturation, hue)

Validacioni i test skupovi nisu augmentirani.

3.4 Eksperimenti

Planirano je 5 eksperimenata sa razlicitim konfiguracijama:
1. Baseline - osnovna konfiguracija
2. Veci model - vise filtera u konvolucionim slojevima
3. Veci dropout - dropout=0.7
4. Bez augmentacije
5. Visa learning rate (0.005)


4. REZULTATI

4.1 Metrike na test skupu

Ukupna tacnost modela na test skupu: 58.42%

Per-class metrike:

| Klasa | Precision | Recall | F1-score | Broj primera |
|-------|-----------|--------|----------|--------------|
| cardboard | 0.84 | 0.79 | 0.81 | 61 |
| glass | 0.45 | 0.63 | 0.53 | 75 |
| metal | 0.50 | 0.11 | 0.18 | 62 |
| paper | 0.60 | 0.83 | 0.70 | 89 |
| plastic | 0.56 | 0.64 | 0.60 | 72 |
| trash | 0.00 | 0.00 | 0.00 | 21 |

| Metrika | Vrednost |
|---------|----------|
| Macro avg | 0.49 (P), 0.50 (R), 0.47 (F1) |
| Weighted avg | 0.55 (P), 0.58 (R), 0.54 (F1) |

4.2 Analiza rezultata

Najbolji rezultati su postignuti za klase:
1. cardboard - F1-score: 0.81 (odlicno)
2. paper - F1-score: 0.70 (dobro)
3. plastic - F1-score: 0.60 (solidno)

Klase sa slabijim rezultatima:
1. trash - F1-score: 0.00 (neprepoznavanje)
2. metal - F1-score: 0.18 (veoma nizak recall)

Razlozi za slabije rezultate:
- Klasa trash ima samo 21 primer u test skupu
- Klasa metal ima specificne vizuelne karakteristike koje se preklapaju
  sa drugim klasama
- Mali broj primera za trash klasu u trening skupu (137 slika)





U ovom radu razvijen je CNN model za klasifikaciju sest vrsta otpada.
Model postize 58.42% tacnosti na test skupu, sto je znatno iznad
slucajnog pogadjanja (16.67%).

Najbolje klasifikovane su klase cardboard (F1: 0.81) i paper (F1: 0.70).
Najveci izazov predstavlja klasa trash, koja ima mali broj primera
u dataset-u i nije prepoznata od strane modela.


