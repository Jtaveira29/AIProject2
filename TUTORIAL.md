# Tutorial — Projeto IART (Injury Risk Prediction)

## 1. O que tens neste pacote

| Ficheiro | Para quê |
|---|---|
| `generate_injury_data.py` | Gera o dataset artificial (corre uma vez) |
| `dataset.csv` | Dataset pronto a usar (output do script acima) |
| `groundtruth.csv` | Dataset + p_true escondida — só para validar o modelo |
| `train_models.py` | Treina e avalia os 3 modelos |
| `roc_curves.png` | Para a slide 8 |
| `calibration.png` | Para a slide 8 |
| `feature_importance.png` | Para a slide 8 |
| `predicted_vs_true.png` | Para a slide 8 |

## 2. Como reproduzir

```bash
# Instalar dependências (uma vez)
pip install numpy pandas scipy scikit-learn matplotlib xgboost

# Gerar os dados (já feito, mas se quiseres re-gerar com outra seed)
python generate_injury_data.py

# Treinar e avaliar
python train_models.py
```

Os scripts são todos `seed=42` → resultados reprodutíveis.

## 3. O que falta fazer

1. Preencher o PowerPoint (guia abaixo, slide a slide)
2. **Construir o web app** que ilustra a aplicação do modelo (Streamlit, ~50-80 linhas) — pendente
3. Submeter:
   - PowerPoint
   - Pasta com código (`generate_injury_data.py`, `train_models.py`, web app)

---

## Conteúdo para cada slide

> **Convenção:** O template está em inglês, portanto sugiro manter as slides em inglês. Os comentários estão em português. Adapta a linguagem conforme o que o teu professor prefere.

---

### Slide 1 — Cover

**Conteúdo:**
- Project 2 — IART
- L.EIC — FEUP
- Nomes e números mecanográficos do grupo
- Data

---

### Slide 2 — Customer

**Bullets sugeridos:**
- **Tottenham Hotspur FC** — Premier League (Inglaterra)
- One of England's most prominent clubs, currently navigating the 2025/26 season under significant disruption
- **Contact person:** Head of Sports Science / Performance Director
- Responsibility: monitor training load, recovery, and reduce injury incidence

**O que dizer (notas para a apresentação):**

Escolhemos o Tottenham por ser um caso real, atual e mediático. Na época 2025/26 o clube enfrenta uma crise de lesões sem precedentes (James Maddison, Romero, Xavi Simons e Odobert com LCA, entre muitos outros), o que levou ex-internacionais como Glenn Hoddle e Karen Carney a questionar publicamente o trabalho do departamento médico. O "contact person" assumido é o Head of Sports Science — a pessoa que tipicamente lidera departamentos de monitorização de carga e prevenção de lesões em clubes deste nível.

> ⚠️ **Truque:** Se o professor perguntar "porquê o Tottenham e não um clube nacional?", responde: *"Escolhemos um caso público e mediático para tornar o problema imediatamente reconhecível. A metodologia generaliza para qualquer clube com infraestrutura de monitorização."*

---

### Slide 3 — Problem

**Bullets sugeridos:**
- Spurs have suffered **15+ long-term injuries** in the 2025/26 season alone (Maddison, Romero, Simons, Odobert — all ACL)
- Pundits and ex-players (Hoddle, Carney) have publicly questioned the medical staff
- **Cost:** sporting (relegation battle, 3 head coaches), financial (~£45M/year on absent players in elite clubs)
- **Current approach:** reactive medical team — monitors GPS/RPE/wellness data, but no systematic *predictive* model ranking risk weekly per player

**O que dizer:**

O problema não é só o número de lesões — é a **falta de antecipação**. O staff médico trata bem quando o jogador já se lesionou, mas a literatura mostra que muitas lesões musculares podiam ser evitadas com gestão proactiva da carga. A nossa proposta é dar ao staff uma ferramenta semanal que prioriza quem está na zona de maior risco.

---

### Slide 4 — Available Data

**Bullets sugeridos:**

The club already collects (or could collect) the following data per player:

- **Medical history** — past injuries, severity, days out (every Tier-1 club has this)
- **Match data** — minutes played, games, competition (already collected by analysts)
- **GPS / wearable data** — distance, sprints, high-intensity accelerations from training/matches (Catapult, STATSports widely deployed)
- **Wellness questionnaires** — daily sleep, fatigue, pain, mood (subjective but standard)
- **Calendar / schedule** — rest intervals between competitive games
- **Anthropometric** — age, weight, height, position, dominant foot

**O que dizer:**

A maior parte destes dados *já existe* nos clubes profissionais — não precisamos de implementar coleta nova. O nosso valor está em integrar e analisar, não em medir.

---

### Slide 5 — Solution Overview

**Bullets sugeridos:**

A weekly injury risk dashboard for the performance/medical staff:

- **Predictive model:** outputs probability of injury in the next 7 days per player
- **Web application:** dashboard with player list sorted by risk, colour-coded (green/yellow/red), drill-down per player
- **Workflow integration:** runs every Monday morning, feeds the weekly training plan
- **Not in scope:** GPS hardware, wellness app collection layer (assumed existing)
- **Future:** could integrate with calendar APIs (training schedule), automated alerts

**O que dizer:**

A nossa POC entrega o componente *predictivo*. A integração com infraestrutura existente (Catapult API, wellness apps tipo PMSys, calendário do clube) é arquitetura standard e fica para a fase de produção. O enunciado diz claramente que só temos de desenvolver o modelo + web app demonstrativa.

---

### Slide 6 — Predictive Model

**Bullets sugeridos:**

- **Target:** binary "did the player sustain an injury in the following 7 days?" — model outputs the **probability** (0-100%) via `predict_proba`
- **Type:** binary classification with probabilistic output (often colloquially called "regression of probability")
- **Features (8):**
  1. `age`
  2. `position` (GK/DEF/MID/FWD, one-hot encoded)
  3. `history_injuries_2_seasons`
  4. `minutes_last_week` — acute load
  5. `minutes_prior_3w` — chronic load (3 weeks before last)
  6. `games_last_14d` — competitive density
  7. `days_since_last_injury`
  8. `rest_days_until_next_game`
- **Inspired by the Acute:Chronic Workload Ratio (ACWR)** framework (Gabbett, 2016) used in elite sports science

**O que dizer:**

Esta separação entre carga aguda (última semana) e carga crónica (3 semanas anteriores) é literalmente o conceito ACWR usado por departamentos de performance no futebol profissional. Não inventámos do nada — é literatura estabelecida.

---

### Slide 7 — Empirical Study: Setup

**Bullets sugeridos:**

- **Synthetic dataset** generated to mimic Premier League statistics:
  - 500 players × 38 weeks = **19,000 player-weeks**
  - Calibrated to real PL stats: ~64 injuries/team/season, ~2.5 per player, ~752k total minutes
  - Hidden ground-truth `p_true` retained for validation
- **Train/Val/Test split: 70/15/15** by **player** (GroupShuffleSplit) — avoids data leakage
  - 350 / 75 / 75 players, zero overlap
- **3 algorithms** evaluated:
  - Logistic Regression (interpretable baseline)
  - Random Forest (non-linear, robust)
  - XGBoost (gradient boosting, SOTA on tabular)
- **Evaluation metrics** chosen for probabilistic prediction:
  - **AUC-ROC** — ranking discrimination
  - **Brier score** — probability quality
  - **Log-loss** — penalises overconfidence
  - **Calibration curves** — visual check of probability accuracy
  - *(not accuracy — misleading with 6.8% positive rate)*

**O que dizer:**

Os dois pontos a sublinhar: (1) o split por jogador é fundamental, senão o modelo memoriza jogadores específicos em vez de aprender padrões generalizáveis; (2) não usámos accuracy porque com 6.8% de positivos, prever sempre "não lesão" dá 93% de accuracy e zero valor de negócio.

---

### Slide 8 — Empirical Study: Results

**Layout:** tabela à esquerda, `roc_curves.png` à direita.

**Tabela:**

| Model | AUC-ROC | Brier | Log-loss |
|---|---|---|---|
| Logistic Regression | 0.766 | 0.0576 | 0.2159 |
| **Random Forest** | **0.789** | **0.0568** | **0.2099** |
| XGBoost | 0.783 | 0.0584 | 0.2153 |
| *Oracle (p_true)* | *0.800* | *0.0558* | *0.2049* |

**Bullets adicionais:**

- **Random Forest wins on all metrics**
- Non-linear models (RF, XGB) outperform Logistic Regression — captures interactions (age × load, post-injury bump)
- RF achieves **98.6% of theoretical ceiling** (Oracle AUC=0.800, RF AUC=0.789)
- Predicted vs true probability correlation: **0.915** for RF — model learned the latent risk function
- Performance aligns with real-world injury prediction literature (typical AUC 0.70-0.80)

**Visualizações para incluir:**
- `roc_curves.png` — comparação visual dos 3 modelos
- `calibration.png` — modelos bem calibrados (RF e LR especialmente)
- `feature_importance.png` — top features: minutes_last_week, history, days_since_inj
- `predicted_vs_true.png` (opcional) — scatter com correlação 0.915

**O que dizer:**

O ponto chave a comunicar é o **tecto teórico**. O Oracle dá AUC=0.800 e atingimos 0.789 — quase tudo o que era possível aprender. Os 0.011 que faltam são ruído irredutível (a `fragility_latent` que escondemos de propósito para criar um tecto realista). Modelos reais nesta literatura raramente passam 0.75 AUC, portanto estamos no nível profissional certo.

---

### Slide 9 — Stakeholders

**Bullets sugeridos:**

Roles to engage for successful deployment:

- **Sponsor:** Head of Performance / Sports Science Director
- **Clinical owner:** Head of Medical Department (validates outputs, integrates with rehab decisions)
- **Primary user:** First team coaching staff (consumes risk dashboard, adjusts training load)
- **Subject:** Players (consent, feedback on dashboard usability)
- **IT / Data Engineering:** owns data pipelines (GPS, wellness, schedules)
- **Performance Analysts:** monitor model drift, retrain regularly
- **Legal / DPO:** ensures GDPR / data protection compliance with sensitive medical data
- **Senior leadership** (Director of Football, CEO): budget approval, strategic alignment

---

### Slide 10 — Expected Challenges

**Bullets sugeridos:**

**Technical:**
- **Data quality:** missing wellness questionnaire entries (player compliance), GPS sensor gaps
- **Cold-start:** new signings arrive with limited history
- **Model drift:** training schedules, coaching philosophy evolve over seasons → retrain quarterly
- **Confounding training vs match injuries:** dataset must distinguish source
- **Latent confounders:** model can't capture player technique, biomechanics, nutrition

**Organizational:**
- **User resistance:** coaches may distrust algorithmic recommendations
- **Over-reliance:** false sense of security if a player is rated "green" and then gets injured
- **Player concerns:** fears of model affecting playing time, contract negotiations
- **Privacy / data governance:** medical and biometric data is GDPR Article 9 (special category)
- **Cultural fit:** integrating into Monday training planning routine takes time

---

### Slide 11 — Unavailable Data

**Bullets sugeridos:**

Additional data that would improve the model but is currently unavailable (or hard to obtain):

- **Genetic markers** — research stage, expensive, ethical sensitivities
- **Sleep quality** (objective, from wearables like Whoop, Oura) — adoption inconsistent
- **HRV (Heart Rate Variability)** — proxy for fatigue, requires daily monitoring
- **Psychological stress indicators** — survey-based, low compliance, subjective
- **Off-pitch lifestyle** — diet adherence, hydration, alcohol — privacy-sensitive
- **Pre-academy injury history** — youth players' early injuries often undocumented
- **Inter-club continuity** — when a player transfers, prior club's data is rarely shared

**Challenges to collecting:**

- Player consent and privacy
- Vendor lock-in (each wearable has its own API)
- Manual entry compliance (wellness questionnaires)
- Data harmonisation across vendors

---

### Slide 12 — Ethical Risks

**Bullets sugeridos:**

- **Impact on playing time and contracts:** high-risk classification may influence coaches' decisions, indirectly affecting careers
- **Player autonomy:** right to play despite elevated risk; informed consent matters
- **Bias:** model trained on elite-level data may not generalise to youth, women's football, or lower divisions
- **False security:** over-trusting a "green" score → underestimating real-time symptoms
- **Data sensitivity:** medical and biometric data require strict GDPR Article 9 handling
- **Power asymmetry:** clubs hold the model and the data; players are subject to its outputs
- **Pressure for early returns:** if model says "low risk", may push for faster return-to-play

**Mitigations to mention:**

- Model must be an *input* to clinical decisions, not a replacement
- Transparent communication of risk to players
- Regular bias audits across age, position, demographic
- Clear data governance and access controls

---

## Notas finais antes da apresentação

1. **Repetir 1 frase central por slide.** Não te percas em detalhes — cada slide tem *uma* mensagem.
2. **Slide 8 é o coração técnico.** Pratica explicar a tabela em ≤30 segundos.
3. **Antecipa as perguntas óbvias:**
   - "Porque é que usaram dados sintéticos?" → enunciado pede explicitamente; permite isolamento de variáveis
   - "Como sabem que a vossa simulação é realista?" → comparámos com taxas reais (752k minutos totais, 64 lesões/equipa)
   - "Porquê 3 modelos?" → para mostrar que não-linearidades importam (RF/XGB ganham à LR)
4. **Tempo total:** ~10 minutos para 12 slides → ~50s por slide em média. Slides 7 e 8 podem ser mais longas, slides 9-12 mais rápidas.
