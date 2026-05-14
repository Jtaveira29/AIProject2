"""
Gerador de dados artificiais para previsão de risco de lesão em jogadores de
futebol profissional.

Cada linha do dataset representa um par (jogador, semana). As features descrevem
o estado do jogador nessa semana, e o target binário 'y' indica se o jogador se
lesionou nos 7 dias seguintes.

Outputs:
  - dataset.csv:      features + label (o que o modelo vê durante treino)
  - groundtruth.csv:  inclui também p_true (a probabilidade "real" usada para
                      samplar o label). Usado para validar interpretabilidade
                      do modelo na apresentação. Não é dado ao modelo.

Cadeira: IART, FEUP L.EIC.
"""

import numpy as np
import pandas as pd
from scipy.stats import truncnorm

# ============================================================================
# CONFIGURAÇÃO
# ============================================================================
N_PLAYERS = 500          # 20 equipas × 25 jogadores (Premier League completa)
N_WEEKS   = 38           # época regular Premier League
SEED      = 42

POSITIONS         = ['GK', 'DEF', 'MID', 'FWD']
POSITION_WEIGHTS  = np.array([1, 4, 4, 3], dtype=float)
POSITION_WEIGHTS /= POSITION_WEIGHTS.sum()

# Minutos base por jogo, por posição (titular nominal). Reflete os minutos
# médios reais que um titular fixo joga na Premier League moderna (com
# substituições, GKs jogam 90, jogadores de campo geralmente 75-85).
# Multiplicados depois pelo factor individual de titularidade.
BASE_MIN_PER_GAME = {'GK': 90, 'DEF': 82, 'MID': 78, 'FWD': 72}

# Coeficientes da função de risco verdadeira (logit linear + termos não-lineares).
# Calibrados manualmente para taxa-base de lesão ~4-6% por semana de exposição.
BETA = {
    'intercept':           -4.0,     # taxa-base
    'age_centered':         0.07,    # (idade-26): linear, mais velho = mais risco
    'history':              0.40,    # preditor #1 na literatura real
    'fragility':            1.7,     # variável latente (não observável)
    'minutes_acute':        0.35,    # NOVO: carga aguda (última semana) — peso forte
    'minutes_chronic':      0.10,    # NOVO: carga crónica (3 semanas anteriores) — peso menor
    'games_14d':            0.32,    # densidade competitiva
    'rest_days':           -0.16,    # menos descanso → mais risco
    'recent_injury_bump':   0.60,    # NÃO-LINEAR: degrau quando voltou < 30d
    'age_x_minutes':        0.014,   # INTERAÇÃO: velho + carga = composto
    'fwd_bonus':            0.22,    # avançados arriscam mais
    'gk_penalty':          -0.50,    # GKs lesionam-se menos (sem sprints repetidos)
    'noise_sigma':          0.10,
}

rng = np.random.default_rng(SEED)


# ============================================================================
# HELPERS
# ============================================================================
def trunc_normal(mean, std, low, high):
    """Sample a single value from a truncated normal distribution."""
    a, b = (low - mean) / std, (high - mean) / std
    return float(truncnorm.rvs(a, b, loc=mean, scale=std, random_state=rng))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================================
# CAMADA 1: GERAR POPULAÇÃO DE JOGADORES (atributos estáveis)
# ============================================================================
def generate_players(n):
    players = []
    for i in range(n):
        age = trunc_normal(26, 4, 16, 40)
        position = str(rng.choice(POSITIONS, p=POSITION_WEIGHTS))

        # Fragilidade latente (não observável): Beta(2, 5) → maioria baixa,
        # cauda pequena de jogadores "frágeis"
        fragility = float(rng.beta(2, 5))

        # Histórico de lesões nas 2 épocas anteriores: Poisson com lambda
        # dependente da idade e da fragilidade latente
        lam = max(0.2, 0.8 + 0.08 * (age - 20) + 2.0 * fragility)
        history = int(rng.poisson(lam))

        players.append({
            'player_id':                   i,
            'age':                         age,
            'position':                    position,
            'fragility_latent':            fragility,    # escondido do modelo
            'history_injuries_2_seasons':  history,
        })
    return players


# ============================================================================
# CAMADA 2: SIMULAR ÉPOCA SEMANA-A-SEMANA
# ============================================================================
def simulate_player_season(player, n_weeks):
    rows = []

    # ---- Estado dinâmico inicial ----
    is_recovering       = False
    recovery_weeks_left = 0
    if player['history_injuries_2_seasons'] == 0:
        days_since_last_injury = 999
    else:
        days_since_last_injury = int(rng.integers(60, 365))

    # Tier de titularidade (uma vez por jogador). Reflete a hierarquia real
    # dum plantel de 25 jogadores a competir por 11 lugares no onze:
    #   ~44% titulares  → jogam quase sempre
    #   ~32% rotação    → jogam ~metade dos jogos
    #   ~24% banco/youth→ jogam pouco
    r = rng.random()
    if r < 0.44:
        starter_factor = float(np.clip(rng.normal(1.00, 0.08), 0.80, 1.10))
    elif r < 0.76:
        starter_factor = float(np.clip(rng.normal(0.50, 0.15), 0.25, 0.80))
    else:
        starter_factor = float(np.clip(rng.normal(0.15, 0.10), 0.00, 0.40))

    typ_min_per_game = min(90, BASE_MIN_PER_GAME[player['position']] * starter_factor)

    # Pre-época: 4 semanas anteriores de actividade para inicializar janelas
    minutes_history = []
    games_history = []
    for _ in range(4):
        if rng.random() < 0.08:                # 8% chance de estar lesionado pré-época
            minutes_history.append(0.0)
            games_history.append(0)
        else:
            g = 2 if rng.random() < 0.25 else 1
            m = float(np.clip(rng.normal(typ_min_per_game * g, 15), 0, 90 * g))
            minutes_history.append(m)
            games_history.append(g)

    # ---- Loop semanal ----
    for week in range(n_weeks):
        # Schedule da semana. Calibrado para média = 1.0 jogos/semana × 38
        # semanas = 38 jogos/época, que é o tamanho real da Premier League.
        # Semanas sem jogo: pausas internacionais (Set, Out, Nov, Mar) + FA Cup.
        # Semanas duplas: período natalício e cruzamento de competições.
        r = rng.random()
        if r < 0.12:
            games_scheduled = 0     # break / sem jogo
        elif r < 0.88:
            games_scheduled = 1
        else:
            games_scheduled = 2     # midweek + fim-de-semana

        # Actividade desta semana
        if is_recovering:
            minutes_this_week = 0.0
            games_played = 0
        else:
            minutes_this_week = float(np.clip(
                rng.normal(typ_min_per_game * games_scheduled, 15),
                0, 90 * games_scheduled
            ))
            games_played = games_scheduled

        minutes_history.append(minutes_this_week)
        games_history.append(games_played)

        # Features observáveis (janelas)
        # Decomposição da carga em DUAS janelas não-sobrepostas:
        #   - aguda: minutos na última semana (= esta semana)
        #   - crónica: minutos nas 3 semanas anteriores (semanas -4 a -2)
        # Inspirado no acute:chronic workload ratio (ACWR) da ciência do desporto.
        minutes_last_week = minutes_history[-1]
        minutes_prior_3w  = sum(minutes_history[-4:-1])
        games_last_14d    = sum(games_history[-2:])

        # Dias até ao próximo jogo. Depende do schedule atual:
        #   - pausa internacional/sem jogo: próximo jogo a 10-14 dias
        #   - semana dupla: jogou midweek, próximo a 3-4 dias
        #   - semana normal: 4-7 dias, distribuição realista
        if games_scheduled == 0:
            rest_days_next = int(rng.choice([10, 14], p=[0.5, 0.5]))
        elif games_scheduled == 2:
            rest_days_next = int(rng.choice([3, 4], p=[0.6, 0.4]))
        else:
            rest_days_next = int(rng.choice([4, 5, 6, 7], p=[0.20, 0.35, 0.30, 0.15]))

        # Probabilidade verdadeira e label
        if is_recovering:
            p_true = 0.0
            y = 0
            recovery_weeks_left -= 1
            if recovery_weeks_left <= 0:
                is_recovering = False
        else:
            logit = (
                BETA['intercept']
                + BETA['age_centered']       * (player['age'] - 26)
                + BETA['history']            * player['history_injuries_2_seasons']
                + BETA['fragility']          * player['fragility_latent']
                + BETA['minutes_acute']      * (minutes_last_week / 100)
                + BETA['minutes_chronic']    * (minutes_prior_3w  / 100)
                + BETA['games_14d']          * games_last_14d
                + BETA['rest_days']          * rest_days_next
                + BETA['recent_injury_bump'] * (1 if days_since_last_injury < 30 else 0)
                + BETA['age_x_minutes']      * (player['age'] - 26) * ((minutes_last_week + minutes_prior_3w) / 100)
                + BETA['fwd_bonus']          * (1 if player['position'] == 'FWD' else 0)
                + BETA['gk_penalty']         * (1 if player['position'] == 'GK'  else 0)
                + float(rng.normal(0, BETA['noise_sigma']))
            )
            p_true = float(sigmoid(logit))
            y = int(rng.random() < p_true)

        # Guardar a linha com features no estado em que estavam quando o label
        # foi sampled (NÃO depois das actualizações de estado)
        rows.append({
            'player_id':                   player['player_id'],
            'week':                        week,
            'age':                         round(player['age'], 1),
            'position':                    player['position'],
            'history_injuries_2_seasons':  player['history_injuries_2_seasons'],
            'minutes_last_week':           round(minutes_last_week, 0),
            'minutes_prior_3w':            round(minutes_prior_3w,  0),
            'games_last_14d':              games_last_14d,
            'days_since_last_injury':      min(days_since_last_injury, 999),
            'rest_days_until_next_game':   rest_days_next,
            'y':                           y,
            'p_true':                      round(p_true, 4),
        })

        # Actualizar estado para a semana seguinte
        if y == 1:
            # Lesionou-se. Sample duração de recuperação (log-normal: maioria
            # 2-4 semanas, cauda longa de lesões graves até 6 meses).
            recovery_days = int(np.clip(rng.lognormal(mean=2.8, sigma=0.7), 7, 180))
            recovery_weeks_left = max(1, recovery_days // 7)
            is_recovering = True
            days_since_last_injury = 7
        else:
            days_since_last_injury = min(days_since_last_injury + 7, 999)

    return rows


# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"Generating data: {N_PLAYERS} players × {N_WEEKS} weeks "
          f"= {N_PLAYERS * N_WEEKS} player-weeks expected\n")

    players = generate_players(N_PLAYERS)

    all_rows = []
    for p in players:
        all_rows.extend(simulate_player_season(p, N_WEEKS))

    df = pd.DataFrame(all_rows)

    # Separar groundtruth (com p_true) do dataset que o modelo vai ver
    groundtruth = df.copy()
    dataset     = df.drop(columns=['p_true'])

    # ---- Summary stats ----
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total rows:               {len(df):,}")
    print(f"Total unique players:     {df['player_id'].nunique()}")
    print(f"Injury rate (y=1):        {df['y'].mean():.3%}")
    print(f"Mean p_true:              {df['p_true'].mean():.4f}")
    print(f"Median p_true:            {df['p_true'].median():.4f}")
    print()
    print("Injury rate by position:")
    print(df.groupby('position')['y'].agg(['mean', 'sum', 'count']).round(4))
    print()
    print("Age distribution:")
    print(df['age'].describe().round(2))
    print()
    print("Minutes last week (acute load):")
    print(df['minutes_last_week'].describe().round(1))
    print()
    print("Minutes prior 3 weeks (chronic load):")
    print(df['minutes_prior_3w'].describe().round(1))
    print()

    # ---- Save ----
    dataset.to_csv('dataset.csv', index=False)
    groundtruth.to_csv('groundtruth.csv', index=False)
    print("Files written: dataset.csv, groundtruth.csv")


if __name__ == '__main__':
    main()
