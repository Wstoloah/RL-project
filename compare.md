# Comparaison DQN Custom vs SB3 DQN sur Highway-v0

## 1. Benchmark commun

Les deux approches sont entrainees et evaluees sur le meme environnement **highway-v0** avec la configuration partagee (`shared_core_config.py`):

- **Observation**: Kinematics (10 vehicules, features: presence, x, y, vx, vy) -> vecteur de dimension 50
- **Actions**: DiscreteMetaAction avec 3 vitesses cibles (20, 25, 30 m/s)
- **Reward**: collision=-1.5, high_speed=0.7, lane_change=-0.02, right_lane=0.0
- **Environnement**: 4 voies, 45 vehicules, duree=30 steps, collision terminale

## 2. Comparaison des hyperparametres

| Parametre | DQN Custom | SB3 DQN |
|-----------|-----------|---------|
| Architecture | MLP 256-256 (ReLU) | MLP 256-256 (ReLU) |
| Learning rate | 5e-4 (AdamW, amsgrad) | 5e-4 (Adam) |
| Gamma | 0.99 | 0.99 |
| Buffer size | 15 000 | 15 000 |
| Batch size | 32 | 32 |
| Target update | hard, tous les 50 steps | hard, tous les 50 steps |
| Learning starts | 200 steps | 200 steps |
| Loss | Smooth L1 (Huber) | Smooth L1 (Huber) |
| Gradient clipping | clip_grad_value=100 | default SB3 |
| **Epsilon decay** | **exponentiel** (eps_decay=5000) | **lineaire** (fraction=0.5) |
| **Volume d'entrainement** | **2 500 episodes** | **75 000 timesteps (8 envs paralleles)** |
| **Parallelisation** | 1 env | 8 envs (DummyVecEnv) |
| Train frequency | chaque step | tous les 4 steps |
| Gradient steps | 1 | 8 |

**Differences cles:**

1. **Epsilon schedule**: le DQN custom utilise un decay exponentiel (`eps = 0.05 + 0.95 * exp(-steps/5000)`), tandis que SB3 utilise un decay lineaire sur 50% du training. Le decay exponentiel est plus agressif au debut mais converge plus vite vers l'exploitation.

2. **Parallelisation**: SB3 utilise 8 environnements en parallele, ce qui diversifie les experiences collectees dans le buffer et accelere l'entrainement en wall-clock time.

3. **Gradient steps**: SB3 effectue 8 gradient steps par update (contre 1 pour le custom), ce qui extrait plus d'information de chaque batch mais peut provoquer de l'overfitting sur le buffer.

## 3. Resultats d'evaluation

Evaluation deterministe, 50 episodes par seed, 3 seeds par methode.

### Resultats agrages

| Metrique | DQN Custom (gamma=0.99) | SB3 DQN |
|----------|------------------------|---------|
| Mean reward (cross-seed) | **20.20 +/- 0.45** | 19.81 +/- 0.33 |
| Crash rate | **2.67%** | 6.0% |
| Mean episode length | **29.47** | 29.0 |

### Detail par seed

**DQN Custom (gamma=0.99):**

| Seed | Mean Reward | Std Reward | Mean Length | Crash Rate |
|------|-------------|-----------|-------------|-----------|
| 0 | 20.49 | 0.69 | 29.88 | 2% |
| 1 | 19.57 | 4.21 | 28.52 | 6% |
| 2 | **20.55** | **0.14** | **30.00** | **0%** |

**SB3 DQN:**

| Seed | Mean Reward | Std Reward | Mean Length | Crash Rate |
|------|-------------|-----------|-------------|-----------|
| 0 | 20.10 | 2.48 | 29.18 | 6% |
| 1 | 19.35 | 3.84 | 28.46 | 8% |
| 2 | 19.98 | 2.42 | 29.34 | 4% |

Le DQN custom avec gamma=0.99 obtient des resultats legerement superieurs en reward moyenne et nettement meilleurs en crash rate. Notamment, le seed 2 du DQN custom atteint un crash rate de 0% avec une variance quasi nulle (std=0.14), indiquant une politique completement stable.

## 4. Analyse du comportement appris

Les deux modeles convergent vers la **meme politique conservatrice**: l'agent roule a vitesse basse (~20 m/s), ne change jamais de voie, et survit la quasi-totalite des episodes.

Ce comportement s'explique par la structure de reward:
- La penalite de collision (-1.5) domine largement la recompense de vitesse (0.7)
- Changer de voie coute -0.02 par action sans benefice clair
- La plage de vitesse recompensee [22, 30] est etroite; rouler juste en dessous (20 m/s) offre un ratio risque/recompense optimal

La reward par episode non-crash est remarquablement constante (~20.45), ce qui confirme que les deux agents ont appris la meme strategie "idle". Les rares variations au-dessus de 20.45 (ex: 26.19 pour SB3 seed 0) correspondent a des episodes ou l'agent a brievement accelere sans consequence.

## 5. Analyse des modes d'echec

### Crashes SB3

Sur 150 episodes d'evaluation (3 seeds x 50), 9 se terminent en collision (6%):

- Seed 0: 3 crashes (rewards: 7.39, 10.00, 26.19*)
- Seed 1: 4 crashes (rewards: 4.77, 6.14, 11.59, 4.09)
- Seed 2: 2 crashes (rewards: 11.59, 5.45)

*Note: le reward de 26.19 (seed 0) indique un crash tardif apres une longue survie avec acceleration.

Les crashes a faible reward (4-7) correspondent a des collisions precoces (avant step 10), probablement causees par des configurations initiales de trafic dense ou un vehicule arrive par derriere a grande vitesse. L'agent, adoptant une vitesse basse, ne peut pas eviter le choc.

### Crashes DQN Custom (gamma=0.99)

Le DQN custom montre un profil similaire mais avec moins de crashes (4 sur 150 episodes):
- Seed 0: 1 crash (reward: 15.80, step ~16)
- Seed 1: 3 crashes (rewards: 2.28, 2.16, 4.89)
- Seed 2: 0 crashes

Les crashes du seed 1 montrent des collisions tres precoces (steps 2-5), suggerant des scenarios de trafic initiaux defavorables.

### Cause commune

Le mode d'echec principal est le **trafic dense initial**: quand des vehicules rapides arrivent par derriere dans la voie de l'agent, celui-ci n'a pas le reflexe de changer de voie pour esquiver (car les changements de voie sont penalises). La politique "rester immobile" est optimale en moyenne mais vulnerable a ces scenarios specifiques.

## 6. Discussion

### Pourquoi le DQN custom est legerement meilleur ?

Malgre des hyperparametres presque identiques, le DQN custom (gamma=0.99) surpasse legerement le SB3:

1. **Epsilon decay exponentiel vs lineaire**: le decay exponentiel permet une transition plus douce vers l'exploitation, potentiellement mieux adaptee a cet environnement simple.

2. **AdamW avec amsgrad**: le DQN custom utilise AdamW avec amsgrad=True, offrant une meilleure stabilite des gradients comparee au Adam standard de SB3.

3. **Volume d'entrainement effectif**: 2500 episodes sur 1 env vs 75K timesteps sur 8 envs representent des quantites d'experience comparables, mais la collecte sequentielle du DQN custom produit des episodes plus coherents dans le buffer.

### Limites de la comparaison

- Les deux approches convergent vers la meme politique "triviale" dictee par la reward. Une comparaison plus discriminante necessiterait une reward favorisant davantage la vitesse ou les depassements.
- La difference de crash rate (2.67% vs 6%) est basee sur un petit nombre absolu de crashes (4 vs 9 sur 150 episodes) et pourrait ne pas etre statistiquement significative.
- Le DQN custom a beneficie d'une etude d'ablation sur gamma (testant 0.8, 0.95, 0.99) qui a permis de selectionner le meilleur gamma, alors que SB3 a ete directement configure avec gamma=0.99.

### Impact du discount factor (etude d'ablation DQN custom)

L'etude d'ablation sur gamma revele un effet dramatique:

| Gamma | Mean Reward | Crash Rate |
|-------|-----------|-----------|
| 0.80 | 18.23 +/- 1.25 | 74.0% |
| 0.95 | 15.18 +/- 2.67 | 76.0% |
| 0.99 | **20.20 +/- 0.45** | **2.67%** |

Avec gamma=0.8, l'agent ne valorise pas suffisamment les rewards futures et adopte un comportement myope, provoquant des collisions frequentes. Gamma=0.99 permet a l'agent de planifier sur l'horizon de l'episode (30 steps) et de comprendre que la survie a long terme est plus rentable.

### Reward shaping et comportement

La configuration de reward partagee produit inevitablement un agent passif. Pour obtenir un comportement plus proche de la conduite humaine (depassements, gestion de voie), il faudrait:
- Augmenter `high_speed_reward` (>1.0)
- Reduire `collision_reward` en valeur absolue
- Supprimer la penalite de lane change
- Ajouter un `right_lane_reward` positif

Cependant, cette configuration est imposee par le projet, et les deux implementations l'exploitent correctement.
