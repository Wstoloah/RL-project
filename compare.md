# Comparaison DQN Custom vs SB3 DQN sur Highway-v0

Ce document compare notre DQN custom a une implementation Stable-Baselines3 sur
le benchmark partage. La comparaison presentee ici est une etude **"toutes choses
egales par ailleurs"**: les hyperparametres de SB3 ont ete deliberement alignes
sur ceux du DQN custom pour isoler l'effet des differences d'implementation
(optimiseur, batching des updates, ordre des operations internes) plutot que
l'effet des hyperparametres eux-memes.

Une comparaison complementaire utilisant les hyperparametres **recommandes par
defaut pour SB3 sur highway-env** sera ajoutee dans un second notebook pour
distinguer ces deux effets.

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
| Gradient clipping | clip_grad_value=100 | max_grad_norm=100 |
| Epsilon decay | exponentiel (eps_decay=5000) | exponentiel (eps_decay=5000) |
| Volume d'entrainement | 2 500 episodes | 2 500 episodes |
| Parallelisation | 1 env | 1 env |
| Train frequency | chaque step | tous les 4 steps |
| Gradient steps | 1 par step | 4 tous les 4 steps |

**Differences cles:**

1. **Optimiseur**: le DQN custom utilise AdamW avec amsgrad=True, tandis que SB3 utilise Adam standard. AdamW offre une meilleure regularisation via le weight decay decoupled, et amsgrad garantit des taux d'apprentissage non-croissants, ce qui stabilise l'entrainement.

2. **Train frequency et gradient steps**: le DQN custom effectue 1 gradient step a chaque step d'environnement. SB3 collecte 4 steps puis effectue 4 gradient steps d'un coup. Le ratio global (1 update par step) est identique, mais le batching de SB3 reduit l'overhead Python/CUDA au prix d'updates moins frequents.

3. **Implementation interne**: SB3 gere automatiquement le replay buffer, le target network sync, et l'epsilon schedule via son propre pipeline, ce qui peut introduire des differences subtiles dans l'ordre des operations (ex: quand exactement le target network est mis a jour par rapport aux gradient steps).

## 3. Resultats d'evaluation

Evaluation deterministe, 50 episodes par seed, 3 seeds par methode.

### Resultats agreges

| Metrique | DQN Custom (gamma=0.99) | SB3 DQN |
|----------|------------------------|---------|
| Mean reward (cross-seed) | **20.20 +/- 0.45** | 15.79 +/- 3.95 |
| Crash rate | **2.67%** | 42.67% |
| Mean episode length | **29.47** | 22.03 |

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
| 0 | 17.32 | 6.54 | 24.90 | 26% |
| 1 | 10.37 | 6.47 | 12.42 | 96% |
| 2 | 19.67 | 3.53 | 28.76 | 6% |

Sous cette configuration alignee, le DQN custom obtient une meilleure moyenne
et surtout une bien meilleure consistance cross-seed (std 0.45 vs 3.95). Il
faut toutefois lire ce tableau avec prudence: le seed 1 de SB3 ne converge pas
(96% crash rate) et tire fortement la moyenne vers le bas. Si on l'exclut,
SB3 fait ~18.5 de reward moyenne avec un crash rate d'environ 16%, ce qui reste
inferieur au custom mais reduit l'ecart de maniere significative. Avec seulement
3 seeds, un unique echec de convergence suffit a biaiser l'agregation; un run
avec davantage de seeds serait necessaire pour conclure de maniere robuste.

## 4. Analyse du comportement appris

Le DQN custom converge systematiquement vers une **politique conservatrice stable**: l'agent roule a vitesse basse (~20 m/s), ne change jamais de voie, et survit la quasi-totalite des episodes. La reward par episode non-crash est remarquablement constante (~20.45).

Le SB3 DQN montre un comportement plus heterogene selon les seeds:
- **Seed 2** converge vers la meme politique conservatrice que le DQN custom (crash rate 6%, reward ~20)
- **Seed 0** adopte une politique partiellement stable mais avec des crashes reguliers (26%), suggerant que l'agent n'a pas completement appris a eviter les collisions
- **Seed 1** echoue completement (96% crash rate), indiquant que l'entrainement n'a pas converge vers une politique viable

Ce comportement conservateur s'explique par la structure de reward:
- La penalite de collision (-1.5) domine largement la recompense de vitesse (0.7)
- Changer de voie coute -0.02 par action sans benefice clair
- La plage de vitesse recompensee [22, 30] est etroite; rouler juste en dessous (20 m/s) offre un ratio risque/recompense optimal

L'ecart entre les deux implementations suggere que, meme lorsque la politique
cible est simple, la fiabilite de la convergence depend des details
d'implementation (optimiseur, ordre des updates, batching). Cette observation
reste cependant a confirmer: avec seulement 3 seeds et une config SB3
deliberement bridee, elle constitue une hypothese plutot qu'une conclusion.

## 5. Analyse des modes d'echec

### Crashes SB3

Sur 150 episodes d'evaluation (3 seeds x 50), 64 se terminent en collision (43%):

- **Seed 0**: 13 crashes (26%) — l'agent a partiellement appris a eviter les collisions mais reste vulnerable. Les crashes montrent un melange de collisions precoces (reward ~2-4) et de collisions tardives (reward ~11-18).
- **Seed 1**: 48 crashes (96%) — echec quasi-total de l'entrainement. L'agent n'a pas converge vers une politique viable et crashe systematiquement.
- **Seed 2**: 3 crashes (6%) — seul seed ou SB3 atteint un niveau comparable au DQN custom.

Analyse detaillee des crashes du meilleur seed (seed 2, 9/100 episodes de failure analysis):
- 1 crash precoce (step 7, reward 4.09)
- 8 crashes tardifs (steps 12-22, rewards 8.69-16.82), indiquant un agent qui survit mais se fait rattraper par le trafic

### Crashes DQN Custom (gamma=0.99)

Le DQN custom montre un profil radicalement different avec seulement 4 crashes sur 150 episodes (2.67%):
- Seed 0: 1 crash (reward: 15.80, step ~16)
- Seed 1: 3 crashes (rewards: 2.28, 2.16, 4.89)
- Seed 2: 0 crashes

Les crashes du seed 1 montrent des collisions tres precoces (steps 2-5), suggerant des scenarios de trafic initiaux defavorables plutot qu'un defaut de la politique.

### Cause commune et differences

Le mode d'echec partage est le **trafic dense initial**: quand des vehicules rapides arrivent par derriere dans la voie de l'agent, celui-ci n'a pas le reflexe de changer de voie pour esquiver (car les changements de voie sont penalises).

La difference majeure est que, sous cette configuration alignee, le DQN custom
apprend de maniere fiable a eviter les collisions sur les 3 seeds, tandis que
SB3 echoue sur 1 seed sur 3 (seed 1) et reste vulnerable sur un autre (seed 0).
Deux hypotheses peuvent l'expliquer: (i) l'optimiseur (AdamW+amsgrad custom vs
Adam SB3), (ii) le batching des updates (1 gradient step par step d'env dans
le custom vs 4 updates groupes tous les 4 steps dans SB3). Ces hypotheses
n'ont pas ete testees isolement: il faudrait injecter AdamW dans SB3 et/ou
passer SB3 en `train_freq=1` pour les departager.

## 6. Discussion

### Ce que cette comparaison alignee mesure (et ce qu'elle ne mesure pas)

Les hyperparametres (architecture, gamma, buffer, epsilon, target update,
volume d'entrainement) sont identiques entre les deux approches. Ce qui
**reste different** est ce que l'on peut raisonnablement appeler les "details
d'implementation":

- optimiseur (AdamW+amsgrad custom vs Adam SB3)
- batching des gradient steps (1 par step d'env vs 4 groupes tous les 4 steps)
- ordre interne des operations (quand exactement le target net est
  synchronise par rapport aux updates, gestion du buffer, etc.)

Les resultats de cette section doivent donc se lire comme **un test de la
sensibilite aux details d'implementation**, pas comme une evaluation
intrinseque de SB3. SB3 est concu pour etre utilise avec ses hyperparametres
recommandes (pour highway-env: `train_freq=1`, epsilon lineaire sur une
fraction significative de l'entrainement, etc.), et le forcer a imiter
exactement le DQN custom lui retire une partie de ses garanties.

### Hypotheses explicatives (non testees isolement)

Sous cette configuration alignee, le DQN custom converge plus fiablement. Deux
hypotheses non exclusives peuvent l'expliquer, mais aucune n'a ete testee en
isolation dans cette etude:

1. **AdamW+amsgrad vs Adam**: AdamW decoupled le weight decay de l'update, et
   amsgrad garantit des taux d'apprentissage effectifs non-croissants. Ces deux
   elements peuvent stabiliser la convergence. Pour tester cette hypothese il
   faudrait injecter AdamW dans SB3 via `policy_kwargs={"optimizer_class":
   torch.optim.AdamW, "optimizer_kwargs": {"amsgrad": True}}` et refaire
   tourner les 3 seeds.

2. **Frequence des updates**: dans le custom, chaque step d'environnement
   declenche immediatement un update, ce qui permet au reseau de reagir plus
   vite aux nouvelles experiences. SB3 groupe 4 steps puis fait 4 updates d'un
   coup. Pour tester cette hypothese, il suffirait de repasser SB3 en
   `train_freq=1, gradient_steps=1`.

Une **troisieme explication possible**, plus simple, est que la variance
cross-seed de SB3 est juste plus elevee par hasard dans cette config et qu'un
echantillon plus large de seeds rapprocherait les deux methodes.

### Limites de la comparaison

- **Outlier seed 1**: l'echec complet du seed 1 (96% crash rate) tire fortement
  la moyenne SB3 vers le bas. Sans ce seed, SB3 fait ~18.5 de reward pour
  ~16% de crash rate, soit un ecart bien plus modere avec le custom. Avec
  seulement 3 seeds, un unique outlier peut basculer la conclusion; un run
  5-10 seeds serait necessaire pour statuer.
- **Hyperparametres "bridant" SB3**: en alignant SB3 sur la config custom,
  on l'eloigne de ses defauts recommandes. Une comparaison equitable des deux
  **outils** (et non des deux implementations sous contrainte d'alignement)
  necessite de laisser SB3 tourner avec ses hyperparams natifs. Ce run
  complementaire fera l'objet d'un second notebook.
- **Hypotheses non testees**: les arguments AdamW et batching sont plausibles
  mais non valides experimentalement dans cette etude.
- **Gamma**: le DQN custom a beneficie d'une ablation sur gamma (0.8 / 0.95 /
  0.99) qui a permis de selectionner la meilleure valeur. SB3 a ete
  directement configure avec gamma=0.99 sans tuning independant.

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
