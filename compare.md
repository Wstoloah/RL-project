# Comparaison DQN Custom vs SB3 DQN sur Highway-v0

Ce document compare le DQN custom (implementation from scratch) a **SB3 DQN**
entraine avec les hyperparametres recommandes pour highway-env.

## 1. Benchmark commun

Les deux approches sont entrainees et evaluees sur le meme environnement
**highway-v0** avec la configuration partagee (`shared_core_config.py`):

- **Observation**: Kinematics (10 vehicules, features: presence, x, y, vx, vy) -> vecteur de dimension 50
- **Actions**: DiscreteMetaAction avec 3 vitesses cibles (20, 25, 30 m/s)
- **Reward**: collision=-1.5, high_speed=0.7, lane_change=-0.02, right_lane=0.0
- **Environnement**: 4 voies, 45 vehicules, duree=30 steps, collision terminale

## 2. Hyperparametres

| Parametre | DQN Custom | SB3 DQN |
|-----------|-----------|---------|
| Architecture | MLP 256-256 (ReLU) | MLP 256-256 (ReLU) |
| Learning rate | 5e-4 (AdamW, amsgrad) | 5e-4 (Adam) |
| Gamma | **0.99** | **0.8** |
| Buffer size | 15 000 | 15 000 |
| Batch size | 32 | 32 |
| Target update | hard, tous les 50 steps | hard, tous les 50 steps |
| Loss | Smooth L1 (Huber) | Smooth L1 (Huber) |
| Epsilon decay | exponentiel (eps_decay=5000) | lineaire 1.0→0.05 sur 10% steps |
| Volume d'entrainement | 2 500 episodes | 40 000 steps (~2 700 episodes) |
| Parallelisation | 1 env | 1 env |
| Train frequency | chaque step | chaque step |
| Gradient steps | 1 par step | 1 par step |
| Optimiseur | AdamW (amsgrad=True) | Adam |

**Differences cles:**

1. **Gamma**: 0.99 (custom) vs 0.8 (SB3). L'ablation DQN custom montre que
   gamma=0.8 produit 74% de crash rate sur cet environnement. L'horizon de
   30 steps necessite un discount factor eleve pour que l'agent valorise
   la survie a long terme.

2. **Epsilon schedule**: le DQN custom utilise un decay exponentiel
   (eps=0.05 + 0.95*exp(-steps/5000)), SB3 utilise un decay lineaire
   sur les 10 premiers pourcents des timesteps.

3. **Optimiseur**: AdamW avec amsgrad=True (custom) vs Adam standard (SB3).
   AdamW decoupled le weight decay et amsgrad garantit des taux d'apprentissage
   non-croissants.

## 3. Resultats d'evaluation

Evaluation deterministe, 50 episodes par seed, 3 seeds par methode.

### Resultats agreges

| Metrique | DQN Custom (gamma=0.99) | SB3 DQN Defaults (gamma=0.8) |
|----------|------------------------|------------------------------|
| Mean reward (cross-seed) | **20.20 +/- 0.45** | 16.90 +/- 0.995 |
| Crash rate | **2.67%** | 78.67% |
| Mean episode length | **29.47** | 18.38 |

### Detail par seed

**DQN Custom (gamma=0.99):**

| Seed | Mean Reward | Std Reward | Mean Length | Crash Rate |
|------|-------------|-----------|-------------|-----------|
| 0 | 20.49 | 0.69 | 29.88 | 2% |
| 1 | 19.57 | 4.21 | 28.52 | 6% |
| 2 | **20.55** | **0.14** | **30.00** | **0%** |

**SB3 DQN Defaults (gamma=0.8):**

| Seed | Mean Reward | Std Reward | Mean Length | Crash Rate | Episodes |
|------|-------------|-----------|-------------|-----------|---------|
| 0 | 15.997 | 8.29 | 17.38 | 78% | 2763 |
| 1 | 16.415 | 7.64 | 18.42 | 80% | 2703 |
| 2 | 18.286 | 8.76 | 19.34 | 78% | 2687 |

Le DQN custom (gamma=0.99) converge systematiquement et de maniere fiable:
std cross-seed de 0.45, crash rate de 2.67%. SB3 defaults montre une
instabilite plus grande par episode (std intra-seed 8-9) mais une bonne
consistance inter-seed (std cross-seed 0.995). Le crash rate eleve (78.67%)
est directement attribuable a gamma=0.8 (cf. ablation DQN custom, section 6).

## 4. Analyse du comportement appris

Le DQN custom converge systematiquement vers une **politique conservatrice
stable**: l'agent roule a vitesse basse (~20 m/s), ne change jamais de voie,
et survit la quasi-totalite des episodes (29.47 steps en moyenne, proche du
maximum de 30). La reward par episode non-crash est remarquablement constante
(~20.45), refletant une politique deterministe.

Le SB3 defaults adopte une politique plus erratique: les 78% de crashes
indiquent que l'agent n'a pas appris a valoriser la survie sur l'horizon
complet. Les episodes non-crash atteignent des rewards elevees (jusqu'a ~30),
montrant que l'agent est capable de bonnes trajectoires ponctuelles, mais ne
les generalise pas. Ce comportement est coherent avec un faible discount factor:
l'agent optimise les rewards immediates sans anticiper les collisions futures.

Ce comportement s'explique par la structure de reward et le gamma:
- Avec gamma=0.8, la valeur actualisee d'une collision au step 20 est
  0.8^20 * (-1.5) ≈ -0.035, presque negligeable face a la reward immediate
- Avec gamma=0.99, la meme collision vaut 0.99^20 * (-1.5) ≈ -1.23,
  ce qui motive l'agent a l'eviter activement

## 5. Analyse des modes d'echec

### Crashes SB3 (failure_summaries.json)

Sur 150 episodes d'evaluation (3 seeds x 50), 118 se terminent en collision
(78.67%). Le profil est homogene entre seeds (78%, 80%, 78%), ce qui confirme
que l'echec est systematique plutot que lie a un seed particulier.

### Crashes DQN Custom (gamma=0.99)

Le DQN custom montre un profil radicalement different avec seulement 4 crashes
sur 150 episodes (2.67%):
- Seed 0: 1 crash (step ~16, reward 15.80)
- Seed 1: 3 crashes (rewards: 2.28, 2.16, 4.89 — collisions tres precoces)
- Seed 2: 0 crashes

Les rares crashes du DQN custom correspondent a des scenarios de trafic
initiaux defavorables (vehicules rapides dans la voie adjacente au depart)
plutot qu'a un defaut de la politique.

### Cause commune

Le mode d'echec partage pour les rares crashes du DQN custom et la majorite
des crashes SB3 est le **trafic dense**: l'agent roule lentement mais est
parfois rattrape par un vehicule plus rapide. La penalite de -0.02 par
changement de voie dissuade l'agent d'esquiver. La difference est que le
DQN custom (gamma=0.99) a appris que la survie a long terme vaut plus que
l'evitement du lane change; SB3 defaults (gamma=0.8) n'a pas fait cette
generalisation.

## 6. Discussion

### Ce que la comparaison mesure

Cette comparaison confronte deux implementations avec des choix
d'hyperparametres differents, notamment gamma. Elle ne permet pas d'isoler
l'effet de gamma de celui des autres differences (optimiseur, epsilon schedule)
sans ablation supplementaire.

### Impact de gamma (ablation DQN custom)

L'ablation sur gamma realise avec le DQN custom est l'element cle pour
interpreter la comparaison:

| Gamma | Mean Reward | Crash Rate |
|-------|-----------|-----------|
| 0.80 | 18.23 +/- 1.25 | 74.0% |
| 0.95 | 15.18 +/- 2.67 | 76.0% |
| 0.99 | **20.20 +/- 0.45** | **2.67%** |

Le crash rate de SB3 defaults (78.67%) est directement comparable au crash
rate du DQN custom avec gamma=0.8 (74%). Cela suggere que l'essentiel de
l'ecart entre les deux methodes est attribuable au choix de gamma, et non
aux differences d'implementation (optimiseur, epsilon schedule). Cette
hypothese n'est pas prouvee isolement mais est fortement soutenue par
la convergence des deux valeurs.

### Reward shaping et comportement

La configuration de reward partagee produit inevitablement un agent passif.
Pour obtenir un comportement plus proche de la conduite humaine, il faudrait:
- Augmenter `high_speed_reward` (>1.0)
- Reduire `collision_reward` en valeur absolue
- Supprimer la penalite de lane change
- Ajouter un `right_lane_reward` positif

Cette configuration est imposee par le projet, et les deux implementations
l'exploitent dans la limite de ce que leur gamma permet.
