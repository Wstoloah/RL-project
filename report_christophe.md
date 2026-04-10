# Rapport Individuel - Christophe Boshra

## 1. Introduction

Dans le cadre de ce projet de Reinforcement Learning, mon role au sein du
groupe a ete de mettre en place l'entrainement d'un agent DQN via
**Stable-Baselines3** (SB3) sur l'environnement highway-v0, puis de realiser
la **comparaison** entre cet agent et le DQN implemente from scratch par mes
coequipiers.

## 2. Contribution : SB3 DQN Training

### Pipeline d'entrainement

J'ai concu le pipeline complet dans `sb3_defaults_training.ipynb`. Il est
structure autour des etapes suivantes:

1. **Configuration de l'environnement**: utilisation de la config partagee
   (`shared_core_config.py`) pour garantir une comparaison equitable avec
   le DQN custom.

2. **Hyperparametres**: les hyperparametres recommandes pour SB3 sur
   highway-env ont ete utilises (gamma=0.8, train_freq=1, gradient_steps=1,
   epsilon lineaire sur 10% des steps). Le volume d'entrainement est fixe a
   40 000 steps, ce qui correspond a environ 2 700 episodes.

3. **Entrainement multi-seed**: 3 seeds (0, 1, 2) pour evaluer la
   robustesse des resultats.

4. **Evaluation**: 50 episodes deterministes par seed avec seeds d'evaluation
   separees (offset=1000) pour eviter le biais de contamination train/eval.

5. **Sauvegarde**: checkpoints des modeles, metriques en JSON, courbes
   d'entrainement, et videos de rollout (seed 2).

### Hyperparametres

| Parametre | Valeur |
|-----------|--------|
| Architecture | MLP [256, 256] (ReLU) |
| Learning rate | 5e-4 (Adam) |
| Gamma | 0.8 |
| Buffer size | 15 000 |
| Batch size | 32 |
| Target update | hard, tous les 50 steps |
| Epsilon | lineaire 1.0→0.05 sur 10% des steps |
| Train frequency | chaque step (train_freq=1) |
| Gradient steps | 1 par step |
| Volume | 40 000 steps (~2 700 episodes) |

### Enregistrement des rollouts

3 episodes video ont ete enregistres pour le seed 2 afin de visualiser
qualitativement la politique apprise. Les videos revelent un agent qui
presente des comportements mixtes: certains episodes ou l'agent survit en
maintenant une vitesse moderee, et d'autres ou il entre en collision.

## 3. Resultats

### Performance SB3

| Seed | Mean Reward | Std | Mean Length | Crash Rate | Episodes |
|------|------------|-----|-------------|-----------|---------|
| 0 | 15.997 | 8.29 | 17.38 | 78% | 2763 |
| 1 | 16.415 | 7.64 | 18.42 | 80% | 2703 |
| 2 | 18.286 | 8.76 | 19.34 | 78% | 2687 |
| **Cross-seed** | **16.899** | **0.995** | **18.38** | **78.67%** | — |

La consistance cross-seed est bonne (std 0.995): les trois seeds convergent
vers des niveaux de performance similaires, sans echec complet de convergence.
En revanche, le crash rate eleve (78.67%) indique que l'agent n'a pas
generalise une politique d'evitement des collisions.

### Comparaison avec DQN Custom

| Metrique | DQN Custom (gamma=0.99) | SB3 DQN (gamma=0.8) |
|----------|------------------------|---------------------|
| Mean reward | **20.20 +/- 0.45** | 16.90 +/- 0.995 |
| Crash rate | **2.67%** | 78.67% |
| Mean episode length | **29.47** | 18.38 |

Le DQN custom (gamma=0.99) obtient de meilleures performances absolues.
L'ecart de crash rate (+76pp) est considerable mais trouve son explication
dans le choix de gamma: l'ablation DQN custom montre que le meme DQN avec
gamma=0.8 atteint 74% de crash rate — valeur tres proche des 78.67% de SB3.
**Ce n'est donc pas SB3 qui est en cause, mais le discount factor.**

Le detail complet de cette comparaison est disponible dans `compare.md`.

## 4. Analyse du comportement et modes d'echec

### Politique apprise

Avec gamma=0.8, l'agent ne valorise pas suffisamment les rewards futures:
la valeur actualisee d'une collision au step 20 est 0.8^20 * (-1.5) ≈ -0.035,
presque negligeable. L'agent priorise donc les rewards immediates (vitesse)
sans apprendre a eviter systematiquement les collisions.

Cela contraste avec le DQN custom (gamma=0.99), pour lequel la meme collision
vaut 0.99^20 * (-1.5) ≈ -1.23, ce qui motive l'agent a adopter une politique
conservative stable (vitesse basse, aucun changement de voie, survie sur les
30 steps).

### Mode d'echec principal

Sur 150 episodes d'evaluation (3 seeds x 50), 118 se terminent en collision
(78.67%). Le profil est homogene entre seeds (78%, 80%, 78%): l'echec est
systematique, pas lie a un seed particulier. L'agent est regulierement
rattrape par des vehicules plus rapides dans sa voie; sans horizon long
(gamma faible), il n'a pas appris a anticiper ce risque.

## 5. Limitations et perspectives

### Limitations

1. **Gamma non optimal pour cet environnement**: gamma=0.8 est la valeur
   par defaut recommandee pour highway-env dans SB3, mais l'ablation DQN
   custom montre qu'elle est inadaptee a l'horizon de 30 steps de cette
   tache. Tester SB3 avec gamma=0.99 permettrait d'evaluer les deux
   implementations sur un pied d'egalite.

2. **Differences non isolees**: la comparaison presente plusieurs differences
   simultanees entre les deux methodes (gamma, optimiseur AdamW vs Adam,
   epsilon schedule). Il n'est pas possible de conclure que l'une d'elles
   en particulier est la cause de l'ecart residuel sans ablation ciblee.

3. **Petit echantillon de seeds**: 3 seeds par methode est le minimum requis.
   Un run 5-10 seeds renforcerait la fiabilite des conclusions.

4. **Politique triviale**: la configuration de reward partagee produit une
   politique triviale (survie maximale = rouler lentement). La comparaison
   serait plus discriminante sur un probleme ou la politique optimale est
   moins evidente.

### Perspectives

- **SB3 avec gamma=0.99**: tester SB3 avec gamma=0.99 pour verifier
  que les deux implementations atteignent des performances similaires
  quand le discount factor est aligne.
- **Algorithmes alternatifs**: tester PPO ou A2C via SB3 pour comparer
  des familles d'algorithmes differentes, pas seulement deux implementations
  du meme DQN.
