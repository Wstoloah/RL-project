# Rapport Individuel - Christophe Boshra

## 1. Introduction

Dans le cadre de ce projet de Reinforcement Learning, mon role au sein du groupe a ete de mettre en place l'entrainement d'un agent DQN via **Stable-Baselines3** (SB3) sur l'environnement highway-v0, puis de realiser la **comparaison** entre cet agent et le DQN implemente from scratch par mes coequipiers.

## 2. Contribution : SB3 DQN Training

### Pipeline d'entrainement

J'ai concu le pipeline complet dans `sb3_training.ipynb`, structure autour des etapes suivantes:

1. **Configuration de l'environnement**: utilisation de la config partagee (`shared_core_config.py`) pour garantir une comparaison equitable avec le DQN custom.

2. **Vectorisation**: 8 environnements paralleles via `DummyVecEnv` pour accelerer la collecte d'experience et diversifier le contenu du replay buffer.

3. **Entrainement multi-seed**: 3 seeds (0, 1, 2) pour evaluer la robustesse. Chaque seed est entraine sur 75 000 timesteps.

4. **Evaluation**: 50 episodes deterministes par seed avec seeds d'evaluation separees (offset=1000) pour eviter le biais.

5. **Sauvegarde**: checkpoints des modeles, metriques en JSON, courbes d'entrainement, et videos de rollout.

### Choix des hyperparametres

J'ai configure le DQN SB3 pour etre aussi proche que possible du DQN custom afin de rendre la comparaison significative:

- Architecture: MLP [256, 256] (identique au custom)
- Learning rate: 5e-4, gamma: 0.99
- Buffer: 15 000, batch size: 32
- Epsilon: lineaire de 1.0 a 0.05 sur 50% du training
- Target update: tous les 50 steps

Les differences avec le DQN custom sont donc limitees aux aspects inherents a SB3: optimiseur Adam (vs AdamW), epsilon lineaire (vs exponentiel), et 8 gradient steps par update.

### Enregistrement des rollouts

J'ai enregistre 3 episodes video pour le meilleur seed (seed 0) afin de visualiser qualitativement la politique apprise. Les videos revelent un agent qui maintient une vitesse basse et ne change jamais de voie, ce qui constitue la politique optimale pour la reward configuree.

## 3. Resultats

### Performance SB3

| Seed | Mean Reward | Std | Crash Rate |
|------|------------|-----|-----------|
| 0 | 20.10 | 2.48 | 6% |
| 1 | 19.35 | 3.84 | 8% |
| 2 | 19.98 | 2.42 | 4% |
| **Cross-seed** | **19.81** | **0.33** | **6.0%** |

### Comparaison avec DQN Custom

Le DQN custom (gamma=0.99) obtient une reward moyenne de 20.20 (+/- 0.45) avec un crash rate de 2.67%, legerement superieur a SB3 (19.81 +/- 0.33, crash rate 6.0%).

Cette difference s'explique principalement par:
- Le decay exponentiel d'epsilon du DQN custom, mieux adapte a cet environnement
- L'utilisation d'AdamW avec amsgrad dans le DQN custom
- La collecte sequentielle qui produit des trajectoires plus coherentes

Neanmoins, les deux approches convergent vers la meme politique conservatrice et les performances restent comparables. Le detail complet de cette comparaison est disponible dans `compare.md`.

## 4. Analyse du comportement et modes d'echec

### Politique apprise

L'agent SB3 adopte systematiquement une politique "idle": vitesse minimale, aucun changement de voie. Cette strategie maximise la reward cumulee en evitant les penalites de collision (-1.5) et de lane change (-0.02) tout en accumulant une petite recompense de vitesse a chaque step.

La constance des rewards (~20.45 par episode non-crash) confirme que l'agent a trouve l'optimum de la fonction de reward plutot qu'un comportement de conduite realiste.

### Mode d'echec principal

Sur 150 episodes d'evaluation, 9 se terminent en collision (6%). Analyse des crashes:

- **Crashes precoces** (reward < 7, episodes courts): l'agent est percute par un vehicule plus rapide arrivant par derriere dans sa voie. La configuration initiale du trafic place l'agent dans une position vulnerable, et sa politique passive ne prevoit aucune manoeuvre d'evitement.

- **Crash tardif** (reward ~26, episode long): un cas isole ou l'agent a survcu longtemps mais a ete rattrape par le trafic en fin d'episode.

La cause commune est l'absence de changement de voie defensif. La penalite de -0.02 par lane change, bien que faible, suffit a dissuader l'agent de toute manoeuvre, meme salvatrice. C'est une illustration concretedu compromis reward shaping: optimiser la reward ne signifie pas toujours optimiser le comportement souhaite.

## 5. Limitations et perspectives

### Limitations du travail

1. **Pas d'hyperparameter tuning**: j'ai aligne les hyperparametres sur ceux du DQN custom plutot que d'optimiser independamment pour SB3. Un grid search sur le learning rate, le buffer size, ou l'exploration fraction aurait pu ameliorer les resultats.

2. **Budget d'entrainement**: 75 000 timesteps est modeste pour un DQN. Des runs plus longs (200K-500K) pourraient reveler des differences de convergence entre SB3 et le DQN custom.

3. **Comparaison limitee par la reward**: la configuration de reward partagee produit une politique triviale pour les deux approches, rendant la comparaison moins discriminante qu'elle ne le serait sur un probleme plus complexe.

### Perspectives

- Tester d'autres algorithmes SB3 (PPO, A2C) pour comparer des familles d'algorithmes differentes, pas seulement des implementations de DQN
- Augmenter la recompense de vitesse pour forcer l'agent a developper une politique de depassement, ce qui rendrait la comparaison des architectures plus interessante
- Exploiter les capacites specifiques de SB3 (callbacks avances, curriculum learning) qui n'ont pas d'equivalent direct dans le DQN custom
