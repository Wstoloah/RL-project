# Rapport Individuel - Christophe Boshra

## 1. Introduction

Dans le cadre de ce projet de Reinforcement Learning, mon role au sein du
groupe a ete de mettre en place l'entrainement d'un agent DQN via
**Stable-Baselines3** (SB3) sur l'environnement highway-v0, puis de realiser
la **comparaison** entre cet agent et le DQN implemente from scratch par mes
coequipiers.

J'ai aborde la comparaison en deux temps. Le premier run, presente dans ce
rapport, est une comparaison **"toutes choses egales par ailleurs"**: j'ai
aligne au maximum les hyperparametres de SB3 sur ceux du DQN custom afin
d'isoler l'effet des differences d'implementation (optimiseur, batching des
updates, ordre interne des operations) plutot que l'effet des hyperparametres.
Un second run, utilisant les hyperparametres recommandes par defaut pour SB3
sur highway-env, viendra completer l'analyse pour distinguer les deux effets.

## 2. Contribution : SB3 DQN Training (run aligne)

### Pipeline d'entrainement

J'ai concu le pipeline complet dans `sb3_training.ipynb`. Ce notebook contient
le **run aligne**: meme config, memes seeds, meme nombre d'episodes que le DQN
custom. Il est structure autour des etapes suivantes:

1. **Configuration de l'environnement**: utilisation de la config partagee (`shared_core_config.py`) pour garantir une comparaison equitable avec le DQN custom.

2. **Entrainement par episodes**: 2 500 episodes par seed (identique au DQN custom), sur 1 seul environnement, avec un callback qui controle l'arret apres le nombre d'episodes cible.

3. **Epsilon exponentiel**: j'ai surcharge le schedule epsilon de SB3 via un callback pour reproduire exactement le decay exponentiel du DQN custom (`eps = 0.05 + 0.95 * exp(-steps/5000)`).

4. **Entrainement multi-seed**: 3 seeds (0, 1, 2) pour evaluer la robustesse.

5. **Evaluation**: 50 episodes deterministes par seed avec seeds d'evaluation separees (offset=1000) pour eviter le biais.

6. **Sauvegarde**: checkpoints des modeles, metriques en JSON, courbes d'entrainement, et videos de rollout.

### Choix des hyperparametres : alignement deliberé

Pour rendre la comparaison interpretable comme "toutes choses egales par
ailleurs", j'ai configure SB3 pour qu'il utilise exactement les memes
hyperparametres que le DQN custom:

- Architecture: MLP [256, 256] (identique au custom)
- Learning rate: 5e-4, gamma: 0.99
- Buffer: 15 000, batch size: 32
- Epsilon: exponentiel (eps_decay=5000), surcharge via callback pour etre
  identique au custom
- Target update: tous les 50 steps
- Gradient clipping: max_grad_norm=100
- Volume: 2 500 episodes (identique au custom)
- Train frequency: 4 steps, avec 4 gradient steps par phase

Il est important de noter que **ce choix eloigne SB3 de ses hyperparametres
recommandes pour highway-env** (notamment `train_freq=1` et un epsilon
lineaire). Ce n'est donc pas un test de SB3 "hors de sa zone d'usage normale":
c'est un test visant a reduire au maximum les facteurs confondants pour
isoler l'effet des differences d'implementation qui persistent malgre
l'alignement:

1. **Optimiseur**: Adam dans SB3 vs AdamW+amsgrad dans le custom.
2. **Batching des updates**: 4 steps groupes vs 1 step a la fois.
3. **Ordre interne des operations**: SB3 gere son propre pipeline de
   buffer/target sync, ce qui peut introduire des decalages fins par rapport
   au custom.

Pour evaluer SB3 dans sa configuration naturelle (et ainsi comparer les deux
**outils** plutot que les deux implementations sous contrainte), un second
run avec les hyperparametres recommandes sera ajoute dans un notebook separe.

### Enregistrement des rollouts

J'ai enregistre 3 episodes video pour le meilleur seed (seed 2) afin de visualiser qualitativement la politique apprise. Les videos revelent un agent qui maintient une vitesse basse et ne change jamais de voie, ce qui constitue la politique optimale pour la reward configuree.

## 3. Resultats

### Performance SB3

| Seed | Mean Reward | Std | Mean Length | Crash Rate |
|------|------------|-----|-------------|-----------|
| 0 | 17.32 | 6.54 | 24.90 | 26% |
| 1 | 10.37 | 6.47 | 12.42 | 96% |
| 2 | 19.67 | 3.53 | 28.76 | 6% |
| **Cross-seed** | **15.79** | **3.95** | **22.03** | **42.67%** |

Les resultats revelent une forte instabilite: seul le seed 2 converge vers une politique viable, tandis que le seed 1 echoue completement avec 96% de crash rate.

### Comparaison avec DQN Custom

Le DQN custom (gamma=0.99) obtient une reward moyenne de 20.20 (+/- 0.45) avec
un crash rate de 2.67%, au-dessus de SB3 aligne (15.79 +/- 3.95, crash rate
42.67%). **Cette moyenne SB3 est toutefois tiree fortement vers le bas par un
outlier**: le seed 1 ne converge pas et atteint 96% de crash rate. Si on
exclut ce seed, SB3 fait ~18.5 de reward pour ~16% de crash rate, soit un
ecart bien plus modere. Avec seulement 3 seeds, un unique echec de
convergence suffit a faire basculer la conclusion.

Deux hypotheses peuvent expliquer l'ecart residuel, mais aucune n'a ete
testee isolement:

- **Optimiseur** (AdamW+amsgrad custom vs Adam SB3): AdamW decouple le weight
  decay, et amsgrad garantit des taux d'apprentissage effectifs non-croissants.
  Pour valider cette hypothese, il faudrait injecter AdamW dans SB3 via
  `policy_kwargs` et refaire tourner les 3 seeds.
- **Frequence des updates**: dans le custom chaque step d'environnement
  declenche immediatement un update, tandis que SB3 groupe 4 steps puis fait
  4 updates. Pour departager, il suffirait de repasser SB3 en
  `train_freq=1, gradient_steps=1`.

Une troisieme explication plus simple est que la variance cross-seed de SB3
est juste plus elevee par hasard dans cette configuration alignee, et qu'un
echantillon 5-10 seeds rapprocherait les deux methodes. Cette possibilite est
consistante avec le fait que, hors seed 1, les performances des deux methodes
ne sont pas radicalement differentes.

Le detail complet de cette comparaison est disponible dans `compare.md`.

## 4. Analyse du comportement et modes d'echec

### Politique apprise

Lorsque l'entrainement converge (seed 2), l'agent SB3 adopte la meme politique "idle" que le DQN custom: vitesse minimale, aucun changement de voie. Cette strategie maximise la reward cumulee en evitant les penalites de collision (-1.5) et de lane change (-0.02) tout en accumulant une petite recompense de vitesse a chaque step.

Cependant, cette convergence n'est pas garantie: sur 3 seeds, seul le seed 2 atteint ce comportement stable. Le seed 0 montre une politique partiellement apprise (26% crash) et le seed 1 n'a pas converge du tout (96% crash).

### Modes d'echec

Sur 150 episodes d'evaluation, 64 se terminent en collision (42.67%). On distingue deux types d'echec:

1. **Echec de convergence** (seed 1): l'agent n'a pas appris a eviter les
   collisions durant l'entrainement. Cela produit un crash rate de 96%,
   indiquant que la politique finale est quasi-aleatoire. Ce mode d'echec
   n'est pas observe sur le DQN custom dans cette comparaison sur 3 seeds,
   mais avec un echantillon aussi petit on ne peut pas conclure qu'il est
   specifique a SB3 — il peut simplement etre plus frequent sous cette
   configuration alignee.

2. **Crashes environnementaux** (seeds 0 et 2): meme avec une politique apprise, l'agent est parfois percute par un vehicule plus rapide arrivant par derriere. L'analyse des 9 crashes du meilleur seed (seed 2, sur 100 episodes de failure analysis) revele:
   - 1 crash precoce (step 7, reward 4.09): configuration initiale defavorable
   - 8 crashes tardifs (steps 12-22, rewards 8.69-16.82): l'agent survit mais se fait rattraper par le trafic dense

La cause des crashes environnementaux est l'absence de changement de voie defensif. La penalite de -0.02 par lane change, bien que faible, suffit a dissuader l'agent de toute manoeuvre, meme salvatrice.

## 5. Limitations et perspectives

### Limitations du travail (run aligne)

1. **Run "aligne" uniquement**: en choisissant d'aligner SB3 sur la config du
   DQN custom, j'ai volontairement bride SB3 par rapport a ses hyperparametres
   recommandes pour highway-env (notamment `train_freq=1` et un epsilon
   lineaire). Ce choix permet une comparaison "toutes choses egales par
   ailleurs", mais ne permet pas d'evaluer ce que SB3 donne en configuration
   naturelle. Un second notebook avec les hyperparametres par defaut
   recommandes par highway-env viendra completer cette analyse.

2. **Instabilite cross-seed et petit echantillon**: l'echec du seed 1 (96%
   crash rate) pourrait etre un signal de fragilite reel ou un simple outlier
   sur 3 seeds. Passer a 5-10 seeds est necessaire pour en decider.

3. **Hypotheses non testees isolement**: les arguments "AdamW+amsgrad" et
   "train_freq" pour expliquer l'ecart restent speculatifs tant que je n'ai
   pas fait de runs d'ablation ciblant chaque facteur separement (AdamW
   injecte via `policy_kwargs` d'un cote, `train_freq=1` de l'autre).

4. **Comparaison limitee par la reward**: la configuration de reward partagee
   produit une politique triviale (rouler lentement, ne pas changer de voie)
   lorsque l'entrainement converge. La comparaison serait plus discriminante
   sur un probleme ou la politique optimale est moins evidente.

### Perspectives

- **Second run "SB3 defaults"** (prochaine etape): utiliser les hyperparams
  recommandes pour highway-env, sur les memes seeds, et comparer les deux
  runs (aligne vs defaults) pour separer l'effet de l'alignement des
  hyperparams de l'effet des details d'implementation.
- **Ablations ciblees**: injecter AdamW dans SB3 via `policy_kwargs`, ou
  passer le custom en mode batching 4x, pour valider ou invalider les
  hypotheses explicatives actuelles.
- **Algorithmes alternatifs**: tester PPO ou A2C via SB3 pour comparer des
  familles d'algorithmes differentes, pas seulement deux implementations
  du meme DQN.
