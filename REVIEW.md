# Revue du Code : LLM Stress Tests

**Date :** 14 Janvier 2026
**Projet :** llm-stress-tests

---

## 1. Synthèse Globale

Le projet est une base solide et fonctionnelle pour effectuer des tests de charge distribués sur des serveurs d'inférence LLM (spécifiquement `llama.cpp`). L'architecture suit les bonnes pratiques de programmation asynchrone en Python, ce qui est crucial pour simuler une charge utilisateur élevée.

Le code est concis, modulaire et offre une très bonne observabilité grâce à l'intégration récente de Prometheus et aux rapports terminaux/JSON détaillés.

**Note de Santé :** 🟢 Stable / Bon départ
**Dette Technique :** Faible, mais l'absence de tests unitaires et de validation stricte de la configuration pourrait poser problème à mesure que le projet grandit.

---

## 2. Points Forts Architecturels

### ✅ Programmation Asynchrone (AsyncIO)
L'utilisation de `aiohttp` et `asyncio` est le bon choix technique pour ce type d'outil I/O bound. La gestion de la concurrence via `asyncio.gather` et les tâches d'arrière-plan (`progress_logger_task`, `metrics_pusher_task`) est bien implémentée.

### ✅ Moduarité
La séparation des responsabilités est claire :
- `main.py` : Orchestration et flux principal.
- `src.client` : Gestion bas niveau des requêtes HTTP et retries.
- `src.metrics` : Calculs statistiques isolés (utilisation de `numpy` pour la performance).
- `src.generators` : Création des prompts.

### ✅ Observabilité
L'outil excelle dans la restitution des résultats :
- Métriques en temps réel dans la console.
- Export temps réel vers Prometheus (Pushgateway) pour monitoring graphique.
- Sauvegarde JSON détaillée pour post-analyse.
- Analyse automatique ("Verdicts") en fin de test (CRITICAL/WARNING/PASS).

---

## 3. Points d'Amélioration (Code & Robustesse)

### ⚠️ Gestion du Protocole (Hardcoding)
Dans `src/client/api_client.py`, le parsing de la réponse streaming est étroitement couplé au format `llama.cpp` (`data: {"content": ...}`).
- **Risque** : Cela rend l'outil incompatible avec d'autres backends standards comme vLLM ou TGI (OpenAI-compatible) qui peuvent avoir des formats légèrement différents.
- **Recommandation** : Abstraire le client pour supporter différents "BackendParsers" (ex: `LlamaCppParser`, `OpenAIParser`).

### ⚠️ Absence de Tests
Il n'y a pas de répertoire `tests/` ni de tests unitaires.
- **Risque** : Les régressions sont difficiles à détecter lors des modifications (ex: changement dans le calcul des centiles).
- **Recommandation** : Ajouter `pytest` et couvrir a minima `metrics/stats.py` (calculs mathématiques simples) et mocker `api_client.py`.

### ⚠️ Validation de la Configuration
La configuration est chargée directement depuis le YAML sans validation de schéma.
- **Risque** : Une clé manquante ou un type incorrect (ex: string au lieu de int) fera planter le script au runtime.
- **Recommandation** : Utiliser `pydantic` pour définir des modèles de configuration et valider le YAML au chargement.

### ℹ️ Typing
Les annotations de type sont présentes mais pas partout, et il n'y a pas de vérification statique.
- **Recommandation** : Compléter les type hints et ajouter `mypy` au processus de développement.

---

## 4. Documentation

Le `README.md` est clair mais commence à être obsolète par rapport aux fonctionnalités du code :
- **Manquant** : Documentation de la configuration "Mixed Warfare" (multi-serveurs).
- **Manquant** : Instructions pour configurer Prometheus/Pushgateway.

---

## 5. Plan d'Action Recommandé

Voici une liste priorisée de tâches pour améliorer le projet :

1.  **Immédiat** : Mettre à jour le `README.md` pour refléter les nouvelles capacités (Prometheus, Multi-serveurs).
2.  **Court Terme** : Ajouter un modèle `pydantic` pour valider `config/workload.yaml` au démarrage.
3.  **Moyen Terme** : Refactoriser `api_client.py` pour supporter une interfacce "OpenAI-compatible" générique, permettant de tester vLLM/Ollama/TGI sans modifier le code.
4.  **Fondamental** : Mettre en place une suite de tests unitaires (`pytest`).
