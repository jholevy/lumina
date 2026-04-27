# ✦ Lumina

**Éclairez vos médias, localement.**

Lumina est un outil d'amélioration de photos et vidéos par intelligence artificielle, qui tourne **entièrement en local** sur votre machine — aucune donnée ne quitte votre ordinateur.

---

## ✨ Fonctionnalités

| Action | Photo | Vidéo | Modèle IA |
|---|---|---|---|
| 🔍 **Upscaling** | x2 / x4 | x2 / x4 | Real-ESRGAN |
| 🧹 **Débruitage** | ✓ | ✓ temporel | Real-ESRGAN / BasicVSR++ |
| 👤 **Restauration visages** | ✓ | ✓ | CodeFormer |
| 🎨 **Correction couleur** | ✓ | ✓ | CNN + CLAHE |
| ✨ **Netteté** | ✓ | — | Filtre passe-haut |
| ⏩ **Frame Interpolation** | — | ✓ (x2, x3) | RIFE |

## 🚀 Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/jholevy/lumina.git
cd lumina

# 2. Installer les dépendances
pip install -e .

# 3. Lancer
python run.py
```

L'interface s'ouvre dans votre navigateur à l'adresse `http://127.0.0.1:7860`.

### Installation des modèles

Les modèles se téléchargent automatiquement au premier usage de chaque fonctionnalité. Vous pouvez aussi tous les pré-télécharger :

```bash
python -c "from lumina.models.manager import model_manager; model_manager.download_all()"
```

## 🖥️ Configuration requise

| Spécification | Minimum | Recommandé |
|---|---|---|
| RAM | 8 GB | **16+ GB** |
| GPU | — | NVIDIA CUDA ou Apple Silicon (MPS) |
| Stockage | 1 GB | 2 GB (modèles) |
| OS | macOS, Linux, Windows | — |

## 🎮 Utilisation

1. Lancez `python run.py`
2. Upload une photo ou vidéo
3. Coche les actions souhaitées
4. Clique "Lancer le traitement"
5. Télécharge le résultat

## 🧠 Architecture

```
lumina/
├── run.py              # Point d'entrée
├── pyproject.toml      # Dépendances
├── lumina/
│   ├── __init__.py
│   ├── config.py       # Détection hardware, chemins
│   ├── core/
│   │   └── pipeline.py # Pipeline de traitement IA
│   ├── models/
│   │   └── manager.py  # Gestionnaire de modèles (DL/swap mémoire)
│   └── ui/
│       └── app.py      # Interface Gradio
└── README.md
```

## ⚠️ Limitations actuelles

- Traitement vidéo frame par frame (cohérence temporelle partielle)
- CodeFormer nécessite PyTorch (pas en fallback CPU pur)
- Les modèles sont téléchargés depuis GitHub (~550 MB au total)

## 📜 Licence

MIT — le code est libre. Les modèles ont leurs propres licences (Real-ESRGAN : BSD-3, CodeFormer : MIT, RIFE : MIT).

---

*Construit avec ❤️ en local.*
