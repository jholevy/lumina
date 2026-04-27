# Lumina

**Éclairez vos médias, localement.**

Lumina est un outil d'amélioration de photos et vidéos par intelligence artificielle, qui tourne **entièrement en local** — aucune donnée ne quitte votre machine.

## ✨ Fonctionnalités

| Action | Photo | Vidéo | Modèle IA |
|---|---|---|---|
| 🔍 Upscaling x2/x4 | ✓ | ✓ | Real-ESRGAN |
| 🧹 Débruitage | ✓ | ✓ temporel | Real-ESRGAN / BasicVSR++ |
| 👤 Restauration visages | ✓ | ✓ | CodeFormer |
| 🎨 Correction couleur | ✓ | ✓ | CNN + CLAHE |
| ✨ Netteté | ✓ | — | Filtre passe-haut |
| ⏩ Frame Interpolation | — | ✓ | RIFE |

## 🚀 Installation

```bash
git clone https://github.com/jholevy/lumina.git
cd lumina
pip install -e .
python run.py
```

L'interface s'ouvre à `http://127.0.0.1:7860`.

## 🖥️ Configuration

| Spécification | Minimum | Recommandé |
|---|---|---|
| RAM | 8 GB | **16+ GB** |
| GPU | — | NVIDIA CUDA / Apple Silicon MPS |
| Stockage | 1 GB | 2 GB |

## Architecture

```
lumina/
├── run.py              # Point d'entrée
├── lumina/
│   ├── config.py       # Détection hardware, mémoire
│   ├── core/
│   │   └── pipeline.py # Pipeline de traitement
│   ├── models/
│   │   └── manager.py  # Gestionnaire de modèles + swap
│   └── ui/
│       └── app.py      # Interface Gradio
└── README.md
```

## 📜 Licence

MIT — code libre. Modèles sous leurs licences respectives.
