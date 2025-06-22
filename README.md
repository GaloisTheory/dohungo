# dohungo 🏯 : A Research-Oriented Go Bot

> ⚠️ **Work in Progress**: This bot is under active development. 

`dohungo` is a compact, research-oriented Go bot aimed at reproducing strong supervised CNN policies (≈2-dan level) trained on KGS high-dan SGF records. The ultimate goal is to develop a bot capable of beating a 4-dan human player. Further I hope to use the trained bot to do research into the net as well. 

**👉 [Quickstart Notebook](notebooks/quickstart.ipynb)** - Interactive demo showing the complete pipeline from SGF download to move prediction in ~10 minutes.

The notebook demonstrates:
- SGF game record downloading and parsing
- Board position encoding (7-plane and 11-plane encoders)
- CNN training with real-time metrics
- Move prediction visualization

## Implementation Status

| Component | Status | Description |
|-----------|--------|-------------|
| SGF downloader | ✅ | Downloads professional games from u-go.net |
| SevenPlaneEncoder | ✅ | 7-plane encoding (stones + liberties + ko + side) |
| SimpleEncoder | ✅ | 11-plane encoding (detailed liberty counts) |
| CNN architecture | ✅ | ~450k parameter model (4 conv + FC layers) |
| Training pipeline | ✅ | PyTorch training with Adadelta optimizer |
| Interactive play | ✅ | Command-line interface to play against AI |
| Quickstart notebook | ✅ | End-to-end demo for reviewers |
| Test suite | ✅ | pytest coverage for core components |
| RL self-play | 🚧 | Policy gradient fine-tuning (future work) |
| Mechanistic analysis | 🚧 | CNN filter/neuron interpretability (future work) |

## Quick Start

### Option 1: Interactive Demo 
```bash
git clone https://github.com/<you>/dohungo.git
cd dohungo
pip install -r requirements.txt
jupyter notebook notebooks/quickstart.ipynb
```

### Option 2: Full Training
```bash
python -m dohungo.train --config configs/small.yaml
```

### Option 3: Play Against Trained Model
```bash
python -m dohungo.play --model checkpoints/best.pt
```

## Repository Structure
```
dohungo/
├── data/
│   ├── downloader.py      # SGF file downloading from u-go.net
│   ├── sgf_reader.py      # SGF parsing and position extraction
│   ├── encoders/          # Board position encoders
│   │   ├── base.py        # Abstract encoder interface
│   │   ├── sevenplane.py  # 7-plane encoder implementation  
│   │   └── simple.py      # 11-plane encoder implementation
│   └── dataloader.py      # PyTorch data loading pipeline
├── models/
│   └── cnn_small.py       # CNN architecture (~450k parameters)
├── train.py               # CLI training entry-point
└── play.py               # Interactive play interface

configs/
└── small.yaml            # Training configuration

notebooks/
└── quickstart.ipynb      # 📓 Interactive demo (START HERE)
```

## Research Roadmap

### Phase 1: Supervised Learning (Current)
- ✅ **Baseline CNN**: Strong supervised policy from high-dan games
- 🚧 **Encoder ablations**: Quantify value of different input planes (liberties, ko detection)

### Phase 2: Reinforcement Learning (Next)
- 🤖 **Self-play training**: Policy gradient fine-tuning with GPU acceleration
- 🎯 **Strength evaluation**: Systematic testing against known benchmarks
- 🏆 **Target strength**: Beat 4-dan human player consistently

### Phase 3: Mechanistic Interpretability (Future - Ambitious ideas)
- 📊 **Opening bias detection**: Compare learned patterns across different training datasets

## Technical Features

- **Fully deterministic**: Reproducible training with fixed random seeds
- **CPU/GPU compatible**: Runs efficiently on both CPU and GPU
- **Streaming data**: Memory-efficient SGF processing for large datasets  
- **Extensible design**: Clean interfaces for future model architectures
- **Research-ready**: Structured for systematic ablation studies

## Dependencies

- Python 3.9+
- PyTorch ≥2.1.0
- OmegaConf (YAML configuration)
- gomill (SGF parsing)
- NumPy, matplotlib, tqdm, pytest

## Testing

```bash
source .venv/bin/activate
pytest -q  # Quick test suite (~10 seconds)
```

---

**Research Goal**: Build a systematically improvable Go bot that reaches 4-dan strength.

MIT License • © 2025 Dohun Lee