# titanx

Python packaging TorchTitan as like the Hugging Face Trainer &rarr; wrapped complicated distributed/parallel/config of TorchTitan

API:
```python
from titanx import Trainer, Config

trainer = Trainer(
    model="llama3-8b",
    dataset=my_dataset,
    config=Config(
        batch_size=8,
        epochs=3,
        precision="bf16",
    )
)

trainer.fit()
```

Internally:

```
User API
   |
   v
titanx.Trainer
   |
   +-- Config Resolver
   |
   +-- ezpz launcher
   |
   +-- TorchTitan Trainer
   |
   +-- Distributed Runtime
   |
   +-- Checkpoint Manager
```

## 1. Package Structure

```
titanx/
├── __init__.py
│
├── trainer/
│   ├── __init__.py
│   ├── trainer.py          # public Trainer API
│   └── state.py
│
├── config/
│   ├── __init__.py
│   ├── config.py           # user config
│   └── resolver.py         # converting TorchTitan config
│
├── runtime/
│   ├── distributed.py      # torchrun/ezpz wrapper
│   ├── device.py
│   └── environment.py
│
├── checkpoint/
│   └── manager.py
│
├── models/
│   └── registry.py
│
└── backend/
    └── torchtitan.py       # TorchTitan adapter
```

# 2. Public API Architecture

```python
from titanx import Trainer, TrainingConfig

config = TrainingConfig(
    epochs=5,
    batch_size=16,
    lr=2e-5,
    precision="bf16",
    strategy="fsdp2",
)

trainer = Trainer(
    model="meta-llama/Llama-3-8B",
    dataset=train_dataset,
    config=config,
)


trainer.fit()
```

# 3. Config layer
## titanx/config/config.py

```python
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    epochs: int = 1
    batch_size: int = 8
    lr: float = 3e-4
    precision: str = "bf16"
    strategy: str = "fsdp2"
    checkpoint_dir: str = "./checkpoints"
    gradient_checkpointing: bool = True
    compile: bool = False
```

# 4. TorchTitan Adapter

TorchTitan is only imported once in:
```
backend/torchtitan.py
```

```python
from torchtitan.train import Trainer as TitanTrainer

class TorchTitanBackend:
    def __init__(self, model, dataset, config):
        titan_config = self.convert_config(config)
        self.engine = TitanTrainer(model=model, dataset=dataset,config=titan_config)

    def fit(self):
        self.engine.train()

    def convert_config(self, cfg):
        return {
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "dtype": cfg.precision,
            "fsdp": cfg.strategy=="fsdp2",
        }
```

# 5. Trainer facade

User class:
```
trainer/trainer.py
```

```python
from titanx.backend.torchtitan import TorchTitanBackend
from titanx.runtime.distributed import distributed_context

class Trainer:
    def __init__(self, model, dataset, config):
        self.backend = TorchTitanBackend(model, dataset, config)

    def fit(self):
        with distributed_context():
            self.backend.fit()
```

# 6. ezpz integration

Put `ezpz` in runtime layer.
```
runtime/distributed.py
```

```python
import ezpz

class distributed_context:
    def __enter__(self):
        ezpz.setup_torch()

    def __exit__(self, exc_type, exc, tb):
        ezpz.cleanup()
```

User only calls
```python
trainer.fit()
```

Internally:
```
ezpz setup
      |
torch.distributed init
      |
FSDP2 setup
      |
TorchTitan train
```

# 7. Providing CLIs
Support yaml ike HF:
```bash
titanx train config.yaml
```

Structure:
```
cli/
 └── train.py
```

Use:
```bash
titanx train --model llama3 --batch-size 8 --strategy fsdp2
```

Internally:
```python
trainer = Trainer(...)
trainer.fit()
```

# 8. Supporting YAML config
example `train.yaml`:
```yaml
model:
  name: llama3-8b

training:
  epochs: 3
  batch_size: 16
  precision: bf16

distributed:
  strategy: fsdp2
```

Load:
```python
cfg = TrainingConfig.from_yaml(
    "train.yaml"
)
```

# 9. Callback system
Like the HF Trainer:
```python
trainer = Trainer(
    callbacks=[
        WandbCallback(),
        CheckpointCallback(),
    ]
)
```

Structure:
```
callbacks/

callback.py
wandb.py
checkpoint.py
```

# 10. End-User Interface
## Single GPU
```python
trainer.fit()
```

## Multi GPU
Originally:
```bash
torchrun \
--nproc_per_node=8 \
train.py
```

New:
```bash
titanx launch train.py
```

Or:
```bash
titanx train --gpus 8
```

# 11. Version Management

```
titanx
 └── depends on
torchtitan @ git+https://github.com/seonghapark/torchtitan.git@ezpz
```

`pyproject.toml`
```toml
dependencies = [
 "torch>=2.5",
 "torchtitan @ git+https://github.com/seonghapark/torchtitan.git@ezpz",
 "ezpz",
]
```

## Final Overview
```
                 User
                  |
                  v
          titanx.Trainer
                  |
        +---------+---------+
        |                   |
   Config API        Runtime API
        |                   |
        v                   v
 Config Resolver           ezpz
        |
        v
 TorchTitan Adapter
        |
        v
 TorchTitan + FSDP2 + Distributed
```

This structure is independent to internal implementation of TorchTitan, therefore `titanx.Trainer` API can be maintained even when the TorchTitan is updated


# Repository Architecture
```
github.com/
├── seonghapark/
│   └── torchtitan          # upstream/fork (backend)
│
└── seonghapark/
    └── titanx               # wrapper/API layer
```

which means:
```
titanx
   |
   | depends on
   v
torchtitan
```

### 1. Due to Different API lifecycle

TorchTitan:
* distributed training framework
* fast research velocity
* high possibility of changing internal API

`titanx`:
* user-facing API
* importance of backward compatibility

### 2. Due to manage pip package

Separate repos allow:
```bash
pip install titanx
```
with a `pyproject.toml`
```toml
[project]
name = "titanx"

dependencies = [
    "torch",
    "ezpz",
    "torchtitan @ git+https://github.com/seonghapark/torchtitan.git@ezpz"
]
```

# Git organization
```
seonghapark/
│
├── torchtitan
│     └── fork
│
├── titanx
│     ├── titanx/
│     ├── tests/
│     ├── examples/
│     └── pyproject.toml
│
└── titanx-examples
      ├── llama3/
      ├── qwen/
      └── finetune/
```

# titanx repo
```
titanx/
│
├── titanx/
│   ├── __init__.py
│   │
│   ├── trainer/
│   │   └── trainer.py
│   │
│   ├── backend/
│   │   └── torchtitan.py
│   │
│   ├── runtime/
│   │   ├── distributed.py
│   │   └── launcher.py
│   │
│   ├── config/
│   │   └── config.py
│   │
│   └── checkpoint/
│       └── manager.py
│
├── examples/
│   └── llama3_fsdp2.py
│
├── tests/
│
├── pyproject.toml
│
└── README.md
```

다르게 잡는 것이 좋습니다.
