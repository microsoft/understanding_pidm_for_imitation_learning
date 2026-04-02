# Understanding PIDM for Imitation Learning

This codebase is accompanying the research paper ["When does predictive inverse dynamics outperform behavior cloning?"](https://arxiv.org/abs/2601.21718) and serves to reproduce the experiments for 2D Navigation tasks. The provided library allows to train imitation learning agents using Behavior Cloning (BC) and Predictive Inverse Dynamics Models (PIDM).

## Citation

If you use this code in your research, please cite the following paper:
Schäfer, Lukas, Pallavi Choudhury, Abdelhak Lemkhenter, Chris Lovett, Somjit Nath, Luis França, Matheus Ribeiro Furtado de Mendonça et al. "When does predictive inverse dynamics outperform behavior cloning?." arXiv preprint arXiv:2601.21718 (2026).

In BibTeX format:

```bibtex
@article{schafer2026pidm,
  title={When does predictive inverse dynamics outperform behavior cloning?},
  author={Sch{\"a}fer, Lukas and Choudhury, Pallavi and Lemkhenter, Abdelhak and Lovett, Chris and Nath, Somjit and Fran{\c{c}}a, Luis and de Mendon{\c{c}}a, Matheus Ribeiro Furtado and Lamb, Alex and Islam, Riashat and Sen, Siddhartha and others},
  journal={arXiv preprint},
  year={2026}
}
```

## Licensing

All source code within this repository is licensed under the [MIT license](CODE_LICENSE.txt). The [released datasets](datasets/README.md) are licensed under the [CDLA 2.0 license](DATA_LICENSE.txt).

## Module Documentation

The codebase is organized into the following main modules, each with detailed documentation:

| Module | Description |
|--------|-------------|
| [pidm_imitation/agents](pidm_imitation/agents/README.md) | Agent definitions, neural networks, training scripts, and inference agents |
| [pidm_imitation/environment/toy_env](pidm_imitation/environment/toy_env/README.md) | 2D toy environment for experiments |
| [datasets](datasets/README.md) | Navigation datasets for training and evaluation |

### Installation
It is recommended to install the package with a virtual environment such as conda:
```bash
    conda create -n pidm_imitation python=3.10
    conda activate pidm_imitation
    git clone https://github.com/microsoft/understanding_pidm_for_imitation_learning
    cd understanding_pidm_for_imitation_learning
    pip install -e .
```

### Recording Data

#### Toy Environment Recording

For the 2D toy environment, use the `play.py` script to record data.:

```bash
python -m pidm_imitation.environment.toy_env.play \
    --config configs/toy_env/toy_env_two_room.yaml \
    --agent random \
    --record \
    --experiment my_experiment \
    --episodes 10
```

See [pidm_imitation/environment/toy_env/README.md](pidm_imitation/environment/toy_env/README.md) for environment details and agent types. See [datasets/README.md](datasets/README.md) for information on provided datasets.

### Training Models

Start local training using the training script, for example to start a short training run of PIDM in the Four Room task:

```bash
python -m pidm_imitation.agents.supervised_learning.train \
    --config configs/supervised_learning/pidm_example.yaml \
    --new
```

See [pidm_imitation/agents/README.md](pidm_imitation/agents/README.md) for all CLI arguments, supported algorithms, and model architectures.

### Evaluation / Rollouts in 2D Toy Environment

To evaluate a trained model in the 2D toy environment, use the `toy_evaluate_model.py` script. For example, to evaluate the PIDM model trained above:

```bash
python -m pidm_imitation.toy_evaluate_model \
    --toy_config datasets/four_room_human_data/four_room_human_00_env_config.yaml \
    --config configs/supervised_learning/pidm_example.yaml \
    --checkpoint checkpoints/last.ckpt \
    --agent toy_idm \
    --episodes 10
```

### Datasets

We provide datasets for the 2D navigation tasks used in our experiments under the `datasets/` directory. These contain 50 trajectories from a human and A\* planner, each, in the Four Room, Zigzag, Maze, and Multiroom tasks. For each collected trajectory, the dataset contains the recorded video, actions, states and observations, other trajectory data, and the environment configuration file. See [datasets/README.md](datasets/README.md) for detailed information on the dataset structure and technical specifications.

## Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

Trademarks This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow Microsoft’s Trademark & Brand Guidelines. Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party’s policies.
