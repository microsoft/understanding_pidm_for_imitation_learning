# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Tuple

import lightning.pytorch as pl
import torch

from pidm_imitation.agents.supervised_learning.base_models import ActionRegressor
from pidm_imitation.agents.supervised_learning.extract_args_utils import (
    ExtractArgsFromConfig,
)
from pidm_imitation.agents.supervised_learning.inference_agents.idm.idm_planners import (
    IdmPlanner,
)
from pidm_imitation.agents.supervised_learning.inference_agents.idm.idm_planners_factory import (
    IdmPlannerFactory,
)
from pidm_imitation.agents.supervised_learning.inference_agents.pytorch_agents import (
    PytorchAgent,
    PytorchIdmAgent,
    PytorchSlidingWindowAgent,
    PytorchSlidingWindowIdmAgent,
)
from pidm_imitation.agents.supervised_learning.inference_agents.pytorch_valid_agents import (
    ValidPytorchAgents,
)
from pidm_imitation.agents.supervised_learning.inference_agents.utils.observation_handlers import (
    StateHandler,
)
from pidm_imitation.agents.supervised_learning.model_factory import ModelFactory
from pidm_imitation.configs.config_offline_pl import OfflinePLConfigFile
from pidm_imitation.constants import CHECKPOINT_EXTENSION, LAST_CHECKPOINT
from pidm_imitation.torch_utils import get_device
from pidm_imitation.utils import Logger

logger = Logger()
log = logger.get_root_logger()


class PytorchAgentFactory:

    @staticmethod
    def _get_agent_class(agent_name: str, model: ActionRegressor) -> type[PytorchAgent]:
        if agent_name == ValidPytorchAgents.BC:
            if model.is_recurrent:
                return PytorchAgent
            return PytorchSlidingWindowAgent
        if agent_name == ValidPytorchAgents.IDM:
            if model.is_recurrent:
                return PytorchIdmAgent
            return PytorchSlidingWindowIdmAgent
        raise ValueError(
            f"Unknown agent name: {agent_name}, must be in {ValidPytorchAgents.ALL}"
        )

    @staticmethod
    def _get_observation_handler(
        config: OfflinePLConfigFile,
    ) -> Any:
        return StateHandler(state_type=config.state_config.type)

    @staticmethod
    def _get_idm_planner(
        config: OfflinePLConfigFile,
        data_hparams: Any,
        **planner_kwargs: Dict[str, Any],
    ) -> IdmPlanner:
        return IdmPlannerFactory.get_idm_planner(
            config=config,
            device=get_device(),
            train_lookahead_slice=data_hparams["lookahead_slice"],
            **planner_kwargs,
        )

    @staticmethod
    def _get_agent_args(
        agent_name: str,
        config: OfflinePLConfigFile,
        model: pl.LightningModule,
        model_hparams: Any,
        data_hparams: Any,
        model_path: str,
        **planner_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        observation_handler = PytorchAgentFactory._get_observation_handler(
            config=config
        )
        agent_args = {
            "config": config,
            "model": model,
            "model_hparams": model_hparams,
            "data_hparams": data_hparams,
            "model_path": model_path,
            "observation_handler": observation_handler,
        }
        if agent_name == ValidPytorchAgents.IDM:
            agent_args["idm_planner"] = PytorchAgentFactory._get_idm_planner(
                config=config,
                data_hparams=data_hparams,
                **planner_kwargs,
            )
        return agent_args

    @staticmethod
    def get_checkpoint_file(
        model_path: str, checkpoint_name: str | None, model_sub_dir: str | None
    ) -> str:
        checkpoint_name = checkpoint_name if checkpoint_name else LAST_CHECKPOINT
        if not checkpoint_name.endswith(CHECKPOINT_EXTENSION):
            checkpoint_name += CHECKPOINT_EXTENSION
        if model_sub_dir:
            checkpoint_file = os.path.join(model_path, model_sub_dir, checkpoint_name)
        else:
            checkpoint_file = os.path.join(model_path, checkpoint_name)
        if not os.path.isfile(checkpoint_file):
            raise ValueError(f"Could not find the checkpoint in {checkpoint_file}.")
        return checkpoint_file

    @staticmethod
    def create_model_and_load_checkpoint(
        config: OfflinePLConfigFile,
        checkpoint_file: str,
    ) -> Tuple[ActionRegressor, Any, Any]:
        algorithm = ExtractArgsFromConfig.get_algorithm(config)
        model_class = ModelFactory.get_model_class(algorithm)
        device = get_device()
        log.info(
            f"Loading model class {model_class.__name__} from checkpoint {checkpoint_file}"
        )
        model: ActionRegressor = model_class.load_from_checkpoint(  # type: ignore
            checkpoint_file, map_location=device  # type: ignore
        )  # type: ignore

        print(model)
        model.eval()
        model.to(device)

        # Training checkpoints contain serialized model objects in their hyperparameters.
        checkpoint = torch.load(
            checkpoint_file, map_location=device, weights_only=False
        )
        model_hparams = checkpoint.get("hyper_parameters", {})  # type: ignore
        datamodule_hparams = checkpoint.get("datamodule_hyper_parameters", {})  # type: ignore

        return model, model_hparams, datamodule_hparams

    @staticmethod
    def get_agent(
        agent_name: str,
        config: OfflinePLConfigFile,
        model_path: str,
        checkpoint_name: str | None,
        model_sub_directory: str | None = None,
        **planner_kwargs: dict[str, Any],
    ) -> PytorchAgent:
        model_path = os.path.realpath(model_path)
        checkpoint_file = PytorchAgentFactory.get_checkpoint_file(
            model_path, checkpoint_name, model_sub_directory
        )
        model, model_hparams, data_hparams = (
            PytorchAgentFactory.create_model_and_load_checkpoint(
                config, checkpoint_file
            )
        )

        agent_class = PytorchAgentFactory._get_agent_class(agent_name, model)
        agent_kwargs = PytorchAgentFactory._get_agent_args(
            agent_name,
            config,
            model,
            model_hparams,
            data_hparams,
            model_path,
            **planner_kwargs,
        )
        return agent_class(**agent_kwargs)
