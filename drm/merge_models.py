import re
import os
import sys
from dataclasses import dataclass
import argparse
import json
from omegaconf import OmegaConf
import logging

import transformers
from transformers import AutoConfig, set_seed
from peft import PeftConfig

from ._average_merging import task_arithmetic_merging
from ._decom_renorm_merge import drm_merging


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, required=True)
    args = parser.parse_args()
    return args


def load_models(config):
    def load_model_from_path(model_path):
        if model_path is None:
            return None

        model_config = AutoConfig.from_pretrained(model_path)

        # Determine the base model class from peft config
        if model_config.architectures is None:
            try:
                peft_config = PeftConfig.from_pretrained(model_path)
                peft_base_model_config = AutoConfig.from_pretrained(peft_config.base_model_name_or_path)
                model_config = AutoConfig.from_pretrained(peft_base_model_config)
                model_class_name = model_config.architectures[0]
            except:
                raise ValueError(f"invalid model path: {model_path}")
        else:
            model_class_name = model_config.architectures[0]

        model_class = getattr(transformers, model_class_name)
        model = model_class.from_pretrained(model_path)
        return model

    # Load the base model
    base_model_path = config.get("base_model", None)
    base_model = load_model_from_path(base_model_path)

    # Load candidate models to be merged
    models = []
    coefficients = []
    for individual_model_config in config.get("models", list()):
        model = load_model_from_path(individual_model_config.model)
        models.append(model)

        coeff = individual_model_config.get('parameters.coefficient', 1.0)
        coefficients.append(coeff)

    return base_model, models, coefficients


def get_linear_weight_parameter_names(config, base_model):
    linear_parameter_regex_pattern = config.get("linear_parameter_regex_pattern", None)
    linear_parameter_ignore_regex_pattern = config.get("linear_parameter_regex_pattern", list())
    if linear_parameter_regex_pattern is None:  # No FFN
        return list()

    linear_parameter_names = []
    for param_name, param in base_model.named_parameters():
        if (
            any(re.match(pattern, param_name) for pattern in linear_parameter_regex_pattern)
            and not any(re.match(pattern, param_name) for pattern in linear_parameter_ignore_regex_pattern)
            and param.dim() == 2
        ):
            linear_parameter_names.append(param_name)
    return linear_parameter_names


if __name__ == "__main__":
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"
    set_seed(1)
    logger = logging.getLogger(__name__)

    args = parse_args()
    config = OmegaConf.load(args.config_path)
    assert config.save_path is not None, "You should specify `save_path` in the config."

    # Load merging candidate models, and a base model
    base_model, models, merging_coefficients = load_models(config)
    if base_model is None:
        raise ValueError('Base model failed to be loaded.')
    elif any(model is None for model in models):
        raise ValueError('Some candidate models failed to be loaded.')

    linear_weight_parameter_names = get_linear_weight_parameter_names(config, base_model)

    stacking_direction = config.get("merging_config.direction", 'horizontal')
    assert stacking_direction in {'horizontal', 'vertical'}

    drm_config = dict(
        singular_matrices_drop_rate=config.get("merging_config.singular_matrices_drop_rate", 0.8),  # Trim rate of processed V (V_tilda)
        non_linear_weight_magnitude_dropping_rate=config.get(
            "merging_config.non_linear_module_entries_drop_rate", 0.0
        ),  # Trim rate for non-linear modules
        enable_disjoint_mean=config.get("merging_config.enable_disjoint_mean", True),
        enable_sign_resolution=config.get("merging_config.enable_sign_resolution", True),
        ignore_patterns=config.get("merging_config.ignore_module_regex_pattern", list()),
        dtype=config.get("merging_config.dtype", "float32"),

        is_horizontal_stack=stacking_direction=='horizontal',
        merging_coeff=merging_coefficients,
        linear_weight_parameter_names=linear_weight_parameter_names,
    )

    # Perform merging
    logger.info(f'Performing merging with the config: {drm_config}')
    merged_model = drm_merging(models=models, base_model=base_model, **drm_config)

    merged_model.save_pretrained(config.save_path)
    logger.info("Merging finished.\n")
