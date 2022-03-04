# standard lib
import argparse
import pdb

# 3rd-party lib
import pandas as pd

# mm lib
import mmcv
from mmcv import Config

# gaia lib
import gaiavision
from gaiavision.model_space import ModelSpaceManager, build_sample_rule


def parse_args():
    parser = argparse.ArgumentParser(description='Collect and display model sub-space of interest')
    parser.add_argument('model_space_path', help='file that defines the entire model space')
    parser.add_argument('rule_path', help='file that defines rules for selecting sub-space')
    # TODO: add other useful functions

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.rule_path)
    model_space = ModelSpaceManager.load(args.model_space_path)
    rule = build_sample_rule(cfg.model_sampling_rules)

    # apply rule
    sub_model_space = model_space.ms_manager.apply_rule(rule)
    model_metas = sub_model_space.ms_manager.pack()
    print(sub_model_space)


if __name__ == '__main__':
    main()
