#!/usr/bin/env python3
"""Compatibility entry point for prediction-only HumanML3D evaluation."""

from eval_babel_predonly import main


if __name__ == "__main__":
    main(default_dataset="humanml3d")
