#!/bin/bash

python3 -m et_converter.et_converter \
    --input_type=New \
    --input_filename=../../../inputs/custom_workload/llm.txt \
    --output_filename=../../../inputs/custom_workload/llm \
    --num_npus=64 \
    --num_dims=1

python3 -m et_visualizer.et_visualizer \
    --input_filename=../../../inputs/custom_workload/llm.0.eg \
    --output_filename=../../../outputs/custom_workload/llm.0

python3 -m et_visualizer.et_visualizer \
    --input_filename=../../../inputs/custom_workload/llm.1.eg \
    --output_filename=../../../outputs/custom_workload/llm.1

python3 -m et_visualizer.et_visualizer \
    --input_filename=../../../inputs/custom_workload/llm.18.eg \
    --output_filename=../../../outputs/custom_workload/llm.18