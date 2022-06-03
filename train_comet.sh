
export TORCH_MODEL_ZOO=/fsx/jacampos/experiments/model_zoo/
export TORCH_HOME=/fsx/jacampos/experiments/model_zoo/

torchrun --nproc_per_node=7 train_comet.py --config ./configs/comet.yaml
