# Code Index

## Core Package

- `omnicell_hf/`: Hugging Face-style OmniCell model API, processor, tokenizer, and H5AD datasets.
- `scripts/convert_legacy_checkpoint.py`: Convert an original OmniCell checkpoint into the public checkpoint layout.
- `scripts/embed_h5ad.py`: Run embedding inference on H5AD inputs.
- `scripts/train_h5ad.py`: Fine-tune on H5AD inputs.
- `scripts/train_memmap_pretrain.py`: Continue pretraining on CSR memmap datasets.

## Continual Pretraining and Fine-Tuning

- `training/prepare_figure1_agebalanced_training_anchors.py`: Construct age-balanced anchor tables.
- `training/train_memmap_multitask_alignment.py`: Multitask OmniCell-CPT fine-tuning used by representation audits.
- `training/embed_figure1_validation_cpt_nonzero_hvg.py`: Embed validation anchors with CPT checkpoints.
- `training/run_figure1_checkpoint_pareto_selection.py`: Select checkpoints from multiple representation metrics.

## Evaluation

- `evaluation/atlas/`: Vascular index construction, embedding inference, and representation metric validation.
- `evaluation/spatial/`: Spatial deconvolution and external method benchmarks.
- `evaluation/ad_readout/`: AD/control sample-level probes and interpretable multicell readouts.
- `evaluation/vascular/`: Vascular density, marker, and spatial validation analyses.
- `evaluation/figures/`: Figure source table and panel-generation scripts.
