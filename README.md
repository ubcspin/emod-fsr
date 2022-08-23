# emod-fsr
Code for analyzing keypress collected during the EEG study. Protocol discussion: _Choose or Fuse: Enriching Data Views with Multi-label Emotion Dynamics. Cang et al. (ACII 2022)_

# How to use
Currently converting from Jupyter nb to python module, not ready for usage.

## File descriptions
- ```read_data.py```: utils to load CSV files
- ```clean_data.py```: utils to clean up raw files, i.e., fix scene and keystroke flags, fix sampling to 30H, create keys _a5_ (sum of all keypress values) and _a6_ (max of all keypress values)
- ```calculate_features.py```: utils to calculate features from keypress data (statistical, frequency and keystroke features)
- ```config.py```: constant declarations
- ```train.py```: main training script, training pipeline defined in ```estimator_helper.py``` (grid search cross-validation with recursive feature elimination)
- ```utils.py```: general purpose methods (pickle and load pickled files)


# TODO 
- [ ] Debug
- [ ] Optimize methods
- [ ] Plot results
- [ ] ```create_training_dataset.py```