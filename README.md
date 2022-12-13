# AIDiagnostic_CV_test_case
Medical image segmentation

## CASE DESCRIPTION

Build baseline for training of segmentation model for pleural effusion detection. 

It could be used as 2D as 3D architecture.

Code should be separated into files:

- file with model architecture
- file with data preprocessing
- file with dataset forming
- file with calculation of [DICE Coef](https://radiopaedia.org/articles/dice-similarity-coefficient#:~:text=The%20Dice%20similarity%20coefficient%2C%20also,between%20two%20sets%20of%20data.)
- main file with training

NOTE:
During training the best epoch (relying on validation) should be chose and the model with weight should be saved in `output` folder. After the training the picture with DICE coef changing through epochs should be saved in the same folder (axis Y - DICE coefficient, axis X - epoch number). Instead of picture any trackers could be used (tensorboard etc.)
