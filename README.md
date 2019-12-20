# CV Project

By: Soham Tamba and Mark Xavier

## Models

The models we trained for the project are all stored in the `models/` directory
in the root of this repository.  They are:

DIR basic_rotnet

These models simply employ the rotation prediction training
implemented by sslime.

- basic_rotnet_not_finetuned.pyth (rotation model not finetuned)
- basic_rotnet_finetuned.pyth (rotation model finetuned)

DIR adv_ontop_of_rotnet

These models were trained to predict rotations (basic rotnet), then afterward
underwent another 60 epochs of training with adverserial noise and an
adverserial eval loop.

- adverserial_ontop_basic_rotnet_nofinetune.pyth (not finetuned after training)
- adverserial_ontop_basic_rotnet_finetuned.pyth (model finetuned after training)

DIR adv_only

These models were trained from the beginning with only adverserial training.

- adv_only_no_finetune.pyth (non-finetuned trained model)
- adv_only_finetuned.pyth (fine-tuned adverserial after training)

## To Train

All models are trained the same way (using STL-10):

`python fair-sslime/tools/train.py --config_file fair-sslime/extra-scripts/<yaml file>`

For basic rotnet, the extra scripts were:
train: unsupervised_vgg_a_rotation_stl_10.yaml
finetune: eval_vgg_a_rotation_stl_10.yaml

For adverserial training on top of basic rotnet:
Start with basic rotnet trained (but not finetuned)
train: adv_unsupervised_vgg_a_rotation_stl10.yaml
finetune: eval_vgg_a_rotation_stl_10.yaml

For adverserial only:
train: adv_unsupervised_vgg_a_rotation_stl10.yaml
finetune: eval_vgg_a_rotation_stl_10.yaml
