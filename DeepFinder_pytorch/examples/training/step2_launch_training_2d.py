import sys
sys.path.append('../../') # add parent folder to path

from deepfinder.training_pylit import Train
import deepfinder.utils.objl as ol

# This script will not work because this repository does not include the training set. However it shows how training
# is realized.

# Input parameters:
path_train = ['/Users/lorenz.lamm/PhD_projects/DeepFinder_MeanShift/2D_test_data/train_images.npy',
              '/Users/lorenz.lamm/PhD_projects/DeepFinder_MeanShift/2D_test_data/train_targets.npy',
              '/Users/lorenz.lamm/PhD_projects/DeepFinder_MeanShift/2D_test_data/train_points.npy']

path_val = ['/Users/lorenz.lamm/PhD_projects/DeepFinder_MeanShift/2D_test_data/val_images.npy',
              '/Users/lorenz.lamm/PhD_projects/DeepFinder_MeanShift/2D_test_data/val_targets.npy',
              '/Users/lorenz.lamm/PhD_projects/DeepFinder_MeanShift/2D_test_data/val_points.npy']

path_data = ['/Users/lorenz.lamm/PhD_projects/Deep_Finder/Deep_Finder_pytorch/examples/training/in/tomo17_bin4_dose-filt.rec',
             '/Users/lorenz.lamm/PhD_projects/Deep_Finder/Deep_Finder_pytorch/examples/training/in/tomo32_bin4_dose-filt.rec']

path_target = ['/Users/lorenz.lamm/PhD_projects/Deep_Finder/Deep_Finder_pytorch/examples/training/in/Tomo17_all_mb_segmentations.mrc',
               '/Users/lorenz.lamm/PhD_projects/Deep_Finder/Deep_Finder_pytorch/examples/training/in/Tomo32_all_mb_segmentations.mrc']
path_objl_train = 'in/object_list_train.xml'
path_objl_valid = 'in/object_list_valid.xml'
objl_train = ol.read_xml(path_objl_train)
objl_valid = ol.read_xml(path_objl_valid)

Nclass = 1
dim_in = 56 # patch size
lr = 1e-5
weight_decay = 1e-3


# Initialize training task:
trainer = Train(Ncl=Nclass, dim_in=dim_in, lr=lr, weight_decay=weight_decay, Lrnd=13,
                tensorboard_logdir='/Users/lorenz.lamm/PhD_projects/DeepFinder_MeanShift/tensorboard_logs', two_D_test=True)
trainer.path_out         = 'out/' # output path
trainer.h5_dset_name     = 'dataset' # if training data is stored as h5, you can specify the h5 dataset
trainer.batch_size       = 16
trainer.epochs           = 100
trainer.steps_per_epoch  = 100
trainer.Nvalid           = 10 # steps per validation
trainer.flag_direct_read     = False
trainer.flag_batch_bootstrap = True
# trainer.Lrnd             = 13 # random shifts when sampling patches (data augmentation)
trainer.class_weights = None # keras syntax: class_weights={0:1., 1:10.} every instance of class 1 is treated as 10 instances of class 0

# Use following line if you want to resume a previous training session:
#trainer.net.load_weights('out/round1/net_weights_FINAL.h5')

# Load object lists:

# Finally, launch the training procedure:
trainer.launch(path_data, path_target, objl_train, objl_valid, two_D_test=True,
                two_D_data_train=path_train, two_D_data_val=path_val)
