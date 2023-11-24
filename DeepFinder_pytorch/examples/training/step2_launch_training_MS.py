import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import argparse
import sys
sys.path.append('../../')  # Add parent folder to path

from deepfinder.training_pylit import Train
import deepfinder.utils.objl as ol

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for Deep Finder")
    parser.add_argument('--threeD', action='store_true', help='Flag to use 3D data')
    parser.add_argument('--eval_plots', action='store_true', help='Flag to store evaluation plots')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay for optimizer')
    parser.add_argument('--dim_in', type=int, default=56, help='Patch size')
    parser.add_argument('--Nclass', type=int, default=2, help='Number of classes')
    parser.add_argument('--bandwidth', type=float, default=4.0, help='Bandwidth parameter')
    parser.add_argument('--num_seeds', type=int, default=256, help='Number of seeds')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum number of iterations')
    parser.add_argument('--use_bce', action='store_true', help='Should we use BCE loss?')
    parser.add_argument('--use_dice', action='store_true', help='Should we use Dice loss?')

    # parser.add_argument('--use_bce', type=bool, default=False, help='Should we use BCE loss?')
    parser.add_argument('--use_MS_loss', action='store_true', help='Should we use MS loss?')
    parser.add_argument('--bce_fac', type=float, default=1.0, help='scaling of BCE loss')
    parser.add_argument('--dice_fac', type=float, default=1.0, help='scaling of Dice loss')
    
    parser.add_argument('--run_token', type=str, default="MS_DF", help='run_token')
    parser.add_argument('--case', type=str, default="spheres", help='run_token')
    return parser.parse_args()

def main():
    # Input parameters:
    args = parse_args()
    threeD = args.threeD

    # Define paths for 2D and 3D data
    if not args.case == "shrec" and not args.case == "experimental" and not args.case == "experimental_sparse":
        paths_2D = {
            'train': ['/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data/train_images_' + args.case + '.npy',
                    '/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data/train_targets_' + args.case + '.npy',
                    '/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data/train_points_' + args.case + '.npy'],
            'val': ['/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data/val_images_' + args.case + '.npy',
                    '/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data/val_targets_' + args.case + '.npy',
                    '/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data/val_points_' + args.case + '.npy']
        }

        paths_3D = {
            'train': ['/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data/train_3D_images_' + args.case + '.npy',
                    '/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data/train_3D_targets_' + args.case + '.npy',
                    '/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data/train_3D_points_' + args.case + '.npy'],
            'val': ['/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data/val_3D_images_' + args.case + '.npy',
                    '/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data/val_3D_targets_' + args.case + '.npy',
                    '/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/2D_test_data/val_3D_points_' + args.case + '.npy']
        }
    elif args.case == "shrec":
        threeD = True
        paths_3D = {
            "train": '/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/Shrec',
            "val": '/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/Shrec',
        }
    elif args.case == "experimental" or args.case == "experimental_sparse":
        threeD = True
        paths_3D = {
            "train": "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/experimental/",
            "val": "/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/experimental/",
        }


    paths = paths_3D if threeD else paths_2D

    path_data = [
        '/Users/lorenz.lamm/PhD_projects/Deep_Finder/Deep_Finder_pytorch/examples/training/in/tomo17_bin4_dose-filt.rec',
        '/Users/lorenz.lamm/PhD_projects/Deep_Finder/Deep_Finder_pytorch/examples/training/in/tomo32_bin4_dose-filt.rec'
    ]
    
    path_target = [
        '/Users/lorenz.lamm/PhD_projects/Deep_Finder/Deep_Finder_pytorch/examples/training/in/Tomo17_all_mb_segmentations.mrc',
        '/Users/lorenz.lamm/PhD_projects/Deep_Finder/Deep_Finder_pytorch/examples/training/in/Tomo32_all_mb_segmentations.mrc'
    ]
    
    path_objl_train = 'in/object_list_train.xml'
    path_objl_valid = 'in/object_list_valid.xml'
    objl_train = ol.read_xml(path_objl_train)
    objl_valid = ol.read_xml(path_objl_valid)

    # Network and training parameters
    Nclass = args.Nclass
    dim_in = args.dim_in
    lr = args.lr
    weight_decay = args.weight_decay
    bandwidth = args.bandwidth
    num_seeds = args.num_seeds
    max_iter = args.max_iter
    use_bce = args.use_bce
    use_MS_loss = args.use_MS_loss
    bce_fac = args.bce_fac
    run_token = args.run_token
    use_dice = args.use_dice
    dice_fac = args.dice_fac

    eval_plots = args.eval_plots

    # Initialize training task:
    trainer = Train(Ncl=Nclass, dim_in=dim_in, lr=lr, weight_decay=weight_decay, Lrnd=13,
                    tensorboard_logdir='/scicore/home/engel0006/GROUP/pool-engel/Lorenz/MeanShift_Loss/DeepFinder_MeanShift/DeepFinder_pytorch/examples/training/tensorboard_logs', 
                    bandwidth=bandwidth, num_seeds=num_seeds, max_iter=max_iter, 
                    two_D_test=not threeD, three_D_test=threeD, use_bce=use_bce, use_MS_loss=use_MS_loss, bce_fac=bce_fac, run_token=run_token, use_dice=use_dice, dice_fac=dice_fac, eval_plots=eval_plots)
    

    # Launch the training procedure:
    trainer.launch(path_data, path_target, objl_train, objl_valid,
                   two_D_data_train=paths['train'], two_D_data_val=paths['val'], shrec=args.case=='shrec', experimental=args.case=="experimental", experimental_sparse=args.case=="experimental_sparse")

if __name__ == "__main__":
    main()