# Mean Shift as a loss function
This repository is an adaption of the [DeepFinder](https://gitlab.inria.fr/serpico/deep-finder) repository. It converts DeepFinder's tensorflow-based network training to a Pytorch-lightning implementation.

We implemented Mean Shift clustering as a loss function to guide the network's predicted scoremaps towards more accurate cluster centers. This can facilitate downstream analyses like subtomogram averaging.

# Usage
We are currently working on an easy-to-use CLI to train your own model using our clustering function.
Furthermore, we will provide a handy Mean Shift Pytorch module that you can readily include in your network architecture to design your own models.

In the meantime, you can find the main functionalities in the [Mean Shift Utils](https://github.com/LorenzLamm/DeepFinder_MeanShift/blob/master/DeepFinder_pytorch/deepfinder/mean_shift_utils.py) and [Pytorch Lightning Model](https://github.com/LorenzLamm/DeepFinder_MeanShift/blob/master/DeepFinder_pytorch/deepfinder/model_pylit.py) files.

# Contact

If you would like to try our loss function or run into any issues, feel free to reach out, either by dropping an [issue](https://github.com/LorenzLamm/DeepFinder_MeanShift/issues) or just write me an email: Lorenz.Lamm@helmholtz-munich.de

