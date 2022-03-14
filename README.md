# Machine learning accelerated transient analysis of stochastic nonlinear structures
This paper presents a non-intrusive surrogate modeling scheme for transient response analysis of nonlinear structures involving random parameters. The proposed scheme utilizes a two-level neural network architecture as a surrogate model. Specifically, it combines feed-forward neural networks with convolutional autoencoders to deliver a highly accurate and inexpensive emulator of the structural system under investigation. The surrogate is built upon an initial set of full model evaluations, which are performed for a small, yet sufficient number of parameter values and serve as the training data set. For each type of degree of freedom in the structural problem, a convolutional autoencoder is trained over the corresponding solution matrices in order to obtain a low-dimensional vector representation through its encoder and a reconstruction map by the decoder. Subsequently, a feed forward neural network is efficiently trained to map points from the problem's parametric space to the latent space given by the encoder, which can be further mapped to the actual, high-dimensional, system responses by the decoder mapping. The proposed surrogate is capable of predicting the entire time history response almost instantaneously and with remarkable accuracy, despite the nonlinearities present in the system's response. The elaborated methodology is demonstrated on the stochastic nonlinear transient analysis of single and multi degree of freedom structural systems.

* Stefanos Nikolopoulos, Ioannis Kalogeris, Vissarion Papadopoulos ["Machine learning accelerated transient analysis of stochastic nonlinear structures"](https://www.sciencedirect.com/science/article/abs/pii/S0952197621004541?via%3Dihub) 

## Citation
    @article{NIKOLOPOULOS2022114020,
    title = {Machine learning accelerated transient analysis of stochastic nonlinear structures},
    journal = {Engineering Structures},
    volume = {257},
    pages = {114020},
    year = {2022},
    issn = {0141-0296},
    doi = {https://doi.org/10.1016/j.engstruct.2022.114020},
    url = {https://www.sciencedirect.com/science/article/pii/S0141029622001663},
    author = {Stefanos Nikolopoulos and Ioannis Kalogeris and Vissarion Papadopoulos}}

![OFF](https://user-images.githubusercontent.com/15322711/139530447-7ede7f0a-b407-4de5-b869-76e75f299c07.png)
![ON](https://user-images.githubusercontent.com/15322711/139530448-169b262f-9dc1-4d4e-a617-686b46ae172e.png)
