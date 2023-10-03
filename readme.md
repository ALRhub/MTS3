# Multi Time Scale World Models

This is the official implementation of the paper "Multi Time Scale World Models".
<figure class="image">
  <img src="images/pgm_mts3.png" alt="pgm" width="700">
  <figcaption>Figure: PGM of a 2 Level MTS3 (Multi Time Scale State Space Model)</figcaption>

</figure>

# In Details
```
MTS3
├── agent
│   ├── Infer
│   │   └── repre_infer_mts3.py - this file contains the inferene process
│   │ 
│   │
│   ├── Learn
│   │   └── repre_learn_mts3.py - this file contains the training/learning loops 
│   │ 
│   │
│   └── worldModels
│       ├── Decoders - this folder contains the decoders of different types
│       │   └── propDecoder.py - this file contains the decoder for the proprioceptive sensor 
│       │   
│       │ 
│       ├── gaussianTransformations - this folder contains the generic gaussian layers (see layers section)
│       │   ├── gaussian_conditioning.py 
│       │   └──  gaussian_marginalization.py 
│       │  
│       │  
│       └── SensorEncoders - this folder contains the encoders for the different sensor modalities
│       │   └──  propEncoder.py - this file contains the encoder for the proprioceptive sensor
│       │ 
│       │ 
│       ├── MTS3.py - this file contains the MTS3 model nn.Module
│       ├── hipRSSM.py - this file contains the hipRSSM model nn.Module
│       └──  acRKN.py - this file contains the acRKN model nn.Module
│       
│       
├── dataFolder
│   └──
│
│
├── experiments
│   ├── latent_plots
│   │   
│   ├── mobileRobot
│   │   └── conf
│   │   └── mts3_exp.py
│   │       
│   ├── output
│   │   └── plots
│   │ 
│   ├──  saved_models
│   │ 
│   └── exp_prediction_mts3.py
│
│
└── utils
    └── vision
```

# MTS3 Architecture

<figure class="image">
  <img src="images/mts3_readme.jpg" alt="pgm" width="700">
  <figcaption>Figure: Scematic of 2-Level MTS3.</figcaption>

</figure>

The task predict (slow time scale) and task-conditiona state predict (fast time scale) are instances of Guassian Marginalization operiation.
The task update (slow time scale) and Observation update (fast time scale) are instances of Guassian Conditioning operiation.

Thus the MTS3 model can be viewed as a hierarchical composition of Gaussian Conditioning and Gaussian Marginalization operations. The building blocks of these operations are described in the next section.

# Building Blocks (Gaussian Transformations)

The following building blocks are used in the MTS3 model to perform inference in each timescale. They can be broadly categorized into two types of [layers/gaussian transformations](https://github.com/vaisakh-shaj/MTS3/tree/master/agent/worldModels/gaussianTransformations): Gaussian Conditioning and Gaussian Marginalization. These building blocks can be used to construct MTS3 with arbitatry number of timescales.

## Gaussian Conditioning
The observation/task update and abstract action inference at every timescale are instances of this layer. It performs the posterior inference over latent state given a set of observations and the prior distribution (see the PGM below).
<figure class="image">
  <img src="images/gclayer.png" alt="pgm" width="900">
  <figcaption></figcaption>

</figure>



## Gaussian Marginalization
The predict step in every timescale is an instance of this layer. It calculates the marginal distribution of the latent state in the next timestep, given a set of causal factors (see the PGM below).
<figure class="image">
  <img src="images/gplayer.png" alt="pgm" width="900">
  <figcaption></figcaption>

</figure>





