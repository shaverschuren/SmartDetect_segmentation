# SmartDetect_segmentation
CXR segmentation using image-to-image translation GANs, created in context of the SmartDetect COVID-19 detection project. 

For recreation of the testing environment (using `anaconda3`), please undertake the following steps:

- Clone this repository and `cd` to its root.
- Create a new conda environment with Python: `conda create -n <your_env_name> python`
- Install the requirements using `pip`: `pip install -r requirements.txt`

Please refer to `doc/finalReport.pdf` for any further information.

**Project file structure:**
```
|--- data --- raw\          % Used for storing raw (immutable) data
|          |
|          -- preprocessed\ % Used for storing the processed data
|          |
|          -- results\      % Used for storing the results (for evaluation)
|
|--- doc\                   % Used for storing documentation (such as the final report)
|
|--- model\                 % Used for storing models and model weights (i.e. the final model)
|
|--- notebook\              % Used for storing the notebooks
|
|--- src\                   % Used for storing the source code
|
|--- util\                  % Used for storing basic utilities
|
|--- logs\                  % Used for storing training logs
|
|--- LICENCE                % MIT Licence
|--- README.md              % This README
```
