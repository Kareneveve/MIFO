# All instance in image

## Abstract 🧐



## Installation 🤗

### Prerequisites 
- Python 3.8+
- CUDA 11.7+
  

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/Kareneveve/All_instance_in_image.git
cd All_instance_in_image

# Install LoRA
cd lora-master
pip install -r requirements.txt
python setup.py install
cd ..

# Install base dependencies
pip install -r requirements.txt
```
  

## Usage 🤩

Make sure you have successfully built python environment and installed pytorch with cuda version. Before running the script, ensure you have all the required packages installed. 

### **Semantic Learning** 📖
---

**[Command-Line Arguments]**

* `--folder_path`: The path of image folder. Default is "example".

* `--instance`: The instance folder name waiting to be learn(please make sure the instance folder have correct structure). Default is "toy_car".

* `--model_path`: The path of Pre-trained diffusion model. Default is "./stabilityai/stable-diffusion-2-1-base".

* `--seed`: Random seed to determine the initial token. Default is None.
  

**[Folder Structure]**

Before start to Semantic Learning, please make sure the instance folder have the structure as follow:

```
{folder_path}/
└── {instance}/
    ├── img.jpg       # Target instance image
    ├── mask0.png     # Semantic mask 0 
    ├── mask1.png     # Semantic mask 1
    └── ...           # Additional masks

```
  

**[Running the Script]**

You can execute the learning step by running the following command:

```bash
python semantic_learning.py
```

The predefined seed are random seed.

The training result will be output on the folder names "Model_ours/{instance name}".
  

### **Semantic Synthesis** 🪄
---

After finish the Semantic Learning state, we can use it to synthesis the instance with a new prompt.

**[Command-Line Arguments]**

* `--batch_size`: The batch size of inference. Default is 1.

* `--model_id`: The name of Semantic learning model in the "Model_ours". Default is "toy_car".

* `--output_path`: Path to save the generated images. Default is "output".

* `--prompt`: The prompt to inference the image. Default is "a photo of &lt;asset0&gt;".

* `--seed`: Random seed to determine the initial latent. Default is None.

* `--subject_token_indices`: The sequence numbers of the asset in the prompt(start from 1). Default is [[4]].
  

**[Running the Script]**

To make it more intuitive, we have placed the definition of `boxes`  in the function `get_boxes` of the `Semantic_synthesis.py` file. Before making the inference, please select the boxes you wish to use and ensure that it matches the number of assets in the prompt.

After that, You can execute the synthesis step by running the following command:

```bash
python semantic_synthesis.py
```

The predefined seed are random seed.

The script will save the result in the folder which create under the `output_path` and named as `model_id`.
  

## Citation
```
@inproceedings{,
title={},
author={},
booktitle={},
year={},
url={}
}
```