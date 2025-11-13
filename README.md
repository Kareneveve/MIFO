# MIFO: Learning and Synthesizing Multi-Instance from One Image

## Abstract 🧐

This paper proposes a method for precise learning and synthesizing multi-instance semantics from a single image. The difficulty of this problem lies in the limited training data, and it becomes even more challenging when the instances to be learned have similar semantics or appearance. To address this, we propose a penalty-based attention optimization to disentangle similar semantics during the learning stage. Then, in the synthesis, we introduce and optimize box control in attention layers to further mitigate semantic leakage while precisely controlling the output layout. Experimental results demonstrate that our method achieves disentangled and high-quality semantic learning and synthesis, strikingly balancing editability and instance consistency. Our method remains robust when dealing with semantically or visually similar instances or rare-seen objects.

## Installation 🤗

### Prerequisites 
- Python 3.10+
- CUDA 11.7+
  
### Dependencies
```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/Kareneveve/MIFO.git
cd MIFO

# Install LoRA
cd lora-master
pip install -r requirements.txt
python setup.py install
cd ..

# Install base dependencies
pip install -r requirements.txt
```


## Usage 🤩

Make sure you have successfully set up the Python environment and installed PyTorch with CUDA support.

Before running the scripts, ensure that all required packages are installed.

### **Semantic Learning** 📖
---

**[Command-Line Arguments]**

* `--folder_path`: Path to the image folder. Default is `"example"`.

* `--instance`: The instance folder name to be learned (please ensure the folder has the correct structure). Default is `"toy_car"`.

* `--model_path`: Path to the pre-trained diffusion model. Default is `"./stabilityai/stable-diffusion-2-1-base"`.

* `--seed`: Random seed to determine the initial token. Default is `None`.

**[Folder Structure]**

Before starting semantic learning, please ensure the instance folder has the following structure:

```
{folder_path}/
└── {instance}/
    ├── img.jpg       # Target instance image
    ├── mask0.png     # Semantic mask 0 
    ├── mask1.png     # Semantic mask 1
    └── ...           # Additional masks

```


**[Running the Script]**

You can execute the learning step with:

```bash
python semantic_learning.py
```

The predefined seed is randomly generated.

Training results will be saved in the folder `Model_ours/{instance name}`.




### **Semantic Synthesis** 🪄
---

After completing the semantic learning stage, you can use the trained model to synthesize instances with a new prompt.

**[Command-Line Arguments]**

* `--batch_size`: Batch size for inference. Default is `1`.

* `--model_id`: The name of the semantic learning model in `"Model_ours"`. Default is `"toy_car"`.

* `--output_path`: Path to save the generated images. Default is `"output"`.

* `--prompt`: The prompt for image synthesis. Default is `"a photo of <asset0>"`.

* `--seed`: Random seed to determine the initial latent. Default is `None`.

* `--subject_token_indices`: The index/indices of assets in the prompt (starting from 1). Default is `[[4]]`.


**[Running the Script]**

To simplify usage, the definition of `boxes` is provided in the function `get_boxes` within the `semantic_synthesis.py` file. Before inference, please select the boxes you wish to use and ensure that the number matches the number of assets in the prompt. (For visualizing boxes before inference, we provide `visual_box_mask.py`, which outputs an image with the coordinates you specify.)

After that, you can execute the synthesis step with:

```bash
python semantic_synthesis.py
```

The predefined seed is randomly generated.

Results will be saved in a folder created under `output_path`, named after the `model_id`.


## Citation
```
@article{su2025mifo,
  title={MIFO: Learning and Synthesizing Multi-Instance from One Image},
  author={Su, Kailun and He, Ziqi and Wang, Xi and Zhou, Yang},
  journal={arXiv preprint arXiv:2511.00542},
  year={2025}
}
```
