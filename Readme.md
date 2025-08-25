# IBCircuit: Circuit Discovery Framework

The example of circuit discovery for transformer models using Information Bottleneck (IB) on pretrained GPT2-small.

## Project Structure

```
IBCircuit/
├── configs/                    # Configuration files
│   ├── config.yaml         
│   └── model            
│   │  └── gpt2_small.yaml
│   └── train           
│       └── train_gpt2.yaml
├── dsets/                      # Dataset implementations  
│   ├── ioi_dataset.py          
│   └── greaterthan_dataset.py
├── models/                     # Model implementations, i.e., IB-based GPT-2 model
│   └── gpt2_IB.py              
└── utils/                      # Utility functions
|   ├── neel_plotly.py 
|   ├── pytorch_utils.py
├── main.py                     # main script
└── README.md                   # This file
```

## Quick Start

**Installation:**
```bash
conda env create -f environment.yml
```

**Circuit discovery:**
```bash
# IOI dataset
python main.py model.pretrained_model_path='PATH_TO_MODEL' train.saved_res_dir='PATH_TO_RESULT' train.data_name='greaterthan'

# GT dataset  
python main.py model.pretrained_model_path='PATH_TO_MODEL' train.saved_res_dir='PATH_TO_RESULT' train.data_name='ioi'
```

## Configuration

Configuration files are located in the `configs/` directory. Key parameters:

**Training parameters:**
- `alpha`: KL regularization weight
- `beta`: Faithfulness weight
- `learning_rate`: Learning rate
- `num_epochs`: Number of training epochs
- `saved_res_dir`: Path to save results
- `data_name`: IOI or greaterthan datasets

**Model parameters:**
- `model_name`: pretrained model, e.g., gpt2-small
- `pretrained_model_path`: Path saved pretrained model


## Datasets

- **IOI (Indirect Object Identification)**: Tests ability to identify indirect objects in sentences
- **GT (Greater Than)**: Tests numerical comparison abilities


## Citation
If you found any part of this code is useful in your research, please consider citing our paper:
```
@inproceedings{bianibcircuit,
  title={IBCircuit: Towards Holistic Circuit Discovery with Information Bottleneck},
  author={Bian, Tian and Niu, Yifan and Yuan, Chaohao and Piao, Chengzhi and Wu, Bingzhe and Huang, Long-Kai and Rong, Yu and Xu, Tingyang and Cheng, Hong and Li, Jia},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```