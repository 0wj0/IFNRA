# Environment

- PyTorch 			1.11.0
- Cuda 			11.3.0
- Python 			3.8
- transformers 		4.21.1

# Project Structure

.   
├── code   
│   ├── log   
│   └── tst_rst   
├── data   
│   ├── boa_vocab   
│   ├── img_roi   
│   │   ├── tw15   
│   │   └── tw17   
│   └── sourcedata   
├── my_trained   
└── pre-trained_model   
    └── bert_uncased_L-12_H-768_A-12   

The image features extracted by BUA can be downloaded from [here.](https://pan.baidu.com/s/14--qIH3v5XioZK39i_EXYA?pwd=0y6j)

Pretrained BERT should be placed in directory **bert_uncased_L-12_H-768_A-12**

# Run

## Train

Run **train.py** or **train.sh** in the **code** directory

## Test

Run **test.py** or **test.sh** in the **code** directory
