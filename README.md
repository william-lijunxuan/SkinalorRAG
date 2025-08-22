# SkinalorRAG
Skinalor combines image and text inputs, leveraging a multimodal model for preliminary screening. It incorporates a Retrieval-Augmented Generation (RAG) mechanism to enhance accuracy and explainability.

This project mainly provides: construction and validation of a RAG system, RAG-based retrieval, and performance evaluation of model baselines and RAG-augmented models.


# clone our project
git clone https://github.com/william-lijunxuan/SkinalorRAG.git

cd SkinalorRAG

# prepare python environment
## create conda 
conda create -n SkinalorRAG python=3.10

conda activate SkinalorRAG

pip install -r requirements.txt
## install flash-attn  
pip install flash-attn --no-build-isolation

Make sure the flash-attn version is compatible; otherwise, you’ll encounter the following error.

If you encounter issues installing flash-attn, please refer to this link.
https://blog.csdn.net/sinat_39179993/article/details/147968969?spm=1001.2014.3001.5501

https://github.com/Dao-AILab/flash-attention/releases?page=1
## download the model to local 
git clone https://huggingface.co/openai/clip-vit-base-patch32

git clone https://huggingface.co/google/medgemma-4b-it

git clone https://huggingface.co/lingshu-medical-mllm/Lingshu-7B

git clone https://huggingface.co/JZPeterPan/MedVLM-R1


## Modify the configuration file
Modify model and RAG configuration file
### Modify model configuration file
/Skinalor/model/config/config_Lingshu.py

/Skinalor/model/config/config_medgemma_4b_it.py

/Skinalor/model/config/config_Lingshu.py
### Modify RAG configuration file

/Skinalor/RAG/buildRAG/MMSkinQA/config.py

/Skinalor/RAG/buildRAG/SkinCAP/config.py


# build RAG dataset
## MMSkinQA
cd /Skinalor/RAG/RAGDataSet/MMSkinQA/

python buildRAG.py

## SkinCAP
cd  /Skinalor/RAG/buildRAG/SkinCAP

python buildRAG.py


# Launch the model
## Option 1 — CLI


cd Skinalor/model

python eval.py --config_model_name Lingshu

python eval.py --config_model_name medgemma_4b_it

python eval.py --config_model_name MedVLM_R1


## Option 2 — Notebooks

For customizable parameters, refer to the eval_sh.py launch-script instructions.

Alternatively, run any of the test_*.ipynb notebooks in the model directory.
