# TWICE: What Advantages Can Low-Resource Domain-Specific Embedding Model Bring? - A Case Study on Korea Financial Texts

<p align="center">
    <a href="https://arxiv.org/abs/2502.07131">
        <img alt="Paper" src="https://img.shields.io/badge/arXiv-2502.07131-b31b1b.svg">
    </a>
    <a href="https://huggingface.co/nmixx-fin">
        <img alt="Huggingface" src="https://img.shields.io/badge/huggingface-nmixx_fin-ffd700.svg">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="#installation">Installation</a> |
        <a href="#example-usage">Example Usage</a> |
        <a href="#arguments">Arguments</a> |
        <a href="#tasks">Task list</a>
    <p>
</h4>


We introduce **KorFinMTEB**, a novel benchmark for the **Korean financial domain**, specifically tailored to reflect its unique cultural characteristics in low-resource languages.

Domain specificity of embedding models is critical for effective performance. However, existing benchmarks, such as FinMTEB, are primarily designed for high-resource languages, leaving low-resource settings, such as Korean, under-explored. Directly translating established English benchmarks often fails to capture the linguistic and cultural nuances present in low-resource domains. Our experimental results reveal that while the models perform robustly on a translated version of FinMTEB, their performance on KorFinMTEB uncovers subtle yet critical discrepancies, especially in tasks requiring deeper semantic understanding, that underscore the limitations of direct translation. 


The basic pipeline is built upon [FinMTEB](https://github.com/yixuantt/finmteb), [MTEB](https://github.com/embeddings-benchmark/mteb)




## Installation

```bash
conda create -n korfinmteb python=3.11
git clone https://github.com/nmixx-fin/TWICE.git
cd TWICE
pip install -r requirements.txt
```



## Example Usage

* Using a Python script:

```python
python eval_KorFinMTEB.py --model_name_or_path KURE --task_type CLUSTERING
```

* Using shell script (sh) file :

> Before execution, you must specify the model to be tested and the task to be tested in the sh file.

```bash
sh run.sh
```




## Arguments
```bash
Arguments:
        --model_name_or_path (str, default="BAAI/bge-large-zh"):
            Path to the pre-trained model or model identifier from Hugging Face Model Hub.
        
        --task_type (str, default=None):
            Specifies the type of task to be executed. Available options include:
            - CLASSIFICATION
            - RETRIEVAL
            - CLUSTERING
            - RERANKING
            - STS
            - SUM
            - PAIRCLASSIFICATION
        
        --add_instruction (bool, default=False):
            If set, includes additional instructions for the query. This is an optional flag.
        
        --pooling_method (str, default='cls'):
            Defines the pooling method for the model. Different models may require different pooling methods.
```



## Tasks
- For comparison with **FinMTEB**, only the **datasets that directly correspond to FinMTEB (1:1 mapping)** will be evaluated first.  
- In the future, **unique sub-tasks built with proprietary Korean data** will also be added to the evaluation code.
- Each task evaluation includes the sub-tasks listed below.
- The task type can be set using the `task_type` argument.


### Classification
- [FinSent-CLS-ko](https://huggingface.co/datasets/nmixx-fin/twice_kr_fin_news_sent_cls)
- [ESG-CLS-ko](https://huggingface.co/datasets/nmixx-fin/twice_kr_esg_cls)
- [FOMC-CLS-ko](https://huggingface.co/datasets/nmixx-fin/twice_kr_fomc_cls)

### PairClassification
- [HeadlineAC-PairCLS-ko](https://huggingface.co/datasets/nmixx-fin/twice_kor_news_headline_pair_cls)

### Reranking
- [FinanceFiQA-Reranking-ko](https://huggingface.co/datasets/nmixx-fin/twice_kr_finance_reranking)

### Clustering
- [DARTCompany2Industry-Clustering-ko](https://huggingface.co/datasets/nmixx-fin/twice_dart_company2industry_clustering)

### STS
- [FinSTS-ko](https://huggingface.co/datasets/nmixx-fin/NMIXX_kor_fin_news_STS)

### Retrieval
- [TATQA-Retrieval-ko](https://huggingface.co/datasets/nmixx-fin/twice_tat_qa_retrieval)

