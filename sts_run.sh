python eval_KorFinMTEB.py --model_name_or_path BAAI/bge-en-icl --task_type STS --pooling_method mean
python eval_KorFinMTEB.py --model_name_or_path Alibaba-NLP/gte-Qwen2-1.5B-instruct --task_type STS --pooling_method mean
python eval_KorFinMTEB.py --model_name_or_path intfloat/e5-mistral-7b-instruct --task_type STS --pooling_method mean
python eval_KorFinMTEB.py --model_name_or_path BAAI/bge-large-en-v1.5 --task_type STS --pooling_method mean
# python eval_KorFinMTEB.py --model_name_or_path text-embedding-3-small --task_type STS --pooling_method mean
python eval_KorFinMTEB.py --model_name_or_path sentence-transformers/all-MiniLM-L12-v2 --task_type STS --pooling_method mean
python eval_KorFinMTEB.py --model_name_or_path hkunlp/instructor-base --task_type STS --pooling_method mean
python eval_KorFinMTEB.py --model_name_or_path nlpai-lab/KURE-v1 --task_type STS --pooling_method mean