import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


bge_m3_ef:Optional[BGEM3EmbeddingFunction] = None

def get_bge_m3_embedding_model():
    global bge_m3_ef

    if bge_m3_ef is not None:
        return bge_m3_ef

    # 1.获取参数
    model_name = os.getenv("BGE_M3_PATH","BAAI/bge-m3")
    device = os.getenv("BGE_DEVICE","cpu")
    use_fp16_str = os.getenv("BGE_FP16",'False')
    use_fp16 = True if use_fp16_str.lower() == 'true' else False

    bge_m3_ef = BGEM3EmbeddingFunction(
        model_name=model_name,
        device=device,  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
        use_fp16=use_fp16  # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
    )

    return bge_m3_ef

if __name__ == '__main__':
    embedding_model = get_bge_m3_embedding_model()
    query = "我喜欢python语言"
    res = embedding_model.encode_queries([query])

    # 稠密
    # print(res['dense'][0].tolist())

    # 稀疏
    start = res['sparse'].indptr[0]
    end = res['sparse'].indptr[1]
    print(f"start: {start}, end: {end}")
    weights = res['sparse'].data[start:end].tolist()
    token_id = res['sparse'].indices[start:end].tolist()
    print(weights)
    print(token_id)

    print(dict(zip(token_id, weights)))
