import os
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

cache_llm_client = {}

def get_llm_client(model_name:str=None,temperature:float=0,response_format:bool=False):
    """
    获取LLM客户端
    Returns:
        返回LLM客户端对象
        缓存的对象是：client
        缓存的key：不同的节点用不同的模型以及同一个节点用不同的响应格式
    """
    model = model_name if model_name else os.getenv("ITEM_MODEL")
    cache_key = (model,response_format) # 复合形式key

    if cache_key in cache_llm_client.keys():
        return cache_llm_client[cache_key]
    model_kwargs = {}

    if response_format:
        model_kwargs = {"response_format":{"type":"json_object"}}
    try:
        client = ChatOpenAI(
            model=model,
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),
            temperature=temperature,
            extra_body={"enable_thinking":False},
            model_kwargs=model_kwargs
        )
        cache_llm_client[cache_key] = client
        return client
    except Exception as e:
        logger.error(f"llm客户端创建失败，原因：{e}")
        raise e


if __name__ == '__main__':
    llm = get_llm_client()
    res = llm.invoke("你好，请问给我讲一个笑话")
    print(res.content)