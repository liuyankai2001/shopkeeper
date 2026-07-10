from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.base import BaseNode, setup_logging, T
from knowledge.processor.import_process.prompts.item_name_prompt import ITEM_NAME_USER_PROMPT_TEMPLATE, \
    ITEM_NAME_SYSTEM_PROMPT
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.utils.bge_m3_embedding_util import get_bge_m3_embedding_model
from knowledge.utils.llm_client import get_llm_client


class ItemNameRecognizationNode(BaseNode):
    name = "item_recognization_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1.参数校验
        file_title, chunks, config = self._validate_inputs(state)

        # 2.构建LLM的上下文（提取商品名）
        item_name_context = self._prepare_item_name_context(file_title, chunks, config)

        # 3.调用LLM模型
        item_name = self._recognize_item_name_by_llm(file_title, item_name_context)

        # 4.嵌入商品名(嵌入模型:OpenAIEmbeddings dashscope:text_embedding_v1(2,3,4)-->稠密向量(语义相似)、稀疏向量(bge-m3)(关键词匹配))
        dense, sparse = self._embedding_item_name(item_name)
        # 5.存储到Milvus数据库中

        # 6.返回
        pass

    def _validate_inputs(self, state: ImportGraphState):
        self.log_step("step1", "校验输入参数")
        config = get_config()
        # 1.获取state信息：file_title以及chunks
        file_title = state.get("file_title")
        chunks = state.get("chunks")

        # 2.判断提取到的参数
        if not file_title:
            raise ValidationError("文件标题为空", self.name)

        if not chunks or not isinstance(chunks, list):
            raise ValidationError("分块内容为空或类型错误", self.name)

        item_name_chunk_k = config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError("商品名识别时使用的切片数量为空或小于等于0", self.name)

        self.logger.info(f"检测到文件标题：{file_title}，对应的分块数量为：{len(chunks)}")
        # 3.返回
        return file_title, chunks, config

    def _prepare_item_name_context(self, file_title: str, chunks: list[dict[str, Any]], config):
        self.log_step("step2", "构建商品名提取的上下文")
        results = []
        # 从前5块中留下的字符数不能超过2000
        total = 0
        for index, chunk in enumerate(chunks[:config.item_name_chunk_k]):
            # 1.判断chunk的类型
            if not isinstance(chunk, dict):
                continue
            # TODO 构建上下文：【切片-1】 标题+body组成(content:标题+body)
            # 2.提取
            content = chunk.get("content")
            species = f"【切片】 - {index + 1} - {content}"
            # 3.计算长度
            total += len(species)
            results.append(species)
            # 4.判断收集到的长度是否超过阈值设定
            if total > config.item_chunk_size:
                break

        return "\n\n".join(results)[:config.item_chunk_size]

    def _recognize_item_name_by_llm(self, file_title: str, item_name_context: str) -> str:

        # 1.实例化llm客户端
        llm_client = get_llm_client()
        if llm_client is None:
            self.logger.error(f"LLM初始化失败，安全回退到文件标题：{file_title}")
            return file_title
        # 2.构建LLM提示词
        prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_context)

        # 3.调用模型
        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=ITEM_NAME_SYSTEM_PROMPT),
                HumanMessage(content=prompt)
            ])
            content = getattr(llm_response, "content", "")
            if not content or content.upper() == "UNKNOWN":
                self.logger.warning(f"LLM无法提取有效商品名，安全回退到文件标题：{file_title}")
                item_name = file_title

                return item_name
            item_name = content
            self.logger.info(f"LLM提取商品名为：{item_name}")
            return item_name
        except Exception as e:
            self.logger.error(f"调用LLM模型出错：{e}")
            return None

    def _embedding_item_name(self, item_name: str) -> tuple[list, dict[Any, Any]]:
        # 1.获取嵌入模型对象
        embedding_model = get_bge_m3_embedding_model()
        # 2.嵌入item_name
        embedding_result = embedding_model.encode_documents([item_name])
        # 3.获取稠密和稀疏向量
        dense = embedding_result['dense'][0].tolist()
        start = embedding_result['sparse'].indptr[0]
        end = embedding_result['sparse'].indptr[1]
        weights = embedding_result['sparse'].data[start:end].tolist()
        token_id = embedding_result['sparse'].indices[start:end].tolist()
        sparse = dict(zip(token_id, weights))
        return dense, sparse
