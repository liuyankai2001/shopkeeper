from typing import Any

from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.base import BaseNode, setup_logging, T
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from knowledge.processor.import_process.exceptions import ValidationError

class ItemNameRecognizationNode(BaseNode):
    name = "item_recognization_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:

        # 1.参数校验
        chunks,file_title,config = self._validate_inputs(state)

        # 2.构建LLM的上下文（提取商品名）
        item_name_context = self._prepare_item_name_context(file_title, chunks,config)

        # 3.调用LLM模型
        item_name = self._recognize_item_name_by_llm(file_title,item_name_context)

        # 4.嵌入商品名

        # 5.存储到Milvus数据库中

        # 6.返回
        pass

    def _validate_inputs(self, state:ImportGraphState):
        self.log_step("step1","校验输入参数")
        config = get_config()
        # 1.获取state信息：file_title以及chunks
        file_title = state.get("file_title")
        chunks = state.get("chunks")

        # 2.判断提取到的参数
        if not file_title:
            raise ValidationError("文件标题为空", self.name)

        if not chunks or not isinstance(chunks,list):
            raise ValidationError("分块内容为空或类型错误", self.name)

        item_name_chunk_k = config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError("商品名识别时使用的切片数量为空或小于等于0", self.name)

        self.logger.info(f"检测到文件标题：{file_title}，对应的分块数量为：{len(chunks)}")
        # 3.返回
        return file_title,chunks,config

    def _prepare_item_name_context(self, file_title:str, chunks:list[dict[str,Any]], config):
        self.log_step("step2","构建商品名提取的上下文")
        results = []
        # 从前5块中留下的字符数不能超过2000
        total = 0
        for index,chunk in enumerate(chunks[:config.item_name_chunk_k]):
            # 1.判断chunk的类型
            if not isinstance(chunk,dict):
                continue
            # TODO 构建上下文：【切片-1】 标题+body组成(content:标题+body)
            # 2.提取
            content = chunk.get("content")
            species = f"【切片】 - {index+1} - {content}"
            # 3.计算长度
            total+=len(species)
            results.append(species)
            # 4.判断收集到的长度是否超过阈值设定
            if total>config.item_chunk_size:
                break

        return "\n\n".join(results)[:config.item_chunk_size]

    def _recognize_item_name_by_llm(self, file_title:str, item_name_context:str):
        # 1.实例化llm客户端

        # 2.调用
        pass

