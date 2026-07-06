
import json
import re
from pathlib import Path

from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.base import BaseNode, setup_logging, T
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config



class DocumentSpliterNode(BaseNode):
    def process(self, state: ImportGraphState) -> T:
        # 1.获取参数
        md_content,file_title,max_content_length = self._get_inputs(state)

        # 2.根据标题切割
        sections,has_title = self._split_by_headings(md_content,file_title)
        return {"sections":sections,"has_title":has_title}
        # 2.处理
        # 2.1 如果section过长 继续进行二次切割
        # 2.2 section内容过短，看能不能合并，如果能合，就合并。如果不能合并，就不合并

        # 3. 组装

        # 4，更新state

        pass

    def _get_inputs(self, state:ImportGraphState):
        config = get_config()
        # 1.获取md_content
        md_content = state.get("md_content")

        # 2.统一换行符
        if md_content:
            md_content.replace("\r\n","\n").replace("\r","\n")

        # 3.获取文件标题
        file_title = state.get("file_title")

        return md_content,file_title,config.max_content_length

    def _split_by_headings(self, md_content:str, file_title:str) -> tuple[list[dict],bool]:
        """
        根据md的标题（1-6）进行切分
        Args:
            md_content:
            file_title:

        Returns:
            list[dict]: 切分后的section
                {
                    "title":"# 第一章",
                    "body":"正文内容...",
                    "file_title":"文件名",
                    "parent_title":"# 第一章"
                }
            bool: 是否有标题
        """
        # 1.定义变量
        sections = []
        has_title = False
        in_fence = False
        body_lines = []
        current_level = 0
        current_title = ""
        hierarchy = [""]*7 # 0号索引不用

        # 2.定义正则表达式(group1:标题的语法符号#【最少1个#，最多6个】)
        heading_re = re.compile(r"^\s*(#{1,6})\s+(.+)")

        # 3.切分
        content_lines = md_content.split("\n")

        def _flush():
            """
            封装section对象
            Returns:
            """

            parent_title = ""
            body = "\n".join(body_lines)
            if current_title or body:
                for i in range(current_level-1,0,-1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break
                if not parent_title:
                    parent_title = current_title if current_title else file_title
                return sections.append({
                    "title":current_title if current_title else file_title, # 一定要存在
                    "body":body,
                    "file_title":file_title,
                    "parent_title":parent_title  # 一定要存在
                })

        for content_line in content_lines:
            # 3.1 判断是否存在代码块
            if content_line.strip().startswith("```") or content_line.strip().startswith("~~~"):
                in_fence = not in_fence
            match = heading_re.match(content_line) if not in_fence else None
            if match:
                has_title = True
                # 当前行是标题
                _flush()
                level = len(match.group(1)) # 当前标题的级别
                current_level = level # 当前标题的级别，_flush使用
                current_title = content_line
                hierarchy[level] = current_title   # 当前标题的名字
                # 存储当前遍历的标题
                body_lines = []
                for i in range(level+1,7):
                    hierarchy[i] = ""
            else:
                # 除了标题行，全都搜集起来
                body_lines.append(content_line)
        _flush()
        return sections,has_title

if __name__ == '__main__':
    document_node = DocumentSpliterNode()
    file_path = r"E:\python_project\shopkeeper\knowledge\test\input\test_document_spliter_node.md"
    with open(file_path,"r",encoding="utf8") as f:
        content = f.read()
    state = {
        "file_title":"万用表的使用",
        "md_content":content,

    }
    print(document_node.process(state).get("sections"))
    print(len(document_node.process(state).get("sections")))