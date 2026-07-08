import json
import re
from email.quoprimime import body_length
from pathlib import Path
from typing import Any

from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.base import BaseNode, setup_logging, T
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.utils.markdown_utils import MarkdownTableLinearizer


class DocumentSplitNode(BaseNode):
    name = "document_split_node"

    def process(self, state: ImportGraphState) -> T:
        # 1.获取参数
        md_content, file_title, max_content_length, min_content_length = self._get_inputs(state)

        # 2.根据标题切割
        sections, has_title = self._split_by_headings(md_content, file_title)
        # return {"sections":sections,"has_title":has_title}
        # 3.处理(切分和合并)
        final_chunks = self.split_and_merge(sections, max_content_length, min_content_length)

        # 3. 组装
        chunks = self._assemble_chunk(final_chunks)
        # 4，更新state
        state['chunks'] = chunks
        return state

    def _get_inputs(self, state: ImportGraphState):
        self.log_step("step1", "切分文档的参数校验以及获取")
        config = get_config()
        # 1.获取md_content
        md_content = state.get("md_content")

        # 2.统一换行符
        if md_content:
            md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")

        # 3.获取文件标题
        file_title = state.get("file_title")

        # 4.校验最大最小值
        if config.max_content_length <= 0 or config.min_content_length <= 0 or config.max_content_length <= config.min_content_length:
            raise ValidationError(f"切片长度参数校验失败")

        return md_content, file_title, config.max_content_length, config.min_content_length

    def _split_by_headings(self, md_content: str, file_title: str) -> tuple[list[dict], bool]:
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
        hierarchy = [""] * 7  # 0号索引不用

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
                for i in range(current_level - 1, 0, -1):
                    if hierarchy[i]:
                        parent_title = hierarchy[i]
                        break
                if not parent_title:
                    parent_title = current_title if current_title else file_title
                return sections.append({
                    "title": current_title if current_title else file_title,  # 一定要存在
                    "body": body,
                    "file_title": file_title,
                    "parent_title": parent_title  # 一定要存在
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
                level = len(match.group(1))  # 当前标题的级别
                current_level = level  # 当前标题的级别，_flush使用
                current_title = content_line
                hierarchy[level] = current_title  # 当前标题的名字
                # 存储当前遍历的标题
                body_lines = []
                for i in range(level + 1, 7):
                    hierarchy[i] = ""
            else:
                # 除了标题行，全都搜集起来
                body_lines.append(content_line)
        _flush()
        return sections, has_title

    def split_and_merge(self, sections: list[dict[str, Any]], max_content_length: int, min_content_length: int):
        """

        Args:
            sections: 根据标题切分后的所有section
            max_content_length:每一个section的内容【title+body】长度最多不能超过指定：将标题注入内容中（标题注入：明确定位这一块的归属）
            min_content_length:每一个section的内容，长度如果比min_content_length小，就尝试进行合并（合并：同源）

        Returns:
            list[section]
        """
        self.log_step("step2", "切分长内容及合并短内容")
        # 1.切分
        current_sections = []
        for section in sections:
            current_sections.extend(self.split_long_section(section, max_content_length))

        # 2.合并
        final_sections = self.merge_sort_section(current_sections, min_content_length)

        # 3.返回
        return final_sections

    def split_long_section(self, section: dict[str, Any], max_content_length: int):
        self.log_step("step3", "切分长内容")

        # 1.获取section对象属性
        title = section.get("title")  # 不可能为空
        body = section.get("body")  # 可能为空
        file_title = section.get("file_title")  # 不可能为空
        parent_title = section.get("parent_title")  # 不可能为空

        # 2.判断表格
        if "<table>" in body:
            body = MarkdownTableLinearizer.process(body)

        # 3.对标题做一个校验
        TITLE_MAX_LENGTH = 50
        if len(title) > TITLE_MAX_LENGTH:
            self.logger.warning(f"标题长度超过限制，已截断：{title}")
            title = title[:TITLE_MAX_LENGTH]

        # 4.拼接title前缀
        title_prefix = f"{title}\n\n"

        # 5.计算总长度{len(title_prefix)+len(body)}
        total_length = len(title_prefix) + len(body)

        # 6.判断
        if total_length <= max_content_length:
            return [section]

        # 7.计算body可用长度
        body_length = max_content_length - len(title_prefix)
        if body_length <= 0:
            return [section]
        # 8.切分：
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=body_length,
                                                       chunk_overlap=0,
                                                       separators=['\n\n', '\n', '。', '!', '；', '.', ' ', ''],
                                                       keep_separator=False)
        texts = text_splitter.split_text(body)
        # 【长度0：body为空，长度1：body只有一个chunk】
        if len(texts) <= 1:
            return [section]
        sub_section = []
        for index, text in enumerate(texts):
            sub_section.append({
                "title": title,
                "body": text,
                "file_title": file_title,
                "parent_title": parent_title,
                "part": f"{index + 1}"
            })
        return sub_section

    def merge_sort_section(self, current_sections: list[dict[str, Any]], min_content_length: int):
        """
        贪心累加算法
        2个局限性：
            1.撑爆最小阈值
            2.孤儿小块
        Args:
            current_sections:
            min_content_length:

        Returns:

        """
        self.log_step("step4", "合并短内容")
        # 1.定义变量
        current_section = current_sections[0]
        # current_section_body = current_section.get('body')
        final_sections = []  # 最终的箱子
        for next_section in current_sections[1:]:
            # 同源
            same_parent = (current_section['parent_title'] == next_section['parent_title'])
            if same_parent and len(current_section.get('body')) < min_content_length:
                # body的合并
                current_section['body'] = (current_section.get('body').rstrip() + next_section.get('body').lstrip())
                # 更新current_title
                current_section['title'] = current_section.get('parent_title')

                current_section['part'] = 0

            else:
                # 将原来的current_section进行封箱
                final_sections.append(current_section)
                # 更新current_section
                current_section = next_section
        # 封装最后一个
        final_sections.append(current_section)

        # 对所有的section的part做处理（为每一个父标题设置对应的part）
        part_counter = {}
        for final_section in final_sections:
            if "part" in final_section.keys():
                parent_title = final_section.get('parent_title')
                part_counter[parent_title] = part_counter.get(parent_title,0) + 1
                new_part = part_counter[parent_title]
                final_section['part'] = new_part
                # final_section['title'] = final_section['parent_title'] + '-' + new_part
                final_section['title'] = final_section['title'] + '-' + new_part
        return final_sections

    def _assemble_chunk(self, final_chunks:list[dict[str,Any]]) -> list[dict[str,Any]]:
        """
        最终组合chunk
        Args:
            final_chunks:

        Returns:

        """
        chunks = []
        for chunk in final_chunks:
            # 1.获取content信息
            title = chunk.get("title")
            file_title = chunk.get("file_title")
            parent_title = chunk.get("parent_title")
            body = chunk.get("body")
            content = f"{title}\n\n{body}"

            assemble_chunk = {
                "title":title,
                "file_title":file_title,
                "parent_title":parent_title,
                "content":content,
            }
            # 2.判断part是否存在
            if "part" in chunk.keys():
                assemble_chunk['part'] = chunk.get("part")
            chunks.append(assemble_chunk)
        return chunks


if __name__ == '__main__':
    setup_logging()
    document_node = DocumentSplitNode()
    file_path = r"E:\python_project\shopkeeper\knowledge\test\input\test_document_spliter_node.md"
    with open(file_path, "r", encoding="utf8") as f:
        content = f.read()
    state = {
        "file_title": "万用表的使用",
        "md_content": content,

    }
    print(document_node.process(state).get("sections"))
    print(len(document_node.process(state).get("sections")))
