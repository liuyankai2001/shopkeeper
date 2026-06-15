import os, re
from pathlib import Path
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError
from knowledge.processor.import_process.base import BaseNode, setup_logging, T
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.processor.import_process.config import get_config
from openai import OpenAI


class MarkDownImageNode(BaseNode):
    """
    处理markdown图片节点
    """
    name = "md_img_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:  # type:ignore
        """

        Args:
            state:

        Returns: 当前节点处理之后的state最新状态（md_content=process_md）

        """
        config = get_config()
        # 1.处理文件路径（1.1 md内容 1.2 md的path 1.3 图片目录）
        md_content, md_path_obj, image_dir = self._get_img_md_content(state)
        if not image_dir.exists():
            # 图片不用处理了，直接更新state
            self.logger.warning(f"文件{md_path_obj.name}中暂无图片要处理")
            state['md_content'] = md_content
            return state
        # 2.扫描并处理图片
        target_images_context = self._scan_images_and_context(image_dir, md_content, config)
        # 3.用VLM为图片生成摘要（图片描述）
        images_summaries = self._extract_img_summary(md_path_obj.stem, target_images_context, config)
        # 4.将图片上传minio（图片在minio上的地址）

        # 5.会写（将图片描述和图片地址）写到md_content中

        # 6.返回state
        return {
            "res":images_summaries
        }

    def _get_img_md_content(self, state: ImportGraphState) -> tuple[str, Path, Path]:
        """

        Args:
            state:

        Returns:
            md_content: md内容
            md_path_obj: md路径
            img_dir: 图片目录
        """
        self.log_step("Step1", "读取md内容以及图片目录")
        # 1.获取md_path
        md_path = state.get("md_path", "")

        # 2.判断是否有内容
        if not md_path:
            raise ValidationError("md文件不存在", self.name)

        # 3.标准化
        md_path_obj = Path(md_path)

        # 4.判断路径是否有效
        if not md_path_obj.exists():
            raise FileProcessingError("md文件路径无效", self.name)

        # 5.读取md内容
        with open(md_path_obj, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 6.构建图片目录
        image_dir = md_path_obj.parent / "images"

        # 7.返回
        return md_content, md_path_obj, image_dir

    def _scan_images_and_context(self, image_dir: Path, md_content: str, config) -> list[
        tuple[str, str, tuple[str, str, str]]]:
        """
        扫描图片并找到图片的上下文
        返回所有图片的丰富信息（image_name, image_path, 图片的上下文）
        图片 上下文  上文 图片 下文 找上文和下文的策略通过max_char_number(max_token)
        最终获取上下文的策略是：
        1.先找到当前图片的最近一个标题
        2.从图片内容的上一行开始向上找，一直找到最近标题的下一行
        3.根据开始索引和结束索引定位到2个索引之间的内容
        4.利用段落和最大字符数，选择从这个区域中最终留下多少

        Args:
            md_path_obj:md路径
            image_dir:图片目录
            md_content:md内容

        Returns:
            list["图片名字","图片的地址",("图片最近的上面一个标题","图片的上文","图片的下文")]
        """
        self.log_step("Step2", "扫描图片文件目录")
        # config = get_config()

        target_images_context = []
        # 1.遍历图片文件目录
        for img_name in os.listdir(image_dir):
            file_ext = os.path.splitext(img_name)[1]
            if file_ext not in config.image_extensions:
                continue  # 继续处理下一个

            # 构建img_path
            img_path = str(image_dir / img_name)
            # 提取到当前图片的唯一上下文信息
            img_context = self._find_img_context_with_limit(md_content, img_name, max_chars=config.img_content_length)
            # 如果图片没有上下文
            if not img_context:
                self.logger.warning("MD文件中暂未提取到有效的图片")
                continue
            primary_img_context = img_context[0]

            # 存储到一个列表中
            target_images_context.append((img_name, img_path, primary_img_context))

        self.logger.info(f"找到{len(target_images_context)}个有效图片")
        return target_images_context

    def _find_img_context_with_limit(self, md_content: str, img_name: str, max_chars: int = 200):
        """
        从md文档中提取图片上下文信息
        使用正则
        Args:
            md_content:要操作的md
            img_name:要定位的图片名字
            max_chars:最大的字符数

        Returns:

        """
        # 1.定义正则的规则(从md中找到图片)   ![图的描述](图的地址"图的提示")
        # 1.1 第一部分
        # r:python不要再对正则中的字符做转义了
        # !:语法
        # [:需要正则转义，在正则中[代表的字符集（a-z，A-Z，0-9 + /）
        # .:任意字符
        # *:任意字符出现的数量 +至少有1个
        # ?:匹配模式是非贪婪模式
        # (:在正则中代表捕获组的意思，因此也需要转义
        # img_name=txt.jpg  :金标准，后缀的名字一定做escapechuli
        re_pattern = re.compile(r"!\[.*?\]\(.*?" + re.escape(img_name) + r".*?\)")

        # 2.从md中定位图片在哪里（按行切分 遍历每一行 看是否满足图片的正则规则）
        md_lines = md_content.split("\n")
        imgs_context = []
        for line_idx, line in enumerate(md_lines):
            if not re_pattern.search(line):
                continue

            # 找到索引，定位这张图片最近的标题 提取上文 提取下文

            # 找上文
            # 找标题 re.match(r"^#{1,6}\s+")
            head_title = ""
            head_index = -1
            for i in range(line_idx - 1, -1, -1):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    head_title = md_lines[i]
                    head_index = i
                    break
            pre_content_start_index = head_index + 1
            pre_content = md_lines[pre_content_start_index:line_idx]
            img_pre_context = self._extract_img_context_with_limit(pre_content, max_chars, direction="front")

            # 找下文
            section_index = len(md_lines)
            for i in range(line_idx + 1, len(md_lines)):
                if re.match(r"^#{1,6}\s+", md_lines[i]):
                    section_index = i
                    break
            post_content_start_index = line_idx + 1
            post_content = md_lines[post_content_start_index:section_index]
            img_post_context = self._extract_img_context_with_limit(post_content, max_chars, direction="end")
            imgs_context.append((head_title, img_pre_context, img_post_context))

        return imgs_context

    def _extract_img_context_with_limit(self, extract_content: list, max_chars: int, direction: str):
        """
        提取图片上下标题之间的内容（段落）
        如何从给定的内容中找段落？   \n\n
        Args:
            extract_content:提取到的内容
            max_chars:最大字符数
            direction:方向。front，自下而上；end，自上而下
        Returns:
        """
        # lines = extract_content.split("\n")
        current_paragraph = []  # 存储当前遍历到的内容
        final_paragraph = []  # 存储最终遍历的内容

        for line in extract_content:
            clean_strip = line.strip()
            if not clean_strip:
                if current_paragraph:
                    final_paragraph.append("\n".join(current_paragraph))
                    current_paragraph = []
            else:
                if re.match(r"!\[.*?\]\(.*?", clean_strip):
                    if current_paragraph:
                        final_paragraph.append("\n".join(current_paragraph))
                        current_paragraph = []
                    continue
                current_paragraph.append(line)
        if current_paragraph:
            final_paragraph.append("\n".join(current_paragraph))

        # 判断方向：
        if direction == "front":
            final_paragraph.reverse()

        # 判断长度
        total = 0
        selected = []
        for para in final_paragraph:
            para_len = len(para)
            if total + para_len > max_chars and selected:
                break
            selected.append(para)
            total += para_len

        if direction == "front":
            selected.reverse()
        return "\n\n".join(selected)

    def _extract_img_summary(self, document_title: str, target_images_context: list[
        tuple[str, str, tuple[str, str, str]]], config):
        """
        为所有图片生成摘要（VLM）
        Args:
            document_title:文档名字
            target_images_context:所有图片信息
            config:配置信息
        Returns:
        """
        self.log_step("Step3", "提取图片摘要")
        summaries = {}
        # 1.构建openai客户端
        try:
            client = OpenAI(
                api_key=config.openai_api_key,
                base_url=config.openai_api_base
            )
        except Exception as e:
            self.logger.error(f"VLM客户端创建失败")
            return summaries
        # 2.发送请求（提取摘要）
        for img_name, img_path, img_context in target_images_context:
            summary = self._get_img_summary(config, client, document_title, img_path, img_context)
            summaries[img_name] = summary

        # 3.返回映射表
        self.logger.info(f"生成{len(summaries)}个图片摘要")
        return summaries

    def _get_img_summary(self, config, client, document_title: str, img_path: str, img_context: tuple[str, str, str]):
        # 1.解包元组构建上下文
        section_title, pre_context, post_context = img_context

        # 2.判断上下文
        context_parts = []
        if section_title:
            context_parts.append(section_title)
        if pre_context:
            context_parts.append(pre_context)
        if post_context:
            context_parts.append(post_context)

        # 3.构建上下文
        final_context = "\n".join(context_parts) if context_parts else "暂无可用上下文"

        # 4.读取图片文件
        import base64
        with open(img_path, 'rb') as f:
            local_img_content = base64.b64encode(f.read()).decode("utf-8")

        # 5.发送请求
        response = client.chat.completions.create(
            model=config.vl_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""
                                    任务：未markdown文档中的图片生成一个简短的中文标题
                                    背景信息：
                                    1.所属文档标题：{document_title}
                                    2.图片上下文：{final_context}
                                    请结合图片视觉内容和上述上下文信息，用中文简要总结这张图片内容
                                    生成一个精准的中文标题（不要包含“图片”二字）。
                                    """
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{local_img_content}"
                            }
                        }
                    ]
                }
            ]
        )
        summary = response.choices[0].message.content.strip()
        return summary


##
if __name__ == '__main__':
    setup_logging()
    img_md_node = MarkDownImageNode()
    state = {
        "md_path": r"E:\python_project\shopkeeper\knowledge\test\output\万用表RS-12的使用\hybrid_auto\万用表RS-12的使用.md"
    }
    res = img_md_node.process(state)
    print(res['res'])
