
import json
from pathlib import Path

from knowledge.processor.import_process.exceptions import ValidationError
from knowledge.processor.import_process.base import BaseNode, setup_logging
from knowledge.processor.import_process.state import ImportGraphState



class EntryNode(BaseNode):
    """
    实体节点
    位置：整个导入流程中的位置（第一位）
    作用：对上传的文件类型做判断（pdf、md）
    """


    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        处理文件类型的检测
        Args:
            state:ImportGraphState

        Returns:ImportGraphState:该节点处理之后的节点状态

        """
        # 1.获取导入文件的路径以及所在的目录
        self.log_step("Step1","[获取文件路径]")
        import_file_path = state.get("import_file_path")
        file_dir = state.get("file_dir")

        # 2.简单校验一下文件的路径以及所在的目录
        self.log_step("Step2","[检测文件路径]")
        if not import_file_path or not file_dir:
            raise ValidationError("文件目录或文件不存在",self.name)

        # import_file_path:"F:\personal\skills\02_Linux\Linux.pdf"
        # 3.使用标准的Path对象操作
        path = Path(import_file_path)

        # 4.获取文件后缀
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
        elif suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
        else:
            self.logger.debug(f"不支持的文件类型{suffix}")
            raise ValidationError("不支持的文件类型",self.name)

        # 6.获取文件的标题名
        state["file_title"] = path.stem

        return state


################################

################################
if __name__ == '__main__':
    setup_logging()
    pdf_path = r"F:\personal\skills\02_Linux\Linux.pdf"
    # 方式一：直接实例该节点对象，调用process方法
    test_entry_state = {
        "file_dir":r"F:\personal\skills\02_Linux",
        "import_file_path":pdf_path
    }
    entry_node = EntryNode()

    # res = entry_node.process(test_entry_state)
    # print(json.dumps(res,ensure_ascii=False,indent=4))
    # 方式二：直接实例该节点对象，调用__call__方法
    res = entry_node(test_entry_state)
    print(json.dumps(res, ensure_ascii=False, indent=4))