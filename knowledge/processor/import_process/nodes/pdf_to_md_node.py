import json
from pathlib import Path
from typing import Tuple
from knowledge.processor.import_process.exceptions import ValidationError, FileProcessingError, PdfConversionError
from knowledge.processor.import_process.base import BaseNode, T, setup_logging
from knowledge.processor.import_process.state import ImportGraphState
import os
import subprocess


class PdfToMdNode(BaseNode):
    """
    pdf 转md节点
    """
    name = "pdf_to_md_node"

    def process(self, state: T) -> T:
        """

        Args:
            state:

        Returns:

        """
        # 1.对参数做校验
        import_file_path, file_dir_path= self._validate_state_input_path(state)

        # 2.利用mineru工具解析pdf为md
        process_code = self._execute_mineru(import_file_path,file_dir_path)
        if process_code != 0:
            raise PdfConversionError("mineru解析pdf失败",self.name)
        # 3.更新state字典的md_path
        md_path = self._get_md_path(import_file_path,file_dir_path)
        # 4.更新state字典的md_path
        state['md_path'] = md_path
        # 5.返回state
        return state

    def _validate_state_input_path(self,state:ImportGraphState) -> Tuple[Path,Path]:
        """

        Args:
            state: 该节点接收到的状态

        Returns:

        """
        self.log_step("step1","对状态的路径输入参数做校验")
        # 1.获取输入pdf文件路径
        import_file_path = state.get("import_file_path","")
        # 2.获取解析后的输出目录
        file_dir = state.get("file_dir","")

        # 3.校验输入的文件路径(非空判断)
        if not import_file_path:
            raise ValidationError("解析的文件路径为空",self.name)

        # 4.Path标准化
        import_file_path_obj = Path(import_file_path)

        # 5.是否是一个真实的路径
        if not import_file_path_obj.exists():
            raise FileProcessingError("解析的文件路径不存在",self.name)

        # 6.判断输出目录是否为空
        if not file_dir:
            # 默认目录做兜底
            file_dir = import_file_path_obj.parent

        # 7.标准输出目录
        file_dir_path_obj = Path(file_dir)
        self.logger.info(f"上传文件的路径:{import_file_path}")
        self.logger.info(f"输出的目录:{file_dir}")

        # 8.返回输出文件以及输出目录的标准Path
        return import_file_path_obj,file_dir_path_obj

    def _execute_mineru(self, import_file_path, file_dir_path):
        """

        Args:
            import_file_path: 解析文件的路径
            file_dir_path: 解析后的文件目录

        Returns:

        """
        self.log_step("step2","执行mineru解析pdf")
        # 执行命令行:mineru -p <input_path> -o <output_path>
        os.environ["MINERU_MODEL_SOURCE"] = "local"
        # 1.构建命令行
        cmd = [
            "mineru",
            "-p",
            str(import_file_path),
            "-o",
            str(file_dir_path)
        ]
        import time
        process_start_time = time.time()
        # 2.执行命令行(子进程执行命令行)自动读取主进程的环境变量
        proc = subprocess.Popen(
            args=cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            errors="replace",
            text=True,
            encoding="utf-8",
            bufsize=1
        )
        # 3.获取日志信息
        for outlog in proc.stdout:
            self.logger.info(f"执行mineru产生的日志:{outlog}")

        # 4.等待子进程做完
        process_code = proc.wait()
        process_end_time = time.time()

        if process_code==0:
            self.logger.info(f"mineru成功解析pdf文件:{import_file_path.name},耗时:{(process_end_time-process_start_time):.2f}")
        else:
            self.logger.error(f"mineru解析pdf文件{import_file_path.name}失败")
        # 5.返回状态码
        return process_code

    def _get_md_path(self, import_file_path:Path, file_dir_path:Path):
        file_name = import_file_path.stem
        md_path = str(file_dir_path / file_name / "hybrid_auto" / f"{file_name}.md")
        return md_path



if __name__ == '__main__':
    setup_logging()
    init_state = {
        "import_file_path":r"C:\Users\28329\Desktop\test\专业实践学习报告.pdf",
        "file_dir":r"C:\Users\28329\Desktop\test"
    }
    pdf_to_md_node = PdfToMdNode()
    res = pdf_to_md_node.process(init_state)
    print(res)