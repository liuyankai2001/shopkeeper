import json

from langgraph.graph import StateGraph

from knowledge.processor.import_process.base import setup_logging
from knowledge.processor.import_process.nodes.entry_node import EntryNode
from knowledge.processor.import_process.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_process.state import  ImportGraphState
from knowledge.processor.import_process.state import create_default_state

def create_import_graph() -> StateGraph:
    """
    定义导入业务的graph状态拓扑图（langgraph构建流水线）整个流水线各个节点读取或写入的节点
    Returns:
    """
    # 1.定义状态图
    graph_pipline = StateGraph(ImportGraphState) # type:ignore
    # 2.定义节点（入口、结束、自己需要添加的）
    graph_pipline.set_entry_point("entry_node")
    nodes = {
        "entry_node":EntryNode(),
        "pdf_to_md_node":PdfToMdNode()
    }
    for k,v in nodes.items():
        graph_pipline.add_node(k,v)


    # 3.定义边（顺序边、条件边）
    graph_pipline.add_edge("entry_node","pdf_to_md_node")
    graph_pipline.add_edge("entry_node","__end__")

    # 4.编译
    return graph_pipline.compile()

graph_app = create_import_graph()
def run_import_graph(import_file_path:str,file_dir:str):
    init_state = {
        "import_file_path":import_file_path,
        "file_dir":file_dir
    }
    state = create_default_state(**init_state)

    final_state = None
    for event in graph_app.stream(state):
        for node_name,state in event.items():
            print(f"运行的节点:{node_name},state:{state}")
            final_state = state

    return final_state


if __name__ == '__main__':
    setup_logging()
    import_file_path = r"C:\Users\28329\Desktop\test\专业实践学习报告.pdf"
    file_dir = r"C:\Users\28329\Desktop\test"
    final_state = run_import_graph(import_file_path,file_dir)
    print(json.dumps(final_state,indent=2,ensure_ascii=False))
    print("-"*50)
    print("图结构：")
    print(graph_app.get_graph().print_ascii())