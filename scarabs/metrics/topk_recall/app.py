import evaluate
from evaluate.utils import launch_gradio_widget

module = evaluate.load("topk_recall")
launch_gradio_widget(module)
