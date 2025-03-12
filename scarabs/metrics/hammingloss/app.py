import evaluate
from evaluate.utils import launch_gradio_widget


module = evaluate.load("hammingloss")
launch_gradio_widget(module)
