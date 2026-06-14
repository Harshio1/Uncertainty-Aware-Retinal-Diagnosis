import gradio as gr
def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
print("Starting simple Gradio app...")
demo.launch(server_name="127.0.0.1", server_port=7861)
print("Gradio app launched.")
