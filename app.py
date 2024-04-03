import pathlib
import tempfile
from typing import Generator

import gradio as gr
import torch
import yaml
from gradio_logsview import LogsView

has_gpu = torch.cuda.is_available()

cli = "mergekit-yaml config.yaml merge --copy-tokenizer" + (
    " --cuda --low-cpu-memory"
    if has_gpu
    else " --allow-crimes --out-shard-size 1B --lazy-unpickle"
)

print(cli)

## This Space is heavily inspired by LazyMergeKit by Maxime Labonne
## https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb


MARKDOWN_DESCRIPTION = """
# mergekit-gui

The fastest way to perform a model merge 🔥

Specify a YAML configuration file (see examples below) and a HF token and this app will perform the merge and upload the merged model to your user profile.
"""

MARKDOWN_ARTICLE = """
___

## Merge Configuration

[Mergekit](https://github.com/arcee-ai/mergekit) configurations are YAML documents specifying the operations to perform in order to produce your merged model.
Below are the primary elements of a configuration file:

- `merge_method`: Specifies the method to use for merging models. See [Merge Methods](https://github.com/arcee-ai/mergekit#merge-methods) for a list.
- `slices`: Defines slices of layers from different models to be used. This field is mutually exclusive with `models`.
- `models`: Defines entire models to be used for merging. This field is mutually exclusive with `slices`.
- `base_model`: Specifies the base model used in some merging methods.
- `parameters`: Holds various parameters such as weights and densities, which can also be specified at different levels of the configuration.
- `dtype`: Specifies the data type used for the merging operation.
- `tokenizer_source`: Determines how to construct a tokenizer for the merged model.

## Merge Methods

A quick overview of the currently supported merge methods:

| Method                                                                                       | `merge_method` value | Multi-Model | Uses base model |
| -------------------------------------------------------------------------------------------- | -------------------- | ----------- | --------------- |
| Linear ([Model Soups](https://arxiv.org/abs/2203.05482))                                     | `linear`             | ✅          | ❌              |
| SLERP                                                                                        | `slerp`              | ❌          | ✅              |
| [Task Arithmetic](https://arxiv.org/abs/2212.04089)                                          | `task_arithmetic`    | ✅          | ✅              |
| [TIES](https://arxiv.org/abs/2306.01708)                                                     | `ties`               | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [TIES](https://arxiv.org/abs/2306.01708)            | `dare_ties`          | ✅          | ✅              |
| [DARE](https://arxiv.org/abs/2311.03099) [Task Arithmetic](https://arxiv.org/abs/2212.04089) | `dare_linear`        | ✅          | ✅              |
| Passthrough                                                                                  | `passthrough`        | ❌          | ❌              |
| [Model Stock](https://arxiv.org/abs/2403.19522)                                              | `model_stock`        | ✅          | ✅              |

"""

examples = [[f.name, f.read_text()] for f in pathlib.Path("examples").glob("*.yml")]


def merge(
    example_filename: str, yaml_config: str, hf_token: str, repo_name: str
) -> Generator[str, None, None]:
    if not yaml_config:
        raise gr.Error("Empty yaml, pick an example below")
    try:
        _ = yaml.safe_load(yaml_config)
    except Exception as e:
        raise gr.Error(f"Invalid yaml {e}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = pathlib.Path(tmpdirname)

        config_path = tmpdir / "config.yaml"
        config_path.write_text(yaml_config)

        cmd = f"cd {tmpdir} && {cli}"
        print(cmd)
        print(tmpdir.is_dir())
        yield from LogsView.run_process(cmd.split())

        ## TODO(implement upload at the end of the merge, and display the repo URL)


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN_DESCRIPTION)

    with gr.Row():
        filename = gr.Textbox(visible=False, label="filename")
        config = gr.Code(
            language="yaml",
            lines=10,
            label="config.yaml",
        )
        with gr.Column():
            token = gr.Textbox(
                lines=1,
                label="HF Write Token",
                info="https://hf.co/settings/token",
                type="password",
                placeholder="optional, will not upload merge if empty (dry-run)",
            )
            repo_name = gr.Textbox(
                lines=1,
                label="Repo name",
                placeholder="optional, will create a random name if empty",
            )
    button = gr.Button("Merge", variant="primary")
    logs = LogsView()
    gr.Examples(examples, label="Examples", inputs=[filename, config], outputs=[logs])
    gr.Markdown(MARKDOWN_ARTICLE)

    button.click(fn=merge, inputs=[filename, config, token, repo_name], outputs=[logs])

demo.queue(default_concurrency_limit=1).launch()
