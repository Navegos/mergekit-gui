import pathlib
import random
import string
import tempfile
from typing import Iterable, List

import gradio as gr
import huggingface_hub
import torch
import yaml
from gradio_logsview.logsview import Log, LogsView, LogsViewRunner
from mergekit.config import MergeConfiguration

has_gpu = torch.cuda.is_available()

# Running directly from Python doesn't work well with Gradio+run_process because of:
# Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
# Let's use the CLI instead.
#
# import mergekit.merge
# from mergekit.common import parse_kmb
# from mergekit.options import MergeOptions
#
# merge_options = (
#     MergeOptions(
#         copy_tokenizer=True,
#         cuda=True,
#         low_cpu_memory=True,
#         write_model_card=True,
#     )
#     if has_gpu
#     else MergeOptions(
#         allow_crimes=True,
#         out_shard_size=parse_kmb("1B"),
#         lazy_unpickle=True,
#         write_model_card=True,
#     )
# )

cli = "mergekit-yaml config.yaml merge --copy-tokenizer" + (
    " --cuda --low-cpu-memory" if has_gpu else " --allow-crimes --out-shard-size 1B --lazy-unpickle"
)

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


## Citation

This GUI is powered by [Arcee's MergeKit](https://arxiv.org/abs/2403.13257).
If you use it in your research, please cite the following paper:

```
@article{goddard2024arcee,
  title={Arcee's MergeKit: A Toolkit for Merging Large Language Models},
  author={Goddard, Charles and Siriwardhana, Shamane and Ehghaghi, Malikeh and Meyers, Luke and Karpukhin, Vlad and Benedict, Brian and McQuade, Mark and Solawetz, Jacob},
  journal={arXiv preprint arXiv:2403.13257},
  year={2024}
}
```

This Space is heavily inspired by LazyMergeKit by Maxime Labonne (see [Colab](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb)).
"""

examples = [[str(f)] for f in pathlib.Path("examples").glob("*.yml")]


def merge(yaml_config: str, hf_token: str, repo_name: str) -> Iterable[List[Log]]:
    runner = LogsViewRunner()

    if not yaml_config:
        yield runner.log("Empty yaml, pick an example below", level="ERROR")
        return
    try:
        merge_config = MergeConfiguration.model_validate(yaml.safe_load(yaml_config))
    except Exception as e:
        yield runner.log(f"Invalid yaml {e}", level="ERROR")
        return

    if not hf_token:
        yield runner.log("You must provide a write-access token.", level="ERROR")
        return
    api = huggingface_hub.HfApi(token=hf_token)

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdir = pathlib.Path(tmpdirname)
        merged_path = tmpdir / "merged"
        merged_path.mkdir(parents=True, exist_ok=True)
        config_path = merged_path / "config.yaml"
        config_path.write_text(yaml_config)
        yield runner.log(f"Merge configuration saved in {config_path}")

        if not repo_name:
            yield runner.log("No repo name provided. Generating a random one.")
            repo_name = f"mergekit-{merge_config.merge_method}"
            # Make repo_name "unique" (no need to be extra careful on uniqueness)
            repo_name += "-" + "".join(random.choices(string.ascii_lowercase, k=7))
            repo_name = repo_name.replace("/", "-").strip("-")

        try:
            yield runner.log(f"Creating repo {repo_name}")
            repo_url = api.create_repo(repo_name, exist_ok=True)
            yield runner.log(f"Repo created: {repo_url}")
        except Exception as e:
            yield runner.log(f"Error creating repo {e}", level="ERROR")
            return

        yield from runner.run_command(cli.split(), cwd=merged_path)

        if runner.exit_code != 0:
            yield runner.log("Merge failed. Deleting repo as no model is uploaded.", level="ERROR")
            api.delete_repo(repo_url.repo_id)
            return

        yield runner.log("Model merged successfully. Uploading to HF.")
        yield from runner.run_python(
            api.upload_folder,
            repo_id=repo_url.repo_id,
            folder_path=merged_path / "merge",
        )
        yield runner.log("Model successfully uploaded to HF.")


with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN_DESCRIPTION)

    with gr.Row():
        filename = gr.Textbox(visible=False, label="filename")
        config = gr.Code(language="yaml", lines=10, label="config.yaml")
        with gr.Column():
            token = gr.Textbox(
                lines=1,
                label="HF Write Token",
                info="https://hf.co/settings/token",
                type="password",
                placeholder="Mandatory. Used to upload the merged model.",
            )
            repo_name = gr.Textbox(
                lines=1,
                label="Repo name",
                placeholder="Optional. Will create a random name if empty.",
            )
    button = gr.Button("Merge", variant="primary")
    logs = LogsView()
    gr.Examples(
        examples,
        fn=lambda s: (s,),
        run_on_click=True,
        label="Examples",
        inputs=[filename],
        outputs=[config],
    )
    gr.Markdown(MARKDOWN_ARTICLE)

    button.click(fn=merge, inputs=[config, token, repo_name], outputs=[logs])

demo.queue(default_concurrency_limit=1).launch()
