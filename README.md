# VQGAN-CLIP-Docker

Zero-Shot Text-to-Image Generation VQGAN+CLIP Dockerized

## Usage

Configuration:

| Argument               | Type           | Descriptions                                                              |
|------------------------|----------------|---------------------------------------------------------------------------|
| `prompts`              | List[str]      | Text prompts                                                              |
| `image_prompts`        | List[FilePath] | Image prompts / target image path                                         |
| `max_iterations`       | int            | Number of iterations                                                      |
| `display_freq`         | int            | Save image iterations                                                     |
| `size`                 | [int, int]     | Image size (width height)                                                 |
| `init_image`           | FilePath       | Initial image                                                             |
| `init_noise`           | str            | Initial noise image ['gradient','pixels']                                 |
| `init_weight`          | float          | Initial weight                                                            |
| `clip_model`           | FilePath       | CLIP model path                                                           |
| `vqgan_checkpoint`     | FilePath       | VQGAN checkpoint path                                                     |
| `vqgan_config`         | FilePath       | VQGAN config path                                                         |
| `noise_prompt_seeds`   | List[int]      | Noise prompt seeds                                                        |
| `noise_prompt_weights` | List[float]    | Noise prompt weights                                                      |
| `step_size`            | float          | Learning rate                                                             |
| `cutn`                 | int            | Number of cuts                                                            |
| `cut_pow`              | float          | Cut power                                                                 |
| `seed`                 | int            | Seed                                                                      |
| `optimizer`            | str            | Optimiser ['Adam','AdamW','Adagrad','Adamax','DiffGrad','AdamP','RAdam']  |
| `output`               | FilePath       | Output file                                                               |
| `augments`             | List[str]      | Enabled augments ['Ji','Sh','Gn','Pe','Ro','Af','Et','Ts','Cr','Er','Re'] |

## Acknowledgments

[VQGAN+CLIP](https://github.com/nerdyrodent/VQGAN-CLIP)

[Taming Transformers](https://github.com/CompVis/taming-transformers)

[CLIP](https://github.com/openai/CLIP)

[DALLE-PyTorch](https://github.com/lucidrains/DALLE-pytorch)

## Citations

```bibtex
@misc{unpublished2021clip,
    title  = {CLIP: Connecting Text and Images},
    author = {Alec Radford, Ilya Sutskever, Jong Wook Kim, Gretchen Krueger, Sandhini Agarwal},
    year   = {2021}
}
```

```bibtex
@misc{esser2020taming,
      title={Taming Transformers for High-Resolution Image Synthesis},
      author={Patrick Esser and Robin Rombach and Björn Ommer},
      year={2020},
      eprint={2012.09841},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

```bibtex
@misc{ramesh2021zeroshot,
    title   = {Zero-Shot Text-to-Image Generation},
    author  = {Aditya Ramesh and Mikhail Pavlov and Gabriel Goh and Scott Gray and Chelsea Voss and Alec Radford and Mark Chen and Ilya Sutskever},
    year    = {2021},
    eprint  = {2102.12092},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```
