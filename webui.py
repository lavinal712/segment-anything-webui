import argparse
import multiprocessing as mp
import os
import time
import cv2

import numpy as np
import torch

import gradio as gr
import gradio_image_prompter as gr_ext

from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry


def get_click_examples():
    assets_dir = os.path.join(os.path.dirname(__file__), "inputs")
    app_images = list(filter(lambda x: x.startswith("app_image"), os.listdir(assets_dir)))
    app_images.sort()
    return [{"image": os.path.join(assets_dir, x)} for x in app_images]


def on_reset_btn():
    click_img = gr.Image(None)
    anno_img = gr.AnnotatedImage(None)
    return click_img, anno_img


def on_submit_btn(click_img, sam_model):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam_models = {
        "ViT-H": "sam_vit_h_4b8939",
        "ViT-L": "sam_vit_l_0b3195",
        "ViT-B": "sam_vit_b_01ec64",
    }
    model_name = sam_model.lower().replace("-", "_")
    model_path = "checkpoints/" + sam_models[sam_model] + ".pth"
    sam = sam_model_registry[model_name](checkpoint=model_path)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    predictor = SamPredictor(sam)

    img, points = None, None
    if click_img is not None:
        img, points = click_img["image"], click_img["points"]
        points = np.array(points).reshape((-1, 2, 3))
    if img is None:
        img = np.zeros((480, 640), dtype="uint8")
        masks = np.zeros(img.shape)
        return img, [(masks, "mask")]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if points is None or points.size == 0:
        masks = mask_generator.generate(img)
        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        anns = []
        for i, mask in enumerate(sorted_masks):
            m = mask["segmentation"]
            anns.append((m, "mask" + str(i)))
        return click_img["image"], anns

    input_point = np.array([point[0][:2] for point in points], dtype=np.uint32)
    input_label = np.array([point[0][-1] for point in points], dtype=np.uint8)
    predictor.set_image(img)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )

    return click_img["image"], [(masks, "mask")]


title = "Segment Anything"
theme = "soft"
with gr.Blocks(title=title, theme=theme) as demo:
    gr.Markdown(
        "<div align='center'>"
        "<h1>Segment Anything</h1>"
        "</div>")
    with gr.Row():
        with gr.Column():
            click_img = gr_ext.ImagePrompter(show_label=False)
            interactions = "LeftClick (FG) | MiddleClick (BG) | PressMove (Box)"
            gr.Markdown("<h3 style='text-align: center'>[🖱️ | 🖐️]: 🌟🌟 {} 🌟🌟 </h3>".format(interactions))
            gr.Examples(get_click_examples(), inputs=[click_img])
            sam_model_radio = gr.Radio(
                ["ViT-H", "ViT-L", "ViT-B"], value="ViT-H", label="Segment Anything models"
            )
            with gr.Row():
                reset_btn = gr.Button("Reset")
                submit_btn = gr.Button("Execute")

        with gr.Column():
            anno_img = gr.AnnotatedImage(elem_id="anno-img", show_label=False)

    reset_btn.click(on_reset_btn, [], [click_img, anno_img])
    submit_btn.click(on_submit_btn, [click_img, sam_model_radio], [anno_img])

demo.launch()
