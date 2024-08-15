import os
from towhee import pipe, ops, DataCollection, register
from towhee.operator import PyOperator
import glob
import numpy as np
import openvino as ov
import cv2
from clip_tokenizer import tokenize
from typing import List
from PIL import Image

core = ov.Core()


class VinoModel:
    def __init__(self, model_path, device="AUTO"):
        self.model = core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]
        print(f"input shape: {self.input_layer.shape}")
        print(f"input names: {self.input_layer.names}")
        self.compiled_model = core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled_model.output(0)
        print(f"output name: {self.output_layer.names}")

    def predict(self, input):
        result = self.compiled_model(input)[self.output_layer]
        return result


img_checkpoint_path = os.path.join(os.getcwd(), "model_checkpoint/image_model.onnx")
text_checkpoint_path = os.path.join(os.getcwd(), "model_checkpoint/text_model.onnx")
image_model = VinoModel(img_checkpoint_path)
text_model = VinoModel(text_checkpoint_path)


def preprocess(image: np.ndarray):
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255.0
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))  # hwc->chw
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image


def _img_feature(input_img: np.ndarray):
    image_features = image_model.predict(input_img)
    img_norm = np.linalg.norm(image_features, axis=-1, keepdims=True)
    image_features = image_features / img_norm
    return image_features


def _text_feature(input_text: List[str]):
    text = tokenize(input_text, context_length=52)
    text_features = []
    for i in range(len(input_text)):
        one_text = np.expand_dims(text[i], axis=0)
        text_feature = text_model.predict(one_text)[0]
        text_features.append(text_feature)
    text_features = np.stack(text_features, axis=0)
    text_norm = np.linalg.norm(text_features, axis=-1, keepdims=True)
    text_features = text_features / text_norm
    return text_features


@register
class img_embedding(PyOperator):
    def __call__(self, path):
        image = np.array(Image.open(path).convert("RGB"))
        return _img_feature(preprocess(image))[0]  # [0]去掉batch_size维度，bchw->chw,其中b=1


@register
class text_embedding(PyOperator):
    def __call__(self, input_strs):
        return _text_feature(input_strs)[0]


class Chinese_CLIP:
    def __init__(self):
        self.image_embedding_pipe = (
            pipe.input("file_name")
            .map("file_name", "embedding", ops.img_embedding())
            # .output("img", "vec")
            .output("embedding")
        )

        self.text_embedding_pipe = (
            pipe.input("text")
            .map("text", "embedding", ops.text_embedding())
            # .output("text", "vec")
            .output("embedding")
        )

    def clip_vit_base_patch16_extract_img_feat(self, image_path):
        feat = self.image_embedding_pipe(image_path)
        return feat.get()[0]

    def clip_vit_base_patch16_extract_text_feat(self, text: List[str]):
        feat = self.text_embedding_pipe(text)
        return feat.get()[0]


def run_clip(image: np.ndarray, input_strs: List[str]):
    image_features = _img_feature(preprocess(image))
    text_features = _text_feature(input_strs)
    logits_per_image = 100 * np.dot(image_features, text_features.T)
    exp_logits = np.exp(logits_per_image - np.max(logits_per_image, axis=-1, keepdims=True))
    max_logit = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    max_str = input_strs[max_logit.argmax()]
    max_str_logit = max_logit.max()
    return max_str, max_str_logit


if __name__ == '__main__':
    # 测试中文CLIP
    print(os.getcwd())
