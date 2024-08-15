from towhee import pipe, ops
import numpy as np
import openvino as ov

class CLIP:
    def __init__(self):
        # 调用本地模型
        # self.image_embedding_pipe = (
        #     pipe.input("path")
        #     .map("path", "img", ops.image_decode.cv2("rgb"))
        #     .map("img", "vec", ops.image_text_embedding.clip(model_name="clip-vit-base-patch16", modality="image",
        #                                                      checkpoint_path=r"H:\deploy\text2img_search_local_deploy\clip-vit-base-patch16"))
        #     .map("vec", "vec", lambda x: x / np.linalg.norm(x))
        #     .output("img", "vec")
        # )

        self.image_embedding_pipe = (
            pipe.input("path")
            .map("path", "img", ops.image_decode.cv2("rgb"))
            .map("img", "embedding", ops.image_text_embedding.clip(model_name="clip-vit-base-patch16", modality="image"))
            .map("embedding", "embedding", lambda x: x / np.linalg.norm(x))
            # .output("img", "vec")
            .output("embedding")
        )

        self.text_embedding_pipe = (
            pipe.input("text")
            .map("text", "embedding", ops.image_text_embedding.clip(model_name="clip_vit_base_patch16", modality="text"))
            .map("embedding", "embedding", lambda x: x / np.linalg.norm(x))
            # .output("text", "vec")
            .output("embedding")
        )

    def clip_vit_base_patch16_extract_img_feat(self, img_path):
        feat = self.image_embedding_pipe(img_path)
        return feat.get()[0]

    def clip_vit_base_patch16_extract_text_feat(self, text):
        feat = self.text_embedding_pipe(text)
        return feat.get()[0]





if __name__ == "__main__":
    MODEL = CLIP()
    MODEL.clip_vit_base_patch16_extract_img_feat(
        'https://raw.githubusercontent.com/towhee-io/towhee/main/towhee_logo.png')
