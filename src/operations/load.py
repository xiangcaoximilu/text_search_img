import sys
import os
from diskcache import Cache
# from mysql_helpers import MySQLHelper
from config import DEFAULT_TABLE
from logs import LOGGER
from milvus_helpers import MilvusHelper
from minio_helpers import MinioHelper
def get_imgs(path):
    # 获取图片路径
    pics = []
    for f in os.listdir(path):
        if ((f.endswith(extension) for extension in
             [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]) and not f.startswith(".DS_Store")):
            pics.append(os.path.join(path, f))
    return pics


def extract_features(img_dir, model):
    '''

    :param img_dir:
    :param model:
    :return: features, names
    '''
    try:
        cache = Cache("./tmp")
        feats = []
        names = []
        img_list = get_imgs(img_dir)
        total = len(img_list)
        cache['total'] = total
        for i, img_path in enumerate(img_list):
            try:
                norm_feat = model.clip_vit_base_patch16_extract_img_feat(img_path)
                feats.append(norm_feat)
                names.append(img_path.encode())
                cache["current"] = i + 1
                LOGGER.info(f"Extracting feature from image No. {i + 1} , {total} images in total")
            except Exception as e:
                LOGGER.error(f"Error with extracting feature from image{e}")
                continue
        return feats, names

    except Exception as e:
        LOGGER.error(f"Error with extracting feature from image {e}")
        sys.exit(1)


def format_data(ids, names):
    data = [(str(i), n) for i, n in zip(ids, names)]
    return data


def do_load(table_name: str, image_dir: str, model, milvus_client: MilvusHelper, mysql_cli=None, minio_cli:MinioHelper=None):
    # 加载向量数据到milvus,加载元数据(图像路径)到mysql
    if not table_name:
        table_name = DEFAULT_TABLE
    if not milvus_client.has_collection(collection_name=table_name):
        milvus_client.create_collection(table_name)
        milvus_client.create_index(table_name)
    vectors, names = extract_features(image_dir, model)
    ids = milvus_client.insert(table_name, vectors)
    # mysql_cli.create_mysql_table(table_name)
    # mysql_cli.load_data_to_mysql(table_name, format_data(ids, names))
    return len(ids)
