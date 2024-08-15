import sys
from config import DEFAULT_TABLE
from logs import LOGGER
from milvus_helpers import MilvusHelper
# from mysql_helpers import MySQLHelper
from encode import CLIP
from encode_chinese_clip import Chinese_CLIP
from minio_helpers import MinioHelper
def do_search(table_name:str, img_path:str, top_k:int, model:Chinese_CLIP, milvus_client:MilvusHelper, mysql_cli=None, minio_cli:MinioHelper=None):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        feat = model.clip_vit_base_patch16_extract_img_feat(img_path)
        vectors = milvus_client.search_vectors(table_name, feat, top_k)
        vids = [str(x.id) for x in vectors[0]]
        paths = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in vectors[0]]
        return paths, distances
    except Exception as e:
        LOGGER.error(f"Error with search:{e}")
        sys.exit(1)


def do_text2img_search(table_name:str, text:str, top_k:int, model:Chinese_CLIP, milvus_client:MilvusHelper, mysql_cli,
                       minio_cli: MinioHelper):
    try:
        if not table_name:
            table_name = DEFAULT_TABLE
        feat = model.clip_vit_base_patch16_extract_text_feat(list(text))
        # print(text.shape)
        vectors = milvus_client.search_vectors(table_name, [feat], top_k)
        ## vids = [str(x.id) for x in vectors[0]]

        paths = [str(x.entity.get("path")) for x in vectors[0]]
        urls =[]
        for i in range(len(paths)):
            url = minio_cli.generate_url(minio_cli.MINIO_DEFAULT_BUCKET, paths[i])
            urls.append(url)
        # paths = mysql_cli.search_by_milvus_ids(vids, table_name)
        distances = [x.distance for x in vectors[0]]
        return paths, distances, urls
    except Exception as e:
        LOGGER.error(f"Error with search:{e}")
        sys.exit(1)