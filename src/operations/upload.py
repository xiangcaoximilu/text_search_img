import os.path
import sys
from config import DEFAULT_TABLE
from logs import LOGGER
from milvus_helpers import MilvusHelper
# from mysql_helpers import MySQLHelper
from encode import CLIP
from encode_chinese_clip import Chinese_CLIP
from minio_helpers import MinioHelper

def do_upload(table_name: str, img_path: str, model: Chinese_CLIP, milvus_client: MilvusHelper, mysql_cli=None,
              minio_cli: MinioHelper=None):
    try:
        # debug
        milvus_client.collection_detail()
        # milvus
        if not table_name:
            table_name = DEFAULT_TABLE
        if not milvus_client.has_collection(table_name):
            milvus_client.create_collection(table_name)
            milvus_client.create_index(table_name)

        # minio
        if not minio_cli.has_bucket(minio_cli.MINIO_DEFAULT_BUCKET):
            minio_cli.create_bucket(bucket_name=minio_cli.MINIO_DEFAULT_BUCKET)
        object_name = str(os.path.basename(img_path))
        minio_cli.upload_img(img_path, minio_cli.MINIO_DEFAULT_BUCKET, object_name=object_name)

        feat = model.clip_vit_base_patch16_extract_img_feat(img_path)
        ids = milvus_client.insert(table_name, feat, object_name)
        # mysql_cli.create_mysql_table(table_name)
        # mysql_cli.load_data_to_mysql(table_name, [(str(ids[0]), img_path.encode())])
        return ids[0]
    except Exception as e:
        LOGGER.error(f"Error with upload : {e}")
        sys.exit(1)
