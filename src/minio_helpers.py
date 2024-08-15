from minio import Minio
from minio.deleteobjects import DeleteObject
from datetime import timedelta
from logs import LOGGER
# from config import
import sys
import os
from typing import List
from config import MINIO_DEFAULT_BUCKET, MINIO_ACCESS, MINIO_SECRET, MINIO_ENDPOINT


class MinioHelper:
    MINIO_DEFAULT_BUCKET = MINIO_DEFAULT_BUCKET

    def __init__(self):
        try:
            self.client = Minio(
                endpoint=MINIO_ENDPOINT,
                access_key=MINIO_ACCESS,
                secret_key=MINIO_SECRET,
                #region="wjq-region",
                secure=False
            )

            LOGGER.debug(f"Successfully connect to Minio with {MINIO_ENDPOINT}")
            print("连接到minio")
            # self.client.bucket_exists("test")
        except Exception as e:
            LOGGER.error(f"Failed to connect Minio:{e}")
            sys.exit(1)

    def create_bucket(self, bucket_name):
        try:
            self.client.make_bucket(bucket_name)
            LOGGER.debug(f"Successfully make bucket:{bucket_name}")
        except Exception as e:
            LOGGER.error(f"Failed to make bucket:{e}")
            sys.exit(1)

    def has_bucket(self, bucket_name):
        try:
            return self.client.bucket_exists(bucket_name)

        except Exception as e:
            LOGGER.error(f"Failed to get bucket:{e}")
            sys.exit(1)

    def list_buckets(self):
        try:
            return self.client.list_buckets()
        except Exception as e:
            LOGGER.error(f"Failed to get list of buckets:{e}")
            sys.exit(1)

    def remove_bucket(self, bucket_name):
        # note: remove an {empty} bucket
        try:
            self.client.remove_bucket(bucket_name)
            LOGGER.debug(f"Successfully remove empty bucket:{bucket_name}")
        except Exception as e:
            LOGGER.error(f"Failed to remove bucket:{e}")
            sys.exit(1)

    def remove_object(self, bucket_name, object_name, version_id=None):
        try:
            self.client.remove_object(bucket_name, object_name, version_id=version_id)
            LOGGER.debug(f"Successfully remove object:{object_name}")
        except Exception as e:
            LOGGER.error(f"Failed remove object:{e}")
            sys.exit(1)

    def remove_objects(self, bucket_name: str, prefix):
        try:
            delete_object_list = map(lambda x: DeleteObject(x.object_name),
                                     self.client.list_objects(bucket_name, prefix, recursive=True))
            errors = self.client.remove_objects(bucket_name, delete_object_list)
            for error in errors:
                print(f"error occurred when deleting object:{error}")
            LOGGER.debug(f"Successfully remove object")
        except Exception as e:
            LOGGER.error(f"Failed remove objects:{e}")
            sys.exit(1)

    def list_bucket_objects(self, bucket_name):
        try:
            objs = self.client.list_objects(bucket_name=bucket_name, prefix=None, recursive=False, start_after=None,
                                            include_user_meta=False, include_version=False, use_api_v1=False,
                                            use_url_encoding_type=True)
            for obj in objs:
                # print(obj)
                LOGGER.info(f"obj: {obj}")
        except Exception as e:
            LOGGER.error(f"Failed to list bucket objs:{e}")
            sys.exit(1)

    def upload_imgs(self, img_path, bucket_name, object_name, ):
        try:
            imgs = os.listdir(img_path)
            for img in imgs:
                self.client.fput_object(bucket_name, object_name, os.path.join(img_path, img))
            LOGGER.debug(f"Successfully upload {len(imgs)} imgs to Minio")
        except Exception as e:
            LOGGER.error(f"Failed to upload imgs:{e}")
            sys.exit(1)

    def upload_img(self, img_path, bucket_name, object_name, ):
        try:
            res = self.client.fput_object(bucket_name, object_name, img_path)
            print(res)
            LOGGER.debug(f"Successfully upload img:{img_path} to Minio")
        except Exception as e:
            LOGGER.error(f"Failed to upload imgs:{e}")
            sys.exit(1)

    def generate_url(self,bucket_name, object_name, expires=7):
        expires = timedelta(days=expires)
        url = self.client.presigned_get_object(bucket_name, object_name,expires)
        return url

    def minio_detail(self):
        ls = self.client.list_buckets()
        LOGGER.info(f"there is/are {len(ls)} bucket/s")
        for l in ls:
            LOGGER.info(f"bucker name:{l.name}, bucket creation_date:{l.creation_date}")


if __name__ == '__main__':
    minio = MinioHelper()
    minio.create_bucket(minio.MINIO_DEFAULT_BUCKET)