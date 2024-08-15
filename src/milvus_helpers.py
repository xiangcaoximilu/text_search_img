import sys
from config import MILVUS_HOST, MILVUS_PORT, VECTOR_DIMENSION, METRIC_TYPE, DEFAULT_TABLE
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from logs import LOGGER


class MilvusHelper:
    def __init__(self):
        try:
            self.collection = None
            connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
            LOGGER.debug(f"Successfully connect to Milvus with IP:{MILVUS_HOST} and PORT:{MILVUS_PORT}")
        except Exception as e:
            LOGGER.error(f"Failed to connect Milvus:{e}")
            sys.exit(1)

    def set_collection(self, collection_name):
        try:
            self.collection = Collection(name=collection_name)
        except Exception as e:
            LOGGER.error(f"Failed to set collection in Milvus:{e}")
            sys.exit(1)

    def has_collection(self, collection_name):
        try:
            return utility.has_collection(collection_name=collection_name)
        except Exception as e:
            LOGGER.error(f"Failed to get collection info to Milvus:{e}")
            sys.exit(1)

    def create_collection(self, collection_name):
        # Create milvus collection if not exists
        try:
            field1 = FieldSchema(name="id", dtype=DataType.INT64, description="int64", is_primary=True, auto_id=True)
            field2 = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, description="float vector",
                                 dim=VECTOR_DIMENSION, is_primary=False)
            field3 = FieldSchema(name="path", dtype=DataType.VARCHAR, description="file path",
                                 max_length=200)
            schema = CollectionSchema(fields=[field1, field2, field3], description="collection description")
            self.collection = Collection(name=collection_name, schema=schema)
            LOGGER.debug(f"Create Milvus collection: {collection_name}")
            return "OK"
        except Exception as e:
            LOGGER.error(f"Failed create collection in Milvus: {e}")
            sys.exit(1)

    def insert(self, collection_name, vectors, obj_name):
        # Batch insert vectors to milvus collection
        try:
            self.set_collection(collection_name)
            data = [[vectors], [obj_name]]
            mr = self.collection.insert(data)
            ids = mr.primary_keys
            LOGGER.debug(
                f"Insert vectors to Milvus in collection:{collection_name} with {len(vectors)} rows"
            )
            return ids
        except Exception as e:
            LOGGER.error(f"Failed to insert data to Milvus:{e}")
            sys.exit(1)

    def create_index(self, collection_name):
        try:
            self.set_collection(collection_name)
            default_index = {"metric_type": METRIC_TYPE, "index_type": "IVF_FLAT", "params": {"nlist": 2048}}
            status = self.collection.create_index(field_name="embedding", index_params=default_index)
            if not status.code:
                LOGGER.debug(
                    f"Successfully create index in collection:{collection_name} with param:{default_index}"
                )
                return status
            else:
                raise Exception(status.message)
        except Exception as e:
            LOGGER.error(f"Failed to create index: {e}")
            sys.exit(1)

    def delete_collection(self, collection_name):
        try:
            self.set_collection(collection_name)
            self.collection.drop()
            LOGGER.debug("Successfully drop collection!")
            return "ok"
        except Exception as e:
            LOGGER.error(f"Failed to drop collection: {e}")
            sys.exit(1)

    def search_vectors(self, collection_name, vectors, top_k):
        try:
            self.set_collection(collection_name)
            self.collection.load()
            search_param = {"metric_type": METRIC_TYPE, "params": {"nprobe": 16}}
            res = self.collection.search(vectors, anns_field="embedding", param=search_param, limit=top_k,
                                         output_fields=["path"])
            LOGGER.debug(f"Successfully search in collection: {res}")
            return res
        except Exception as e:
            LOGGER.error(f"Failed to search vectors in Milvus: {e}")
            sys.exit(1)

    def count(self, collection_name):
        try:
            self.set_collection(collection_name)
            self.collection.flush()
            num = self.collection.num_entities
            LOGGER.debug(f"Successfully get the num:{num} of the collection:{collection_name}")
            return num
        except Exception as e:
            LOGGER.error(f"Failed to count vectors in Milvus: {e}")
            sys.exit(1)

    def collection_detail(self):
        res = utility.list_collections()
        print(res)
        # if "milvus_text2img_search" in res:
        #     print("更新collection")
        #     self.delete_collection(DEFAULT_TABLE)
        #
        #     res = utility.list_collections()
        #     print(res)
