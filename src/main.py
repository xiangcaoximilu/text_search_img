import uvicorn
import os
from diskcache import Cache
from fastapi import FastAPI, File, UploadFile
from fastapi_offline import FastAPIOffline

from fastapi.param_functions import Form
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import FileResponse
# from encode import CLIP
from encode_chinese_clip import Chinese_CLIP
from milvus_helpers import MilvusHelper
# from mysql_helpers import MySQLHelper
from minio_helpers import MinioHelper
from config import TOP_K, UPLOAD_PATH
from operations.load import do_load
from operations.upload import do_upload
from operations.search import do_search, do_text2img_search
from operations.count import do_count
from operations.drop import do_drop
from logs import LOGGER
from pydantic import BaseModel
from typing import Optional
from urllib.request import urlretrieve
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.staticfiles import StaticFiles

app = FastAPIOffline(debug=True)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = Chinese_CLIP()  # 这里换模型
MILVUS_CLI = MilvusHelper()
# MYSQL_CLI = MySQLHelper()
MINIO_CLI = MinioHelper()


if not os.path.exists(UPLOAD_PATH):
    os.makedirs(UPLOAD_PATH)
    LOGGER.info(f"mkdir the path:{UPLOAD_PATH}")


@app.get("./data")
def get_img(img_path):
    # get the img file
    try:
        LOGGER.info(f"Successfully load image: {img_path}")
        return FileResponse(img_path)
    except Exception as e:
        LOGGER.error(f"Get image error: {e}")
        return {"status": False, "msg": e}, 400


@app.get('/progress')
def get_progress():
    try:
        cache = Cache("./tmp")
        return f"current: {cache['current']}, total: {cache['total']}"
    except Exception as e:
        LOGGER.error(f"upload image error: {e}")
        return {'status': False, 'msg': e}, 400


class Item(BaseModel):
    Table: Optional[str] = None
    File: str


@app.post("/img/load")
async def load_images(item: Item):
    # Insert all the image under the file path to Milvus/MySQL
    try:
        total_num = do_load(item.Table, item.File, MODEL, MILVUS_CLI, None, MINIO_CLI)
        LOGGER.info(f"Successfully loaded data, total count: {total_num}")
        return "Successfully loaded data!"
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post("/img/upload")
async def upload_image(image: UploadFile = File(None), url: str = None, table_name: str = None):
    # Insert the upload image to Milvus/MySQL
    try:
        # Save the upload image to server
        if image is not None:  # 直接传文件的形式
            print("image")
            content = await image.read()  # 读取上传文件的内容，并保存到content对象
            img_path = os.path.join(UPLOAD_PATH, image.filename)
            with open(img_path, "wb") as f:
                f.write(content)
        elif url is not None:  # 通过图片url的形式
            img_path = os.path.join(UPLOAD_PATH, os.path.basename(url))
            urlretrieve(url, img_path)
        else:
            return {"status": False, "msg": "Image and url are required"}, 400
        vector_id = do_upload(table_name, img_path, MODEL, MILVUS_CLI, None, MINIO_CLI)
        LOGGER.info(f"Successfully uploaded data, vector id: {vector_id}")
        return "Successfully loaded data: " + str(vector_id)
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post("/img/search")
async def search_images(image: UploadFile = File(...), topk: int = Form(TOP_K), table_name: str = None):
    try:
        # save upload image to server
        content = await image.read()
        img_path = os.path.join(UPLOAD_PATH, image.filename)
        with open(img_path, "wb+") as f:
            f.write(content)
        paths, distances = do_search(table_name, img_path, topk, MODEL, MILVUS_CLI, None)
        res = dict(zip(paths, distances))
        res = sorted(res.items(), key=lambda item: item[1])
        LOGGER.info("Successfully searched similar images!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post("/text_search_img/search")
async def search_images(text: str, topk: int = Form(TOP_K), table_name: str = None):
    try:

        paths, distances, urls = do_text2img_search(table_name, text, topk, MODEL, MILVUS_CLI, None, MINIO_CLI)
        #### res = dict(zip(paths, distances, urls))
        res = [{"path":a,"distance":b, "url":c}for a,b,c in zip(paths, distances, urls)]
        #### res = sorted(res.items(), key=lambda item: item[1])
        res = sorted(res, key=lambda x:x["distance"])
        LOGGER.info("Successfully searched similar images!")
        return res
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post("/img/count")
async def count_images(table_name: str = None):
    try:
        num = do_count(table_name, MILVUS_CLI)
        LOGGER.info("Successfully count the number of images!")
        return num
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400


@app.post("/img/drop")
async def drop_table(table_name: str = None):
    try:
        status = do_drop(table_name, MILVUS_CLI, None)
        LOGGER.info("Successfully drop tables in Milvus and MySQL!")
        return status
    except Exception as e:
        LOGGER.error(e)
        return {'status': False, 'msg': e}, 400



# 静态文件位置
static_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=f"{static_dir}/static"), name="static")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
    )


@app.get(app.swagger_ui_oauth2_redirect_url, include_in_schema=False)
async def swagger_ui_redirect():
    return get_swagger_ui_oauth2_redirect_html()


@app.get("/redoc", include_in_schema=False)
async def redoc_html():
    return get_redoc_html(
        openapi_url=app.openapi_url,
        title=app.title + " - ReDoc",
        redoc_js_url="/static/redoc/redoc.standalone.js",
    )


@app.get("/")
def read_root():
    return {"Hello": "World"}




if __name__ == '__main__':

    MILVUS_CLI.collection_detail()
    uvicorn.run(app=app, host="0.0.0.0", port=5009)
    # 每次程序运行结束都删除掉表
    # MILVUS_CLI.delete_collection("milvus_text2img_search")
