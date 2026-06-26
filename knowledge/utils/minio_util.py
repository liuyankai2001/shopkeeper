import os
import logging
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
load_dotenv()


def get_minio_client():
    # 1.实例化minio客户端
    try:
        client = Minio(
            endpoint=os.getenv("MINIO_ENDPOINT",""),
            access_key=os.getenv("MINIO_ACCESS_KEY"),
            secret_key=os.getenv("MINIO_SECRET_KEY"),
            secure=False  # http协议
        )
        # 2.判断桶是否存在
        bucket_name = os.getenv("MINIO_BUCKET_NAME")
        found = client.bucket_exists(bucket_name)
        if not found:
            client.make_bucket(bucket_name)
            logging.info(f"桶：{bucket_name} 不存在")
        else:
            print("Bucket",bucket_name,"already exists")
        return client
    except S3Error as e:
        logging.error("minio客户端创建失败")
        return None

if __name__ == '__main__':
    client = get_minio_client()
    print(client)