import boto3
from botocore.exceptions import ClientError
import mimetypes

def upload_image_to_s3(local_path, bucket, key, region="us-east-2"):
    s3 = boto3.client("s3", region_name=region)

    # Properly guess the MIME type from file extension
    content_type, _ = mimetypes.guess_type(local_path)
    if content_type is None:
        content_type = "image/png"

    try:
        s3.upload_file(
            Filename=local_path,
            Bucket=bucket,
            Key=key,
            ExtraArgs={
                "ContentType": content_type
            }
        )
        return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"
    except ClientError as e:
        raise RuntimeError(f"Failed to upload {local_path} to {bucket}/{key}: {e}")
