import os
import json
import logging
import boto3
from botocore.exceptions import ClientError


def handler(event, context):
    if "queryStringParameters" in event:
        action = event["queryStringParameters"]["action"]
        if action.lower() == "start":
            return {
                'statusCode': 200,
                'body': 'START'
            }
        elif action.lower() == "end":
            return {
                'statusCode': 200,
                'body': 'END'
            }
        else:
            return {
                'statusCode': 400,
                'body': 'uh-oh'
            }


    object_name = os.path.basename(event["params"]["querystring"]["file"])
    bucket_name = os.environ["BUCKET"]
    response = presigned_post(bucket_name, object_name)
    return {
        'statusCode': 200,
        'body': json.dumps(response)
    }


def presigned_post(
        bucket_name,
        object_name,
        fields=None,
        conditions=None,
        expiration=3600):
    """Generate a presigned URL S3 POST request to upload a file

    :param bucket_name: string
    :param object_name: string
    :param fields: Dictionary of prefilled form fields
    :param conditions: List of conditions to include in the policy
    :param expiration: Time in seconds for the presigned URL to remain valid
    :return: Dictionary with the following keys:
        url: URL to post to
        fields: Dictionary of form fields and values to submit with the POST
    :return: None if error.
    """

    # Generate a presigned S3 POST URL
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_post(
            bucket_name,
            object_name,
            Fields=fields,
            Conditions=conditions,
            ExpiresIn=expiration)
    except ClientError as e:
        logging.error(e)
        return None

    # The response contains the presigned URL and required fields
    return response
