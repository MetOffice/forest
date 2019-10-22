""" basic hello world lambda """
import json


def handler(event, context):
    """ Lambda Handler.
    Returns Hello World and the event and context objects
    """

    print(event)
    print(context)

    return {
        "body": json.dumps('Hello World!')
}
