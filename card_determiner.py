import logging
from pprint import pprint
import boto3
from botocore.exceptions import ClientError
import boto3

logger = logging.getLogger(__name__)

rekognition_client = boto3.client('rekognition')

class RekognitionText:
    """Encapsulates an Amazon Rekognition text element."""

    def __init__(self, text_data):
        """
        Initializes the text object.

        :param text_data: Text data, in the format returned by Amazon Rekognition
                          functions.
        """
        self.text = text_data.get('DetectedText')
        self.kind = text_data.get('Type')
        self.id = text_data.get('Id')
        self.parent_id = text_data.get('ParentId')
        self.confidence = text_data.get('Confidence')
        self.geometry = text_data.get('Geometry')

    def to_dict(self):
        """
        Renders some of the text data to a dict.

        :return: A dict that contains the text data.
        """
        rendering = {}
        if self.text is not None:
            rendering['text'] = self.text
        if self.kind is not None:
            rendering['kind'] = self.kind
        if self.geometry is not None:
            rendering['polygon'] = self.geometry.get('Polygon')
        return rendering


client = boto3.client('rekognition')
import meilisearch

client = meilisearch.Client('http://127.0.0.1:7700', 'masterKey')

# An index is where the documents are stored.
index = client.index('cards')

class CardDeterminer:
    """ Stuff """

    def __init__(self) -> None:
        pass

    def detect_text(self, img_file_name) -> None:
        """
        Detects text in the image.

        :return The list of text elements found in the image.
        """
        try:
            with open(img_file_name, 'rb') as img_file:
                image = {'Bytes': img_file.read()}
            response = rekognition_client.detect_text(Image=image)
            texts = [RekognitionText(text)
                     for text in response['TextDetections']]

            logger.info("Found %s texts in %s.", len(texts), img_file_name)
        except ClientError:
            logger.exception("Couldn't detect text in %s.", img_file_name)
            raise
        else:
            results = index.search(texts[0].text)
            print(results)
            return results[0]
