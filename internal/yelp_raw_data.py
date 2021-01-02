import os
import io
import csv
import tarfile

"""
Usage:

sys.path.insert(0, '.../Yelp_Dataset/')
from internal.yelp_raw_data import YelpRawData

with YelpRawData('.../Yelp_Dataset/data/raw/') as yrd:
    for v in islice(yrd, 10):
        print(v)
"""


class YelpRawData:

    def __init__(self, location='data/raw', data='polarity', mode='train'):
        if data not in ['polarity', 'full']:
            raise ValueError(f'Unknown data type {data}')
        if mode not in ['train', 'test']:
            raise ValueError(f'Unknown mode type {mode}')
        self.location = location
        self.data = data
        self.mode = mode

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __iter__(self):
        rawFile = f'{self.location}/yelp_review_{self.data}_csv.tgz'
        csvName = f'yelp_review_{self.data}_csv/{self.mode}.csv'
        if not os.path.exists(rawFile):
            raise ValueError('{rawFile} does not exist, check the location or run get_raw_data.sh to download it from S3.')
        with tarfile.open(rawFile, "r:*") as tar:
            csvMember = next((member for member in tar.getmembers() if member.name == csvName), None)
            if csvMember is None:
                raise ValueError('{csvName} not found in {rawFile}')
            reader = csv.reader(io.TextIOWrapper(tar.extractfile(csvMember), encoding='utf-8'))
            for reviewId, line in enumerate(reader):
                yield {
                    'reviewId': reviewId,
                    'score': line[0],
                    'text': line[1].replace('\\"', '"').replace('\\n', '\n')
                }
