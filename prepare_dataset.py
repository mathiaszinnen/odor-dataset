import wget, zipfile
import os


def prepare_folders():
    if not os.path.isdir('data'):
        os.makedirs('data')

def download_dataset():
    dataset_url = 'https://zenodo.org/api/records/11070878/files-archive'
    fn = wget.download(dataset_url, out='data/', bar=None)
    return fn

def unzip(fn, target):
    with zipfile.ZipFile(fn) as zf:
        zf.extractall(target)


def cleanup(fn):
    os.remove(fn)
    os.remove('data/images.zip')

if __name__ == '__main__':
    print('Preparing folders..')
    prepare_folders()

    print('Downloading dataset..')
    fn = download_dataset()
    
    print('Unzipping dataset..')
    unzip(fn, 'data/')

    unzip('data/images.zip', 'data/')

    print('Removing temporary files')
    cleanup(fn)

    print('Dataset succesfully prepared in data/')


