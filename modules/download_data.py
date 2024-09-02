import gdown

FILE_ID = "1srZ31CdVq6e4IlPT9ZwzzfmEGnkqmE69"
FILE_NAME = "DATA.zip"

gdown.download(id=FILE_ID, output=FILE_NAME, quiet=False)
