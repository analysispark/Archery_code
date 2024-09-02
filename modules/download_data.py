import gdown

# FILE_ID = "1tdRa4mGWBXTC976LsS146c8o_rmttE1P"
FILE_ID = "1FTPtF7eIxanm8tQOqqH1eIS_86Q8VH4Q"
FILE_NAME = "DATA.zip"

gdown.download(id=FILE_ID, output=FILE_NAME, quiet=False)
