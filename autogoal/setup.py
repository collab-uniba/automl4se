import nltk
nltk.download()

import os
from pathlib import Path

DATA_PATH = Path.home() / ".autogoal" / "contrib" / "transformers"
os.environ["TRANSFORMERS_CACHE"] = str(DATA_PATH)

try:
    import torch
    import transformers
except:
    print(
        "(!) Code in `autogoal.contrib.transformers` requires `pytorch==0.1.4` and `transformers==2.3.0`."
    )
    print("(!) You can install it with `pip install autogoal[transformers]`.")
    raise


from autogoal.contrib.transformers._bert import BertEmbedding
from autogoal.contrib import ContribStatus

def download():
    BertEmbedding.download()
    return True

def status():    

    try:
        BertEmbedding.check_files()
    except OSError:
        print("(!) BertEmbedding files not found.")
        return ContribStatus.RequiresDownload

    print("(i) BertEmbedding files found.")
    return ContribStatus.Ready

if (status() == ContribStatus.RequiresDownload):
    if(download()):
        print("(i) BertEmbedding files downloaded.")
        assert status() == ContribStatus.Ready
    else:
        print("(!) BertEmbedding files not downloaded.")
