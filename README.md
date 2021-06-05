## RuSimScore
Code for paper "RuSimScore: unsupervised scoring function for Russian sentence simplification quality"

## Install
`python -r requirements.py`

`chmod +x ./download.sh && ./download.sh`

## Training

`python prepare-train.py`

`train.sh`

## Download best model

Best model can be downloaded from https://huggingface.co/orzhan/rugpt3-simplify-large

## Running

`python generate.py --input-text "14 апреля 2003 году архиепископом Новосибирским и Бердским Тихоном пострижен в монашество с наречением имени Феодор в честь праведного Феодора Томского."`

`python generate.py --input-file hidden_test.csv --control 5`