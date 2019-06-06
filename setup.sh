#!/bin/bash

export DIR_TMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export PYTHONPATH=$PYTHONPATH:$DIR_TMP/src
export PYTHONPATH=$PYTHONPATH:$DIR_TMP/utest
export PYTHONPATH=$PYTHONPATH:$DIR_TMP/dep_packages
export PYTHONPATH=$PYTHONPATH:$DIR_TMP/dep_packages/DrQA

echo PYTHONPATH=$PYTHONPATH

#download spacy model for multiple languages
python -m spacy download en_core_web_sm

#install elasticsearch if not exists
