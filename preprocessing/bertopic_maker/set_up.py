import pandas as pd
import numpy as np
import os
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from sklearn.cluster import AgglomerativeClustering

import warnings
warnings.filterwarnings('ignore')
os.environ['MallocStackLogging'] = '0'

load_dotenv()
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
DRAMA_FILE_PATH = os.getenv("DRAMA_FILE_PATH")
MOVIE_FILE_PATH = os.getenv("MOVIE_FILE_PATH")
HIT_FILE_PATH = os.getenv("HIT_FILE_PATH")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 기본 불용어 로드
base_stopwords = list(ENGLISH_STOP_WORDS)

# 드라마 불용어 설정

# 영화 불용어 설정
additional_movie_stopwords = [
    # ========== 영화 도메인 공통어 ==========
    'film', 'films', 'movie', 'movies', 'story', 'stories',
    'character', 'characters', 'scene', 'scenes', 'plot',
    'protagonist', 'audience', 'viewer', 'viewers',
    'series', 'sequel', 'part', 'chapter',
    'director', 'actor', 'actress', 'cast', 'crew',
    'documentary', 'footage', 'screen',

    # ========== 줄거리 서술 상투어 ==========
    'based', 'true', 'real', 'events', 'set',
    'follows', 'following', 'centers', 'revolves', 'tells',
    'takes', 'place', 'turns', 'finds', 'discovers', 'house',
    'begins', 'starts', 'ends', 'leads', 'brings', 'named', 'live', 'lives', 'meet', 'meets',

    # ========== 일반적 시간/수량 표현 ==========
    'time', 'times', 'year', 'years', 'day', 'days', 'night', 'nights',
    'moment', 'moments', 'later', 'ago', 'soon',
    'one', 'two', 'three', 'first', 'second', 'third', 'last',

    # ========== 일반적 인물 지칭 ==========
    'man', 'woman', 'men', 'women', 'people', 'person', 'guy', 'guys', 'self',
    'group', 'team', 'crew', 'members', 'girl', 'girls', 'boy', 'boys','dog', 'dogs',

    # ========== 기타 인물 이름 ==========
    'lena', 'jack', 'john', 'mary', 'sarah', 'mike', 'david', 'james', 'robert',
]

english_stopwords_movie = list(set(base_stopwords + additional_movie_stopwords))

embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')