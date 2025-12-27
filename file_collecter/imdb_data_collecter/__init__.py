VERSION = "1.0.0"

from .imdb_reviews import crawl_imdb_reviews
from .imdb_rating import crawl_imdb_rating
__all__ = ["crawl_imdb_reviews", "crawl_imdb_rating"]