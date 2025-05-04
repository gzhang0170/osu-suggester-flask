import os, pymysql, functools
from contextlib import contextmanager

@functools.lru_cache(maxsize=1)
def _conn():
    return pymysql.connect(
        host=os.environ["DB_HOST"],
        user=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        database=os.environ["DB_NAME"],
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

@contextmanager
def db():
    conn = _conn()
    cur = conn.cursor()
    try:
        yield cur
    finally:
        cur.close()
