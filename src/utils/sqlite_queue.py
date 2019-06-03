import sqlite3
import queue
import threading


def synchronized(func):
    func.__lock__ = threading.Lock()

    def synced_func(*args, **kws):
        with func.__lock__:
            return func(*args, **kws)

    return synced_func


def singleton(cls):
    instances = {}

    @synchronized
    def _singleton(*args, **kw):
        if cls not in instances:
            instances[cls] = cls(*args, **kw)
        return instances[cls]

    return _singleton


@singleton
class SQLiteUtil(object):
    __queue_conn = queue.Queue(maxsize=1)
    __path = None

    def __init__(self, path):
        self.__path = path
        print('path:', self.__path)
        self.__create_conn()

    def __create_conn(self):
        conn = sqlite3.connect(self.__path, check_same_thread=False)
        self.__queue_conn.put(conn)

    def __close(self, cursor, conn):
        if cursor is not None:
            cursor.close()
        if conn is not None:
            cursor.close()
            self.__create_conn()

    def execute_query(self, sql, params):
        conn = self.__queue_conn.get()
        cursor = conn.cursor()
        value = None
        try:
            records = None
            if not params is None:
                records = cursor.execute(sql, params).fetchall()
            else:
                records = cursor.execute(sql).fetchall()
        finally:
            self.__close(cursor, conn)
        return records

    def executescript(self, sql):
        conn = self.__queue_conn.get()
        cursor = conn.cursor()
        try:
            cursor.executescript(sql)
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            self.__close(cursor, conn)

    def execute_update(self, sql, params):
        return self.execute_update_many([sql], [params])

    def execute_update_many(self, sql_list, params_list):
        conn = self.__queue_conn.get()
        cursor = conn.cursor()
        count = 0
        try:
            for index in range(len(sql_list)):
                sql = sql_list[index]
                params = params_list[index]
                if params is not None:
                    count += cursor.execute(sql, params).rowcount
                else:
                    count += cursor.execute(sql).rowcount
                    conn.commit()
        except Exception as e:
            conn.rollback()
            raise
        finally:
            self.__close(cursor, conn)
        return count