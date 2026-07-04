import threading

import pymysql
from pymysql.cursors import DictCursor


class MysqlConfig:
    def __init__(self):
        self._lock = threading.Lock()
        self.conn = self._create_connection()
        self.cursor = self.conn.cursor(DictCursor)

    def _create_connection(self):
        return pymysql.connect(
            host="127.0.0.1",
            port=3306,
            user="root",
            password="root",
            database="pydb",
            charset="utf8mb4",
            autocommit=False,
            connect_timeout=10,
            read_timeout=30,
            write_timeout=30,
        )

    def _ensure_connection(self) -> None:
        """长任务后连接可能超时，执行前检测并在必要时重连。"""
        try:
            self.conn.ping(reconnect=True)
        except pymysql.Error:
            self.conn = self._create_connection()
            self.cursor = self.conn.cursor(DictCursor)

    def _safe_rollback(self) -> None:
        try:
            if self.conn and self.conn.open:
                self.conn.rollback()
        except pymysql.Error as exc:
            print(f"回滚失败: {exc}")

    def query(self, sql: str, params: tuple | list | None = None):
        with self._lock:
            try:
                self._ensure_connection()
                self.cursor.execute(sql, params)
                return self.cursor.fetchall()
            except pymysql.Error as exc:
                self._safe_rollback()
                print(f"查询出错: {exc}")
                return None

    def insert(self, sql: str, params: tuple | list | None = None) -> bool:
        """
        通用插入方法。
        :return: 成功返回 True，失败返回 False（不会向外抛出 pymysql 异常）
        """
        with self._lock:
            try:
                self._ensure_connection()
                self.cursor.execute(sql, params)
                self.conn.commit()
                return True
            except pymysql.Error as exc:
                self._safe_rollback()
                print(f"插入出错: {exc}")
                return False

    def execute(self, sql: str, params: tuple | list | None = None) -> int:
        """执行 INSERT/UPDATE/DELETE，成功时返回受影响行数。"""
        with self._lock:
            self._ensure_connection()
            self.cursor.execute(sql, params)
            self.conn.commit()
            return self.cursor.rowcount

    def close(self) -> None:
        with self._lock:
            if self.cursor:
                self.cursor.close()
            if self.conn and self.conn.open:
                self.conn.close()
