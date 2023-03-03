from env import DB_NAME, DB_PASS, DB_USER, DB_HOST
import psycopg2
class Database:
    def __init__(self):
        self.conn = None
        self.cusrsor = None
        self.DB_NAME = DB_NAME
        self.DB_USER = DB_USER
        self.DB_PASS = DB_PASS
        self.DB_HOST = DB_HOST

    def connect(self):
        self.conn = psycopg2.connect(

            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASS,
            host=DB_HOST

        )

        self.cursor = self.conn.cursor()

    def write(self, records):
        query = """INSERT INTO IMAGES_PROCESSED (IMAGE_NAME) VALUES (%s);"""

        self.cursor.execute(query, (records,))
        self.conn.commit()


