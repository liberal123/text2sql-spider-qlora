import sqlite3
import unittest
from text2sql_finetune.schema_utils import spider_schema_to_context


class SchemaTests(unittest.TestCase):
    def test_composite_key_produces_valid_create_table(self):
        schema = {
            'table_names_original': ['order items'],
            'column_names_original': [[-1, '*'], [0, 'order_id'], [0, 'item_id'], [0, 'a"b']],
            'column_types': ['text', 'number', 'number', 'text'],
            'primary_keys': [1, 2], 'foreign_keys': [],
        }
        context = spider_schema_to_context(schema)
        connection = sqlite3.connect(':memory:')
        try:
            connection.executescript(context)
            info = connection.execute('PRAGMA table_info("order items")').fetchall()
            self.assertEqual([col[5] for col in info], [1, 2, 0])
            self.assertEqual(info[2][1], 'a"b')
        finally:
            connection.close()


if __name__ == '__main__':
    unittest.main()
