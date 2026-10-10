import unittest

from text2sql_finetune.formatting import extract_sql, is_readonly_select, normalize_sql


class FormattingTests(unittest.TestCase):
    def test_quoted_semicolon_is_not_truncated(self):
        self.assertEqual(extract_sql("```sql\nSELECT 'a;b';\n```"), "SELECT 'a;b';")

    def test_multiple_statements_are_not_silently_dropped(self):
        sql = "SELECT 1; DROP TABLE employees;"
        self.assertEqual(extract_sql(sql), sql)
        self.assertFalse(is_readonly_select(sql))

    def test_literal_case_and_spaces_survive_normalization(self):
        self.assertNotEqual(normalize_sql("SELECT 'ABC'"), normalize_sql("SELECT 'abc'"))
        self.assertNotEqual(normalize_sql("SELECT 'a  b'"), normalize_sql("SELECT 'a b'"))

    def test_readonly_keyword_in_literal(self):
        self.assertTrue(is_readonly_select("SELECT 'update; drop table'"))
        self.assertFalse(is_readonly_select("WITH t AS (SELECT 1) DELETE FROM employees"))


if __name__ == '__main__':
    unittest.main()
