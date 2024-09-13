import unittest
from app import app

class APITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_extract_keywords(self):
        response = self.app.post('/extract', json={'title': 'Understanding Natural Language Processing'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('keywords', response.get_json())

    def test_no_title(self):
        response = self.app.post('/extract', json={})
        self.assertEqual(response.status_code, 400)
        self.assertIn('error', response.get_json())

if __name__ == '__main__':
    unittest.main()
