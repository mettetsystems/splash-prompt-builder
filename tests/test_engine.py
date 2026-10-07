import unittest

from backend.engine import MockEngine, python_cursor, utf16_length


class EngineTests(unittest.TestCase):
    def test_exact_anchors_and_output(self):
        for text in ('', '   \n\t', 'A  fox,\nunder\tthe 🌙.', 'été 世界 café!', 'foo foo foo foo', '<script>alert(1)</script>'):
            with self.subTest(text=text):
                result = MockEngine().update(text, utf16_length(text))
                self.assertEqual(''.join(s['text'] for s in result['segments'] if s['kind'] == 'anchor'), text)
                self.assertEqual(''.join(s['text'] for s in result['segments']), result['output'])

    def test_frozen_infill_survives_insertion_and_deletion(self):
        engine = MockEngine()
        text = 'one two three four five six seven eight nine ten'
        engine.update(text, len(text), radius=1)
        cached = engine.additions.copy()
        new_text = 'new ' + text
        result = engine.update(new_text, 0, radius=1)
        self.assertEqual(engine.additions[2:], cached[1:])
        self.assertTrue(any(s.get('frozen') for s in result['segments']))
        engine.update(text, 0, radius=1)
        self.assertEqual(engine.additions[2:], cached[2:])

    def test_active_details_can_change(self):
        engine = MockEngine()
        engine.update('forest at dawn', 0, radius=1)
        first = engine.additions[0]
        variants = []
        for word in ('night', 'sunset', 'midnight', 'morning', 'dusk', 'winter'):
            engine.update(f'forest {word} dawn', 0, radius=1)
            variants.append(engine.additions[0])
        self.assertTrue(any(value != first for value in variants))

    def test_repeated_words_keep_frozen_cache_after_insertion(self):
        engine = MockEngine()
        text = ' '.join(['same'] * 250)
        engine.update(text, len(text), radius=1)
        cached = engine.additions.copy()
        engine.update('new ' + text, 0, radius=1)
        self.assertEqual(engine.additions[2:], cached[1:])

    def test_commit_turns_everything_into_anchors(self):
        engine = MockEngine()
        output = engine.update('a quiet lake', 12)['output']
        committed = engine.update(output, len(output), commit=True)
        self.assertEqual(committed['output'], output)
        self.assertTrue(all(s['kind'] == 'anchor' for s in committed['segments']))
        # Repeated cursor updates of a committed prompt do not re-enrich it.
        self.assertEqual(engine.update(output, 0, commit=True)['output'], output)

    def test_independent_deterministic_sessions(self):
        self.assertEqual(MockEngine().update('quiet lake', 10), MockEngine().update('quiet lake', 10))

    def test_utf16_cursor(self):
        self.assertEqual(utf16_length('🌙 a'), 4)
        self.assertEqual(python_cursor('🌙 a', 3), 2)
        result = MockEngine().update('🌙 a moon', 4, radius=0)
        self.assertEqual(''.join(s['text'] for s in result['segments'] if s['kind'] == 'anchor'), '🌙 a moon')


if __name__ == '__main__':
    unittest.main()
