import json
import unittest
from types import SimpleNamespace

from modal_app.full_processor import (
    build_extraction_chunks,
    call_openai_structured,
    context_evidence_is_source_bounded,
    deduplicate_candidates,
    normalize_text,
)


class PipelineQualityTests(unittest.TestCase):
    def test_normalize_text_handles_smart_punctuation(self):
        self.assertEqual(normalize_text('“Supply—path” isn\'t neutral.'), 'supply path isnt neutral')

    def test_extraction_chunks_preserve_global_ids_and_overlap(self):
        segments = [
            {'id': index, 'text': f'segment {index}', 'start': index * 2, 'end': index * 2 + 1}
            for index in range(10)
        ]
        chunks = build_extraction_chunks(segments, max_chars=48, overlap_segments=2)
        self.assertGreater(len(chunks), 1)
        self.assertIn('[0] segment 0', chunks[0])
        first_tail = chunks[0].splitlines()[-2:]
        self.assertEqual(first_tail, chunks[1].splitlines()[:2])

    def test_deduplication_keeps_stronger_near_duplicate(self):
        weak = {
            'text': 'The open web needs a more efficient supply path.',
            'start_segment_id': 10,
            'end_segment_id': 12,
            'domain_specificity': 0.5,
            'novelty': 0.4,
            'provocation': 0.3,
            'evidence_quality': 0.8,
        }
        strong = {
            **weak,
            'text': 'The open web needs a much more efficient supply path.',
            'domain_specificity': 0.9,
            'novelty': 0.8,
        }
        result = deduplicate_candidates([weak, strong])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['text'], strong['text'])

    def test_direct_context_evidence_must_stay_within_quote_span(self):
        valid = [
            {'evidence_type': 'direct_transcript', 'segment_ids': [7, 8]},
            {'evidence_type': 'domain_inference', 'segment_ids': []},
        ]
        invalid = [{'evidence_type': 'direct_transcript', 'segment_ids': [7, 11]}]
        self.assertTrue(context_evidence_is_source_bounded(valid, 7, 9))
        self.assertFalse(context_evidence_is_source_bounded(invalid, 7, 9))

    def test_structured_call_uses_strict_schema_and_no_storage(self):
        calls = []

        class Responses:
            def create(self, **kwargs):
                calls.append(kwargs)
                return SimpleNamespace(status='completed', output_text=json.dumps({'items': []}))

        client = SimpleNamespace(responses=Responses())
        schema = {
            'type': 'object',
            'properties': {'items': {'type': 'array', 'items': {'type': 'string'}}},
            'required': ['items'],
            'additionalProperties': False,
        }
        result = call_openai_structured(
            client,
            model='test-model',
            system_prompt='system',
            user_prompt='user',
            schema_name='test_schema',
            schema=schema,
            reasoning_effort='high',
        )
        self.assertEqual(result, {'items': []})
        self.assertTrue(calls[0]['text']['format']['strict'])
        self.assertFalse(calls[0]['store'])


if __name__ == '__main__':
    unittest.main()
