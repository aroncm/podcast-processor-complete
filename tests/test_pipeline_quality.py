import json
import unittest
from types import SimpleNamespace

from modal_app.full_processor import (
    build_extraction_chunks,
    build_caption_evidence,
    align_quote_to_segments,
    call_openai_structured,
    conversation_mapping_is_reviewable,
    context_evidence_is_source_bounded,
    deduplicate_candidates,
    fetch_conversation_taxonomy,
    historical_mapping_is_reviewable,
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

    def test_conversation_mapping_requires_supported_named_connections(self):
        mapping = {
            'theme_name': 'Performance TV',
            'theme_summary': 'How television is being connected to business outcomes.',
            'question_text': 'What should performance accountability look like on TV?',
            'question_summary': 'The measurement and buying assumptions in dispute.',
            'connection_context': 'The take distinguishes TV measurement from search-style accountability while keeping addressability and incrementality in the same industry conversation.',
            'related_people': [{
                'name': 'Nikhil Lai',
                'relationship': 'Speaker',
                'description': 'Frames the measurement tension.',
                'evidence_type': 'speaker_identity',
                'evidence': 'Attributed speaker in the processed episode.',
                'segment_ids': [],
            }],
            'related_companies': [],
        }
        self.assertTrue(conversation_mapping_is_reviewable(mapping, 7, 9))
        mapping['related_people'][0]['evidence_type'] = 'direct_transcript'
        mapping['related_people'][0]['segment_ids'] = [6]
        self.assertFalse(conversation_mapping_is_reviewable(mapping, 7, 9))

    def test_caption_evidence_is_numbered_and_bounded(self):
        captions = [
            {'start': index * 10, 'end': index * 10 + 5, 'raw_text': f'caption {index}'}
            for index in range(20)
        ]
        evidence = build_caption_evidence(
            captions,
            start_time=50,
            end_time=70,
            padding_seconds=10,
            max_events=10,
        )
        self.assertEqual(evidence['start_segment'], 4)
        self.assertEqual(evidence['end_segment'], 8)
        self.assertIn('[4] caption 4', evidence['excerpt'])
        self.assertIn('[8] caption 8', evidence['excerpt'])

    def test_transcript_alignment_requires_a_unique_high_confidence_match(self):
        segments = [
            {'start': 0, 'end': 4, 'raw_text': 'A generic opening about the market.'},
            {'start': 10, 'end': 14, 'raw_text': 'Television measurement needs a different feedback loop'},
            {'start': 14, 'end': 18, 'raw_text': 'than search attribution because exposure and response are separated.'},
            {'start': 30, 'end': 34, 'raw_text': 'A closing thought about teams.'},
        ]
        aligned = align_quote_to_segments(
            'Television measurement needs a different feedback loop than search attribution because exposure and response are separated.',
            segments,
            expected_start=9,
            expected_end=20,
        )
        self.assertIsNotNone(aligned)
        self.assertEqual(aligned['start'], 10)
        self.assertEqual(aligned['end'], 18)
        self.assertGreaterEqual(aligned['confidence'], 0.75)

    def test_historical_mapping_quality_gate_requires_confidence_and_depth(self):
        mapping = {
            'theme_name': 'Performance TV',
            'theme_summary': 'How television is being connected to accountable media outcomes.',
            'question_text': 'What should performance accountability look like on television?',
            'question_summary': 'The measurement, optimization, and market incentives in dispute.',
            'connection_context': (
                'The take separates television accountability from search-style attribution and '
                'places incrementality, addressability, and buyer expectations in the same '
                'operating debate. It gives practitioners a concrete way to discuss which '
                'feedback loops belong in performance television and which would distort it.'
            ),
            'mapping_confidence': 0.84,
            'related_people': [{
                'name': 'Example Speaker',
                'relationship': 'Speaker',
                'description': 'Introduces the measurement distinction.',
                'evidence_type': 'speaker_identity',
                'evidence': 'Named speaker in episode metadata.',
                'segment_ids': [],
            }],
            'related_companies': [],
        }
        self.assertTrue(historical_mapping_is_reviewable(mapping, 4, 8))
        mapping['mapping_confidence'] = 0.6
        self.assertFalse(historical_mapping_is_reviewable(mapping, 4, 8))

    def test_conversation_taxonomy_preserves_reviewed_vocabulary(self):
        rows = {
            'conversation_themes': [{'id': 'theme-1', 'name': 'Performance TV', 'summary': 'TV and performance media.'}],
            'conversation_questions': [{'theme_id': 'theme-1', 'question_text': 'How should TV be measured?', 'summary': 'Measurement expectations.'}],
            'conversation_entities': [{'entity_type': 'company', 'name': 'Forrester', 'description': 'Research organization.'}],
        }

        class Query:
            def __init__(self, data):
                self.data = data

            def select(self, *_args): return self
            def eq(self, *_args): return self
            def order(self, *_args, **_kwargs): return self
            def limit(self, *_args): return self
            def execute(self): return SimpleNamespace(data=self.data)

        class Database:
            def table(self, name): return Query(rows[name])

        taxonomy = json.loads(fetch_conversation_taxonomy(Database()))
        self.assertEqual(taxonomy['themes'][0]['name'], 'Performance TV')
        self.assertEqual(taxonomy['questions'][0]['theme'], 'Performance TV')
        self.assertEqual(taxonomy['entities'][0]['name'], 'Forrester')

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
