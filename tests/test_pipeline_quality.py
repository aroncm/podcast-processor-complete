import json
import unittest
from types import SimpleNamespace

from modal_app.full_processor import (
    apply_transcript_corrections,
    build_extraction_chunks,
    build_caption_evidence,
    align_quote_to_segments,
    bind_candidate_to_directories,
    call_openai_structured,
    calculate_bakeoff_metrics,
    candidate_has_publishable_length,
    conversation_mapping_is_reviewable,
    context_evidence_is_source_bounded,
    deduplicate_candidates,
    editorial_gate_invalidations,
    fetch_conversation_taxonomy,
    historical_mapping_is_reviewable,
    missing_take_verification_fields,
    normalize_text,
    quote_word_count,
    staged_analysis_should_skip_source_retry,
    staged_analysis_write_plan,
    theme_match_is_controlled,
)


class PipelineQualityTests(unittest.TestCase):
    def test_staged_analysis_batches_skip_known_source_failures_but_allow_retry(self):
        record = {
            "analysis_review_flags": {"ai_draft_status": "source_unavailable"},
        }
        self.assertTrue(staged_analysis_should_skip_source_retry(record))
        self.assertFalse(
            staged_analysis_should_skip_source_retry(record, explicitly_targeted=True)
        )
        self.assertFalse(
            staged_analysis_should_skip_source_retry(
                record,
                mode="regenerate_unreviewed",
            )
        )

    def test_staged_analysis_backfill_never_overwrites_human_or_approved_work(self):
        manual = {
            "editorial_context": "An SME already drafted this context.",
            "context_review_status": "unreviewed",
            "proposed_theme_name": "Performance TV",
            "mapping_review_status": "unreviewed",
        }
        fill_missing = staged_analysis_write_plan(manual, mode="fill_missing")
        self.assertFalse(fill_missing["context"])
        self.assertFalse(fill_missing["mapping"])

        regenerate = staged_analysis_write_plan(manual, mode="regenerate_unreviewed")
        self.assertTrue(regenerate["context"])
        self.assertTrue(regenerate["mapping"])

        locked = {
            **manual,
            "context_review_status": "approved",
            "mapping_review_status": "approved",
        }
        locked_plan = staged_analysis_write_plan(locked, mode="regenerate_unreviewed")
        self.assertFalse(locked_plan["context"])
        self.assertFalse(locked_plan["mapping"])

    def test_take_approval_requires_complete_human_verified_identity(self):
        record = {
            "quote_text": "A complete source-grounded take.",
            "speaker_name": "Operator",
            "guest_id": "operator-1",
            "speaker_title": "",
            "speaker_company": None,
            "category": "Measurement",
            "category_id": "measurement",
        }
        self.assertEqual(
            missing_take_verification_fields(record),
            ["speaker title", "speaker company"],
        )

    def test_directory_binding_uses_only_exact_or_episode_scoped_identity(self):
        directory = {
            "category_by_name": {"measurement": {"id": "cat-1", "name": "Measurement"}},
            "person_by_name": {
                "jane operator": {
                    "id": "guest-1", "name": "Jane Operator", "title": "CEO",
                    "company": "Signal Co", "linkedin_url": "https://example.com/jane",
                },
            },
            "person_by_id": {
                "guest-1": {
                    "id": "guest-1", "name": "Jane Operator", "title": "CEO",
                    "company": "Signal Co", "linkedin_url": "https://example.com/jane",
                },
            },
        }
        bound = bind_candidate_to_directories(
            {"speaker": "Jane Operator", "category": "Measurement"},
            directory,
        )
        self.assertEqual(bound["guest_id"], "guest-1")
        self.assertEqual(bound["category_id"], "cat-1")
        self.assertEqual(bound["speaker_company"], "Signal Co")

        diarized = bind_candidate_to_directories(
            {
                "guest_id": "guest-1", "speaker": "Unknown Speaker",
                "category": "Measurement",
                "directory_resolution": {"speaker_source": "diarized_explicit_identity_evidence"},
            },
            directory,
        )
        self.assertEqual(diarized["speaker"], "Jane Operator")
        self.assertEqual(
            diarized["directory_resolution"]["speaker_source"],
            "diarized_explicit_identity_evidence",
        )

        unresolved = bind_candidate_to_directories(
            {"speaker": "Someone Else", "category": "Broad Business"},
            directory,
        )
        self.assertIsNone(unresolved["guest_id"])
        self.assertIsNone(unresolved["category_id"])

    def test_audited_edits_reopen_only_affected_editorial_gates(self):
        before = {
            "approval_status": "approved",
            "quote_text": "Original take",
            "editorial_context": "Original context",
            "proposed_theme_name": "Performance TV",
            "context_review_status": "approved",
            "mapping_review_status": "approved",
        }
        mapping_only = editorial_gate_invalidations(
            before,
            {"proposed_theme_name": "A different theme"},
        )
        self.assertNotIn("approval_status", mapping_only)
        self.assertNotIn("context_review_status", mapping_only)
        self.assertEqual(mapping_only["mapping_review_status"], "unreviewed")

        take_edit = editorial_gate_invalidations(before, {"quote_text": "Edited take"})
        self.assertEqual(take_edit["approval_status"], "pending")
        self.assertEqual(take_edit["context_review_status"], "unreviewed")
        self.assertEqual(take_edit["mapping_review_status"], "unreviewed")

    def test_quote_length_gate_restores_readable_legacy_range(self):
        self.assertFalse(candidate_has_publishable_length("too short"))
        self.assertTrue(candidate_has_publishable_length(" ".join(["signal"] * 20)))
        self.assertTrue(candidate_has_publishable_length(" ".join(["signal"] * 80)))
        self.assertFalse(candidate_has_publishable_length(" ".join(["signal"] * 81)))
        self.assertEqual(quote_word_count("one two three"), 3)

    def test_transcript_corrections_preserve_raw_and_require_confidence(self):
        segments = [{"id": 0, "text": "Apple oven changed mobile measurement."}]
        corrected, applied, rejected = apply_transcript_corrections(segments, [
            {
                "segment_id": 0,
                "original_phrase": "Apple oven",
                "corrected_phrase": "AppLovin",
                "correction_type": "company",
                "confidence": 0.98,
                "rationale": "The named AdTech company fits the exact discussion.",
            },
            {
                "segment_id": 0,
                "original_phrase": "mobile measurement",
                "corrected_phrase": "mobile attribution",
                "correction_type": "industry_term",
                "confidence": 0.7,
                "rationale": "Uncertain semantic rewrite.",
            },
        ])
        self.assertEqual(corrected[0]["raw_text"], segments[0]["text"])
        self.assertIn("AppLovin", corrected[0]["text"])
        self.assertEqual(len(applied), 1)
        self.assertEqual(rejected[0]["reason"], "below_confidence_gate")

    def test_theme_action_must_use_controlled_registry_exactly(self):
        taxonomy = json.dumps({
            "active_theme_registry": [{"canonical_name": "Performance TV"}],
            "themes": [],
        })
        self.assertTrue(theme_match_is_controlled("existing_theme", "Performance TV", taxonomy))
        self.assertFalse(theme_match_is_controlled("existing_theme", "Performance CTV", taxonomy))
        self.assertTrue(theme_match_is_controlled("propose_new", "Agentic Advertising", taxonomy))
        self.assertTrue(theme_match_is_controlled("abstain", "", taxonomy))

    def test_bakeoff_metrics_use_latest_append_only_review(self):
        items = [{
            "id": "item-1",
            "strategy_key": "hybrid_v3",
            "quote_word_count": 42,
        }]
        reviews = [
            {
                "id": "older",
                "bakeoff_item_id": "item-1",
                "decision": "reject",
                "quality_rating": 2,
                "failure_codes": ["generic"],
                "preferred_in_episode": False,
                "created_at": "2026-08-24T10:00:00+00:00",
            },
            {
                "id": "newer",
                "bakeoff_item_id": "item-1",
                "decision": "approve",
                "quality_rating": 5,
                "failure_codes": [],
                "preferred_in_episode": True,
                "created_at": "2026-08-24T11:00:00+00:00",
            },
        ]
        metrics = calculate_bakeoff_metrics(items, reviews)
        self.assertEqual(metrics["review_coverage"], 1)
        self.assertEqual(metrics["strategies"]["hybrid_v3"]["approval_rate"], 1)
        self.assertEqual(metrics["strategies"]["hybrid_v3"]["average_rating"], 5)

    def test_normalize_text_handles_smart_punctuation(self):
        self.assertEqual(normalize_text('“Supply—path” isn\'t neutral.'), 'supply path isnt neutral')

    def test_extraction_chunks_preserve_global_ids_and_overlap(self):
        segments = [
            {
                'id': index, 'text': f'segment {index}', 'start': index * 2,
                'end': index * 2 + 1, 'speaker_label': 'chunk-0:A',
            }
            for index in range(10)
        ]
        chunks = build_extraction_chunks(segments, max_chars=120, overlap_segments=2)
        self.assertGreater(len(chunks), 1)
        self.assertIn('[0] [speaker=chunk-0:A] segment 0', chunks[0])
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
        self.assertFalse(context_evidence_is_source_bounded([], 7, 9))

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
