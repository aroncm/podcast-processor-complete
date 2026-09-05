import json
import unittest
from types import SimpleNamespace

from modal_app.full_processor import (
    _caption_cache,
    _parse_timedtext_captions,
    apply_transcript_corrections,
    align_timestamps_to_youtube_captions_detailed,
    align_quote_to_segments_semantically,
    build_extraction_chunks,
    build_caption_evidence,
    align_quote_to_segments,
    bind_candidate_to_directories,
    call_openai_structured,
    calculate_bakeoff_metrics,
    candidate_has_publishable_length,
    connection_context_is_substantive,
    conversation_mapping_is_reviewable,
    context_evidence_is_source_bounded,
    deduplicate_candidates,
    directory_selection_changed,
    editorial_gate_invalidations,
    estimate_openai_text_cost,
    fetch_conversation_taxonomy,
    first_numeric_value,
    historical_mapping_is_reviewable,
    legacy_integer_timestamp,
    missing_take_verification_fields,
    merge_reviewed_question_taxonomy,
    merge_tentative_conversation_candidates,
    merge_verified_speaker_connections,
    normalize_text,
    openai_error_is_account_blocking,
    openai_error_is_retryable,
    prepare_category_directory_record,
    prepare_theme_registry_record,
    quote_word_count,
    rank_source_alignment_candidates,
    record_openai_response_usage,
    start_openai_usage_tracking,
    staged_analysis_should_skip_source_retry,
    staged_analysis_write_plan,
    summarize_openai_usage,
    theme_match_is_controlled,
)


class PipelineQualityTests(unittest.TestCase):
    def test_tentative_historical_candidates_are_labeled_and_deduplicated(self):
        taxonomy = json.dumps({"active_theme_registry": []})
        candidate = {
            "proposed_theme_name": "Data Control and Vertical Market Power",
            "proposed_theme_summary": "Vertical data and workflow advantages.",
            "proposed_question_text": "When does vertical specialization create a durable moat?",
            "proposed_question_summary": "Scale versus specialist depth.",
        }
        first = merge_tentative_conversation_candidates(taxonomy, [candidate])
        second = merge_tentative_conversation_candidates(first, [{
            "theme_name": candidate["proposed_theme_name"],
            "theme_summary": candidate["proposed_theme_summary"],
            "question_text": candidate["proposed_question_text"],
            "question_summary": candidate["proposed_question_summary"],
        }])
        parsed = json.loads(second)
        self.assertEqual(len(parsed["tentative_historical_candidates"]), 1)
        self.assertEqual(
            parsed["tentative_historical_candidates"][0]["review_state"],
            "unreviewed_historical_candidate",
        )

    def test_openai_cost_estimate_prices_cached_and_reasoning_tokens(self):
        # Output usage already includes reasoning tokens, so it is billed once
        # at the output rate rather than added a second time.
        cost = estimate_openai_text_cost(
            "gpt-5.6-sol-2026-08-21",
            input_tokens=1_000_000,
            cached_input_tokens=200_000,
            cache_write_tokens=100_000,
            output_tokens=100_000,
        )
        self.assertEqual(cost, 9.76)

    def test_openai_usage_summary_retains_per_call_audit_details(self):
        start_openai_usage_tracking()
        response = SimpleNamespace(
            model="gpt-5.6-terra",
            _request_id="req-cost-audit",
            usage=SimpleNamespace(
                input_tokens=1_000,
                input_tokens_details=SimpleNamespace(
                    cached_tokens=100,
                    cache_write_tokens=0,
                ),
                output_tokens=200,
                output_tokens_details=SimpleNamespace(reasoning_tokens=80),
                total_tokens=1_200,
            ),
        )
        record_openai_response_usage(response, "podthreads_quote_ranking")
        summary = summarize_openai_usage()
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["call_count"], 1)
        self.assertEqual(summary["reasoning_tokens"], 80)
        self.assertEqual(summary["calls"][0]["request_id"], "req-cost-audit")
        self.assertAlmostEqual(summary["estimated_cost_usd"], 0.00422)

    def test_unknown_model_makes_episode_cost_incomplete(self):
        start_openai_usage_tracking()
        response = SimpleNamespace(
            model="future-unpriced-model",
            usage=SimpleNamespace(
                input_tokens=100,
                input_tokens_details={},
                output_tokens=20,
                output_tokens_details={},
                total_tokens=120,
            ),
        )
        record_openai_response_usage(response, "future_operation")
        summary = summarize_openai_usage()
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["unpriced_call_count"], 1)

    def test_category_creation_reuses_normalized_existing_record(self):
        existing = {"id": "measurement", "name": "Measurement", "description": None}
        resolved, should_create = prepare_category_directory_record(
            [existing],
            "  measurement  ",
        )
        self.assertEqual(resolved, existing)
        self.assertFalse(should_create)

    def test_category_creation_builds_stable_collision_safe_record(self):
        resolved, should_create = prepare_category_directory_record(
            [{"id": "market-structure", "name": "Different label"}],
            "Market Structure",
        )
        self.assertTrue(should_create)
        self.assertEqual(resolved["name"], "Market Structure")
        self.assertRegex(resolved["id"], r"^market-structure-[a-f0-9]{8}$")

    def test_category_creation_rejects_placeholder_labels(self):
        with self.assertRaisesRegex(ValueError, "specific industry category"):
            prepare_category_directory_record([], "Other")

    def test_inline_theme_creation_requires_scope_boundaries(self):
        with self.assertRaisesRegex(ValueError, "inclusion and exclusion"):
            prepare_theme_registry_record(
                [],
                "Agentic Micropayments",
                "How autonomous software changes the operating controls for small data payments.",
                [],
                [],
                activate=True,
            )

    def test_inline_theme_creation_builds_active_controlled_record(self):
        record = prepare_theme_registry_record(
            [],
            "  Agentic   Micropayments ",
            "How autonomous software changes the operating controls for small data payments.",
            ["Machine-initiated payments for data or content"],
            ["Consumer checkout with no autonomous agent"],
            activate=True,
        )
        self.assertEqual(record["canonical_name"], "Agentic Micropayments")
        self.assertEqual(record["status"], "active")

    def test_inline_theme_creation_rejects_normalized_duplicate(self):
        with self.assertRaisesRegex(ValueError, "select the existing Theme"):
            prepare_theme_registry_record(
                [{"canonical_name": "Performance TV"}],
                " performance tv ",
                "How television buying becomes accountable to measured business outcomes.",
                ["Television buying tied to outcomes"],
                ["Generic streaming content strategy"],
                activate=True,
            )

    def test_review_payload_normalizes_legacy_integer_timestamps(self):
        self.assertEqual(legacy_integer_timestamp(2154.0), 2154)
        self.assertEqual(legacy_integer_timestamp("2120.0"), 2120)
        self.assertEqual(legacy_integer_timestamp(0.0), 0)
        self.assertIsNone(legacy_integer_timestamp(None))

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

    def test_staged_analysis_can_regenerate_only_the_rejected_context_layer(self):
        record = {
            "editorial_context": "A rejected context draft.",
            "context_review_status": "rejected",
            "proposed_theme_name": "Performance TV",
            "proposed_question_text": "How should television outcomes be measured?",
            "mapping_review_status": "unreviewed",
        }
        context_only = staged_analysis_write_plan(
            record,
            mode="regenerate_unreviewed",
            layers=["context"],
        )
        self.assertTrue(context_only["context"])
        self.assertFalse(context_only["mapping"])

    def test_staged_analysis_rejects_unknown_layer_names(self):
        with self.assertRaisesRegex(ValueError, "Unsupported staged analysis layer"):
            staged_analysis_write_plan({}, layers=["context", "take"])

    def test_concise_specific_connection_context_is_reviewable(self):
        self.assertTrue(connection_context_is_substantive(
            "The Instagram analogy places ChatGPT within the familiar progression from curated launch inventory to automated marketplace mechanics."
        ))
        self.assertFalse(connection_context_is_substantive("This Take connects to the broader industry conversation."))

    def test_take_approval_requires_complete_human_verified_identity(self):
        record = {
            "quote_text": "A complete source-grounded take.",
            "speaker_name": "Operator",
            "guest_id": "operator-1",
            "speaker_title": "",
            "speaker_company": None,
        }
        self.assertEqual(
            missing_take_verification_fields(record),
            ["speaker title", "speaker company"],
        )

    def test_take_approval_requires_verified_youtube_clock(self):
        record = {
            "quote_text": "A complete source-grounded take.",
            "speaker_name": "Jane Operator",
            "guest_id": "operator-1",
            "speaker_title": "CEO",
            "speaker_company": "Signal Co",
            "youtube_id": "video-1",
            "youtube_alignment_status": "failed",
        }
        self.assertEqual(
            missing_take_verification_fields(record),
            ["exact YouTube segment"],
        )
        record["youtube_alignment_status"] = "verified"
        self.assertEqual(missing_take_verification_fields(record), [])

    def test_take_category_is_not_an_approval_requirement(self):
        record = {
            "quote_text": "A complete source-grounded take.",
            "speaker_name": "Jane Operator",
            "guest_id": "operator-1",
            "speaker_title": "CEO",
            "speaker_company": "Signal Co",
        }
        self.assertEqual(missing_take_verification_fields(record), [])

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

        self.assertEqual(
            editorial_gate_invalidations(before, {"category_id": "legacy-category"}),
            {},
        )

    def test_unchanged_directory_selections_do_not_create_material_edits(self):
        before = {"guest_id": "guest-1", "category_id": "measurement"}
        self.assertFalse(directory_selection_changed(before, "guest_id", "guest-1"))
        self.assertFalse(directory_selection_changed(before, "category_id", "measurement"))
        self.assertTrue(directory_selection_changed(before, "guest_id", "guest-2"))

    def test_verified_speaker_and_company_seed_editable_mapping_suggestions(self):
        merged = merge_verified_speaker_connections(
            {"related_people": [], "related_companies": []},
            {
                "speaker": "Ari Paparo",
                "speaker_title": "Co-founder and Contributor",
                "speaker_company": "Marketecture Media",
                "guest_id": "ari-paparo",
            },
        )
        self.assertEqual(merged["related_people"][0]["name"], "Ari Paparo")
        self.assertEqual(merged["related_people"][0]["guest_id"], "ari-paparo")
        self.assertEqual(merged["related_people"][0]["evidence_type"], "speaker_identity")
        self.assertEqual(merged["related_companies"][0]["name"], "Marketecture Media")
        self.assertNotIn("description", merged["related_people"][0])
        self.assertNotIn("description", merged["related_companies"][0])

    def test_verified_connection_seeding_deduplicates_model_entities(self):
        merged = merge_verified_speaker_connections(
            {
                "related_people": [{
                    "name": "Ari Paparo",
                    "relationship": "Speaker",
                    "description": "",
                    "evidence_type": "speaker_identity",
                    "evidence": "Episode identity.",
                    "segment_ids": [],
                }],
                "related_companies": [],
            },
            {"speaker": "Ari Paparo", "guest_id": "ari-paparo"},
        )
        self.assertEqual(len(merged["related_people"]), 1)
        self.assertEqual(merged["related_people"][0]["directory_id"], "ari-paparo")
        self.assertNotIn("description", merged["related_people"][0])

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

    def test_approved_staged_question_is_reused_under_its_parent_theme(self):
        merged = merge_reviewed_question_taxonomy(
            [{"id": "theme-1", "name": "Performance TV"}],
            [{
                "theme_id": "theme-1",
                "question_text": "Who should control the CTV transaction layer?",
                "summary": "Graph version",
            }],
            [{
                "mapping_review_status": "approved",
                "proposed_theme_name": "Performance TV",
                "proposed_question_text": "Who should control the CTV transaction layer?",
                "proposed_question_summary": "Duplicate staged version",
            }, {
                "mapping_review_status": "approved",
                "proposed_theme_name": "Performance TV",
                "proposed_question_text": "How should television incrementality be measured?",
                "proposed_question_summary": "Approved but not yet published",
            }, {
                "mapping_review_status": "unreviewed",
                "proposed_theme_name": "Performance TV",
                "proposed_question_text": "Should an unreviewed Question leak into suggestions?",
            }],
        )
        self.assertEqual(len(merged), 2)
        self.assertEqual({item["theme"] for item in merged}, {"Performance TV"})
        self.assertEqual(merged[1]["review_state"], "approved_staged_mapping")

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

    def test_extraction_chunks_with_zero_overlap_do_not_retain_prior_chunk(self):
        segments = [
            {
                'id': index,
                'text': f'segment {index} ' + ('x' * 30),
                'start': index * 2,
                'end': index * 2 + 1,
            }
            for index in range(12)
        ]
        chunks = build_extraction_chunks(
            segments,
            max_chars=120,
            overlap_segments=0,
        )
        rendered_ids = [
            int(line.split(']', 1)[0][1:])
            for chunk in chunks
            for line in chunk.splitlines()
        ]
        self.assertEqual(rendered_ids, list(range(12)))
        self.assertLess(len(chunks), len(segments))

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

    def test_semantic_alignment_stages_a_bounded_candidate_without_verifying_it(self):
        segments = [
            {'start': 0, 'end': 4, 'raw_text': 'A generic introduction to the episode.'},
            {'start': 20, 'end': 25, 'raw_text': 'The real risk is measurement becoming a procurement checklist'},
            {'start': 25, 'end': 31, 'raw_text': 'instead of a tool for understanding whether advertising changed behavior.'},
            {'start': 80, 'end': 85, 'raw_text': 'A separate discussion about creative teams.'},
        ]

        class Responses:
            def create(self, **_kwargs):
                return SimpleNamespace(
                    status='completed',
                    output_text=json.dumps({
                        'supported': True,
                        'candidate_id': 0,
                        'match_type': 'faithful_paraphrase',
                        'confidence': 0.91,
                        'supporting_segment_ids': [1, 2],
                        'reason': 'The source expresses the same measurement and behavior distinction.',
                    }),
                )

        candidates = rank_source_alignment_candidates(
            'The risk is turning measurement into procurement instead of learning whether advertising changes behavior.',
            segments,
            expected_start=18,
            expected_end=34,
        )
        self.assertGreaterEqual(len(candidates), 1)
        aligned = align_quote_to_segments_semantically(
            'The risk is turning measurement into procurement instead of learning whether advertising changes behavior.',
            segments,
            expected_start=18,
            expected_end=34,
            client=SimpleNamespace(responses=Responses()),
        )
        self.assertIsNotNone(aligned)
        self.assertTrue(aligned['verification_required'])
        self.assertEqual(aligned['start'], 20)
        self.assertEqual(aligned['end'], 31)

    def test_first_numeric_value_preserves_zero_and_skips_invalid_values(self):
        self.assertEqual(first_numeric_value(None, '', 'bad', 0, 12), 0)

    def test_youtube_alignment_uses_full_track_when_rss_edit_drift_is_large(self):
        youtube_id = "fixture-large-edit-drift"
        _caption_cache[youtube_id] = [
            {
                "start": 100.0,
                "end": 104.0,
                "raw_text": "Retail media needs a durable measurement layer",
                "norm_text": "retail media needs a durable measurement layer",
                "caption_source": "fixture",
            },
            {
                "start": 104.0,
                "end": 109.0,
                "raw_text": "before buyers can compare outcomes across closed ecosystems.",
                "norm_text": "before buyers can compare outcomes across closed ecosystems",
                "caption_source": "fixture",
            },
            {
                "start": 800.0,
                "end": 804.0,
                "raw_text": "A generic discussion about buyers and outcomes.",
                "norm_text": "a generic discussion about buyers and outcomes",
                "caption_source": "fixture",
            },
        ]
        try:
            aligned = align_timestamps_to_youtube_captions_detailed(
                "Retail media needs a durable measurement layer before buyers can compare outcomes across closed ecosystems.",
                youtube_id,
                whisper_start=900,
                whisper_end=910,
            )
        finally:
            _caption_cache.pop(youtube_id, None)
        self.assertEqual(aligned["status"], "verified")
        self.assertEqual(aligned["details"]["search_scope"], "full_caption_track")
        self.assertAlmostEqual(aligned["start"], 98.5)
        self.assertAlmostEqual(aligned["end"], 110.5)

    def test_long_asr_quote_matches_as_word_sequence_without_autojunk(self):
        quote = (
            "So blended ROAS metric, this is where brands lean often on their agencies "
            "or some of the late stage DTC brands are doing this with heads of growth. "
            "You would have a baseline of what the ROAS was before you plus up a certain "
            "channel. The idea is that the efficiency of retargeting on social is going "
            "to be benefited by incremental reach on CTV, which is then going to have a "
            "downstream impact on the cost per clicks of search. And so the blended ROAS "
            "four to one, five to one, seven to one looks like ROAS, but it is trying to "
            "harmonize and normalize performance across all the disparate channels."
        )
        segments = [
            {"start": 607.4, "end": 617.04, "raw_text": "The blended ROAS metric actually this is where brands lean often on their agencies"},
            {"start": 617.04, "end": 627.72, "raw_text": "or some late stage D2C brands are doing this with heads of growth but you would have a baseline"},
            {"start": 627.72, "end": 640.48, "raw_text": "of what the ROAS was before you plus up a certain channel and the efficiency of retargeting on social is benefited by incremental reach on CTV"},
            {"start": 640.48, "end": 653.08, "raw_text": "which has a downstream impact on cost per clicks of search and so blended ROAS four to one five to one seven to one looks like ROAS"},
            {"start": 653.08, "end": 656.52, "raw_text": "but tries to harmonize and normalize performance across all those disparate channels"},
            {"start": 800.0, "end": 810.0, "raw_text": "A separate generic discussion about campaign performance and metrics"},
        ]
        aligned = align_quote_to_segments(
            quote,
            segments,
            expected_start=665,
            expected_end=711,
            global_fallback=True,
            max_window_events=32,
        )
        self.assertIsNotNone(aligned)
        self.assertEqual(aligned["start"], 607.4)
        self.assertEqual(aligned["end"], 656.52)
        self.assertGreaterEqual(aligned["confidence"], 0.70)

    def test_timedtext_xml_parser_preserves_caption_clock(self):
        captions = _parse_timedtext_captions(
            b'<transcript><text start="35.25" dur="2.5">Exact &amp; sourced</text></transcript>',
            "fixture_xml",
        )
        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0]["raw_text"], "Exact & sourced")
        self.assertEqual(captions[0]["start"], 35.25)
        self.assertEqual(captions[0]["end"], 37.75)
        self.assertEqual(captions[0]["caption_source"], "fixture_xml")

    def test_ttml_parser_preserves_hour_minute_second_clock(self):
        captions = _parse_timedtext_captions(
            b'''<?xml version="1.0"?><tt xmlns="http://www.w3.org/ns/ttml"><body><div>
            <p begin="00:35:23.520" end="00:35:27.880">A precisely timed source moment</p>
            </div></body></tt>''',
            "fixture_ttml",
        )
        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0]["raw_text"], "A precisely timed source moment")
        self.assertAlmostEqual(captions[0]["start"], 2123.52)
        self.assertAlmostEqual(captions[0]["end"], 2127.88)
        self.assertEqual(captions[0]["caption_source"], "fixture_ttml")

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

    def test_openai_quota_errors_fail_without_retrying(self):
        quota = RuntimeError("429 insufficient_quota credit_balance_exhausted")
        transient = RuntimeError("429 rate_limit temporarily unavailable")
        self.assertTrue(openai_error_is_account_blocking(quota))
        self.assertFalse(openai_error_is_account_blocking(transient))
        self.assertFalse(openai_error_is_retryable(quota))
        self.assertTrue(openai_error_is_retryable(transient))


if __name__ == '__main__':
    unittest.main()
