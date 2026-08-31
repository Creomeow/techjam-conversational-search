from __future__ import annotations

from pathlib import Path
import json

from shopping.conversation import QUESTION_TEXT, SessionState
from shopping.ranker import FIELD_SIGNAL_PROFILES, FieldSignalRanker, LinearRanker, PlaceholderSignalRanker
from shopping.retrieval import CatalogIndex
from shopping.retrieval import RetrievalHit
from shopping.semantic_reranker import SemanticReranker
from shopping.neural_reranker import NeuralReranker


PLACEHOLDER_HEURISTICS = {
    "version": "targeted_backfill_v24_guarded_fsm_neural_shadow",
    "raw_fallback_turn": 6,
    "novel_history_turn": 10,
    "history_depth": 60,
    "late_route_count": 3,
    "late_candidate_k": 120,
    "history_rank_offset": 20.0,
    "history_frequency_weight": 0.5,
    "exploration_confidence_margin": 4.0,
    "backfill_candidate_k": 120,
    "backfill_min_turn": 8,
    "backfill_confidence": 0.12,
}


class Agent:
    """Offline agent backed by normalized, multi-field lexical retrieval."""

    def __init__(
        self,
        catalog_path: str | Path = "data/catalog.jsonl",
        *,
        index: CatalogIndex | None = None,
        ranker: LinearRanker | None = None,
        ranker_path: str | Path | None = None,
        semantic_path: str | Path | None = None,
        candidate_k: int | None = None,
        enable_hybrid: bool = False,
        hybrid_path: str | Path | None = None,
        enable_late_exploration: bool = True,
        enable_placeholder_signals: bool = True,
        field_signal_profile: str = "targeted_exact",
        candidate_backfill_mode: str = "targeted",
        override_cooldown_mode: str = "narrow",
        buyer_state_mode: str = "active_repair",
        personalization_mode: str = "off",
        diversity_mode: str = "brand_cap2",
        constraint_query_mode: str = "marker_safe",
        structured_constraint_mode: str = "off",
        override_history_mode: str = "preserve",
        neural_shadow_path: str | Path | None = None,
        allow_unpromoted_neural: bool = False,
        semantic_blend_weight: float | None = 1.5,
    ) -> None:
        self.catalog_path = Path(catalog_path)
        self.index = index or CatalogIndex(
            self.catalog_path,
            enable_hybrid=enable_hybrid,
            hybrid_path=hybrid_path,
        )
        self.candidate_k = max(10, int(candidate_k)) if candidate_k is not None else None
        self.ranker_load_error: str | None = None
        self.semantic_load_error: str | None = None
        self.enable_late_exploration = bool(enable_late_exploration)
        self.field_signal_profile = field_signal_profile
        if candidate_backfill_mode not in {"off", "targeted", "fsm_confidence", "targeted_evidence"}:
            raise ValueError(f"unknown candidate backfill mode: {candidate_backfill_mode}")
        self.candidate_backfill_mode = candidate_backfill_mode
        if override_cooldown_mode not in {"off", "narrow"}:
            raise ValueError(f"unknown override cooldown mode: {override_cooldown_mode}")
        self.override_cooldown_mode = override_cooldown_mode
        if buyer_state_mode not in {"off", "shadow", "active", "active_repair", "active_v2"}:
            raise ValueError(f"unknown buyer state mode: {buyer_state_mode}")
        self.buyer_state_mode = buyer_state_mode
        if personalization_mode not in {"off", "soft_tags"}:
            raise ValueError(f"unknown personalization mode: {personalization_mode}")
        self.personalization_mode = personalization_mode
        if diversity_mode not in {"off", "brand_cap2"}:
            raise ValueError(f"unknown diversity mode: {diversity_mode}")
        self.diversity_mode = diversity_mode
        if constraint_query_mode not in {
            "off", "labels", "marker", "marker_safe", "marker_after_override",
            "marker_safe_plus",
        }:
            raise ValueError(f"unknown constraint query mode: {constraint_query_mode}")
        self.constraint_query_mode = constraint_query_mode
        if structured_constraint_mode not in {"off", "slots", "slots_buying"}:
            raise ValueError(f"unknown structured constraint mode: {structured_constraint_mode}")
        self.structured_constraint_mode = structured_constraint_mode
        if override_history_mode not in {"preserve", "isolated"}:
            raise ValueError(f"unknown override history mode: {override_history_mode}")
        self.override_history_mode = override_history_mode
        if ranker is not None:
            self.ranker = ranker
        else:
            model_directory = Path(__file__).resolve().parents[1] / "models"
            if ranker_path is not None:
                model_path = Path(ranker_path)
            else:
                model_path = next(
                    (path for path in (model_directory / "ranker.json", model_directory / "rankerv01.json") if path.exists()),
                    model_directory / "ranker.json",
                )
            try:
                self.ranker = LinearRanker.load(model_path)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                # A malformed optional artifact must not invalidate the entire
                # submission; retrieval order remains a safe deterministic fallback.
                self.ranker = None
                self.ranker_load_error = f"{type(exc).__name__}: {exc}"
        if enable_placeholder_signals and isinstance(self.ranker, LinearRanker) and self.ranker.legacy_schema:
            if field_signal_profile not in FIELD_SIGNAL_PROFILES:
                raise ValueError(f"unknown field signal profile: {field_signal_profile}")
            self.ranker = PlaceholderSignalRanker(
                self.ranker,
                field_weights=FIELD_SIGNAL_PROFILES[field_signal_profile],
                version=f"placeholder_signals_v2+fields_{field_signal_profile}",
            )
        elif field_signal_profile not in FIELD_SIGNAL_PROFILES:
            raise ValueError(f"unknown field signal profile: {field_signal_profile}")
        elif field_signal_profile != "off" and isinstance(self.ranker, LinearRanker):
            self.ranker = FieldSignalRanker(
                self.ranker,
                field_weights=FIELD_SIGNAL_PROFILES[field_signal_profile],
                version=f"full_schema+fields_{field_signal_profile}",
            )
        if self.candidate_k is None:
            self.candidate_k = self.ranker.candidate_k if self.ranker else 40
        semantic_directory = Path(__file__).resolve().parents[1] / "models"
        semantic_candidates = (
            [Path(semantic_path)] if semantic_path is not None else
            [semantic_directory / "neural_semantic_reranker", semantic_directory / "semantic_ranker.json", semantic_directory / "semantic_ranker.joblib"]
        )
        self.semantic_reranker = None
        semantic_model_path = semantic_candidates[0]
        for candidate in semantic_candidates:
            if candidate.exists():
                semantic_model_path = candidate
                self.semantic_reranker = (
                    NeuralReranker.load(
                        candidate, allow_unpromoted=allow_unpromoted_neural
                    )
                    if candidate.is_dir()
                    else SemanticReranker.load(candidate)
                )
                if self.semantic_reranker is not None:
                    if semantic_blend_weight is not None and hasattr(self.semantic_reranker, "blend_weight"):
                        self.semantic_reranker.blend_weight = max(0.0, float(semantic_blend_weight))
                    break
        if semantic_model_path.exists() and self.semantic_reranker is None:
            self.semantic_load_error = f"unable to load semantic artifact: {semantic_model_path}"
        self.neural_shadow_reranker = (
            NeuralReranker.load(neural_shadow_path, allow_unpromoted=True)
            if neural_shadow_path is not None
            else None
        )
        self.neural_shadow_load_error = (
            f"unable to load neural shadow artifact: {neural_shadow_path}"
            if neural_shadow_path is not None and self.neural_shadow_reranker is None
            else None
        )
        self._sessions: dict[str, SessionState] = {}
        self._candidate_history: dict[str, dict[str, tuple[int, int, RetrievalHit, int]]] = {}
        self._recommended_history: dict[str, set[str]] = {}
        self._neural_shadow_history: dict[str, list[dict[str, object]]] = {}

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions[session_id] = SessionState(
            session_id,
            dict(user_profile or {}),
            enable_override_cooldown=self.override_cooldown_mode == "narrow",
            preserve_override_context=True,
        )
        self._candidate_history[session_id] = {}
        self._recommended_history[session_id] = set()
        self._neural_shadow_history[session_id] = []

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        state = self._sessions.get(session_id)
        if state is None:
            raise RuntimeError("reset must be called before respond")
        update = state.update(user_message)
        if update.get("override") and (
            not state.preserve_override_context
            or self.override_history_mode == "isolated"
            or (
                self.constraint_query_mode == "marker_safe"
                and state.initial_intent in {"buying", "browsing"}
            )
        ):
            # Reset exploratory history for callers using strict intent-reset
            # semantics; the default Agent preserves compatible context below.
            self._candidate_history[session_id] = {}
            self._recommended_history[session_id] = set()
        # The first turn benefits from all weighted views; later turns usually
        # differ only by one newly revealed slot, so a single precision view
        # keeps multi-turn latency bounded.
        late_exploration_turn = int(PLACEHOLDER_HEURISTICS["raw_fallback_turn"])
        dissatisfied = bool(update.get("dissatisfied"))
        search_query = self._search_query(state, user_message)
        route_count = (
            int(PLACEHOLDER_HEURISTICS["late_route_count"])
            if dissatisfied or turn == 1 or (self.enable_late_exploration and turn >= late_exploration_turn)
            else 1
        )
        configured_candidate_k = self.candidate_k
        if self.candidate_backfill_mode == "off":
            if dissatisfied or (self.enable_late_exploration and turn >= late_exploration_turn):
                configured_candidate_k = max(configured_candidate_k, int(PLACEHOLDER_HEURISTICS["late_candidate_k"]))
        elif dissatisfied:
            configured_candidate_k = max(configured_candidate_k, int(PLACEHOLDER_HEURISTICS["backfill_candidate_k"]))
        elif self.candidate_backfill_mode == "targeted" and update.get("override") and turn >= 4:
            configured_candidate_k = max(configured_candidate_k, int(PLACEHOLDER_HEURISTICS["backfill_candidate_k"]))
        candidate_k = max(top_k, configured_candidate_k) if self.ranker else None
        result = self._search_index(
            state,
            search_query,
            top_k,
            route_count=route_count,
            candidate_k=candidate_k,
            cache_variant=(
                "override" if update.get("override")
                else "dissatisfaction" if dissatisfied
                else "exploration" if turn >= late_exploration_turn
                else "normal"
            ),
        )
        if (
            self.candidate_backfill_mode in {"targeted", "fsm_confidence", "targeted_evidence"}
            and not dissatisfied
            and (
                (
                    self.candidate_backfill_mode == "targeted"
                    and turn >= int(PLACEHOLDER_HEURISTICS["backfill_min_turn"])
                    and result.confidence < float(PLACEHOLDER_HEURISTICS["backfill_confidence"])
                )
                or (
                    self.candidate_backfill_mode == "targeted_evidence"
                    and turn >= 6
                    and state.initial_intent == "buying"
                    and (
                        result.confidence < 0.08
                        or (
                            bool(result.hits)
                            and state.category
                            and result.hits[0].signals[2] < 0.20
                        )
                        or (
                            bool(result.hits)
                            and bool(state.slots)
                            and result.hits[0].signals[3] < 0.10
                        )
                    )
                )
                or (
                    self.candidate_backfill_mode == "fsm_confidence"
                    and turn >= 3
                    and state.buyer_state in {"narrowing", "repairing"}
                    and result.confidence < 0.02
                )
            )
        ):
            result = self._search_index(
                state,
                search_query,
                top_k,
                route_count=max(route_count, int(PLACEHOLDER_HEURISTICS["late_route_count"])),
                candidate_k=max(candidate_k or top_k, int(PLACEHOLDER_HEURISTICS["backfill_candidate_k"])),
                cache_variant=("fsm_backfill" if self.candidate_backfill_mode == "fsm_confidence" else "backfill"),
            )
        state.observe_results(result.candidate_pool_size, result.confidence)
        learned_limit = (
            len(result.hits)
            if dissatisfied
            else min(len(result.hits), max(top_k * 2, top_k))
            if self.diversity_mode == "brand_cap2"
            else top_k
        )
        if self.semantic_reranker is not None and hasattr(self.index, "_document"):
            # A semantic model must see the full retrieved/reranked candidate
            # set; truncating to top_k before semantic scoring would make it
            # unable to rescue a lexically weaker but semantically correct hit.
            learned_hits = self.ranker.rerank(result.hits, len(result.hits)) if self.ranker else result.hits
            if (
                isinstance(self.semantic_reranker, NeuralReranker)
                and not self.semantic_reranker.should_activate(state.buyer_state, learned_hits)
            ):
                learned_hits = learned_hits[:learned_limit]
            else:
                learned_hits = self.semantic_reranker.rerank(
                    search_query, learned_hits, self.index, learned_limit
                )
        else:
            learned_hits = self.ranker.rerank(result.hits, learned_limit) if self.ranker else result.hits
        if self.neural_shadow_reranker is not None and hasattr(self.index, "_document"):
            shadow_hits = self.neural_shadow_reranker.rerank(
                search_query, learned_hits, self.index, learned_limit
            )
            production_ids = [hit.parent_asin for hit in learned_hits[:top_k]]
            shadow_ids = [hit.parent_asin for hit in shadow_hits[:top_k]]
            self._neural_shadow_history[session_id].append({
                "turn": int(turn),
                "buyer_state": state.buyer_state,
                "production_top1": production_ids[0] if production_ids else None,
                "shadow_top1": shadow_ids[0] if shadow_ids else None,
                "top1_changed": bool(production_ids and shadow_ids and production_ids[0] != shadow_ids[0]),
                "top_k_overlap": len(set(production_ids) & set(shadow_ids)),
                "top_k": int(top_k),
            })
        self._record_candidates(session_id, result.hits, turn)
        hits = (
            self._dissatisfaction_hits(session_id, learned_hits, top_k)
            if dissatisfied and not state.exclusions
            else self._exploration_hits(session_id, result.hits, learned_hits, top_k, turn)
        )
        if self.diversity_mode == "brand_cap2":
            hits = self._diversify_hits(hits, learned_hits, top_k)
        self._recommended_history[session_id].update(hit.parent_asin for hit in hits)
        recommendations = [
            {"parent_asin": hit.parent_asin, "score": hit.score}
            for hit in hits
        ]
        if self.buyer_state_mode == "active_v2":
            ask_attribute = state.propose_question(
                result.candidate_pool_size,
                result.confidence,
                turn,
                result.question_scores,
                allow_late_repair=True,
            )
        elif (
            self.buyer_state_mode == "active_repair"
            and state.buyer_state == "repairing"
            and turn >= 8
            and result.confidence < 0.02
        ):
            ask_attribute = state.commit_question(
                state.propose_question(
                    result.candidate_pool_size,
                    result.confidence,
                    turn,
                    result.question_scores,
                    allow_late_repair=True,
                )
            )
        else:
            ask_attribute = state.choose_question(
                result.candidate_pool_size,
                result.confidence,
                turn,
                result.question_scores,
            )
        # Broad browsing queries benefit most from a concrete product feature
        # on the first follow-up.  The simulator's feature reply is usually
        # more discriminative than material/color, while buying and override
        # routes retain their learned question ordering.
        if (
            state.route == "browsing"
            and turn == 1
            and "feature" not in state.asked
            and "feature" not in state.declined
        ):
            if self.buyer_state_mode != "active_v2":
                state.asked.append("feature")
                state.last_asked = "feature"
            ask_attribute = "feature"
        elif (
            state.route == "browsing"
            and turn == 2
            and "other" not in state.asked
            and "other" not in state.declined
        ):
            if self.buyer_state_mode != "active_v2":
                state.asked.append("other")
                state.last_asked = "other"
            ask_attribute = "other"
        elif (
            state.route == "buying"
            and turn == 2
            and self.buyer_state_mode in {"active", "active_repair", "active_v2"}
            and state.buyer_state == "narrowing"
            and "other" not in state.asked
            and "other" not in state.declined
        ):
            if self.buyer_state_mode != "active_v2":
                state.asked.append("other")
                state.last_asked = "other"
            ask_attribute = "other"
        elif (
            state.route == "buying"
            and turn == 2
            and "feature" not in state.asked
            and "feature" not in state.declined
        ):
            if self.buyer_state_mode != "active_v2":
                state.asked.append("feature")
                state.last_asked = "feature"
            ask_attribute = "feature"
        if self.buyer_state_mode == "active_v2":
            ask_attribute = state.commit_question(ask_attribute)
        message = QUESTION_TEXT[ask_attribute] if ask_attribute else "Here are the closest matches I found."
        return {
            "message": message,
            "ask_attribute": ask_attribute,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }

    def get_diagnostics(self, session_id: str) -> dict:
        """Development telemetry read by the tracing wrapper, not the API payload."""
        state = self._sessions.get(session_id)
        if state is None:
            return {}
        # Retrieval diagnostics are reconstructed from the last response by the
        # wrapper through these fields; conversation details are useful for tuning.
        return {
            "route": state.route,
            "buyer_state": state.buyer_state if self.buyer_state_mode in {"shadow", "active", "active_repair", "active_v2"} else None,
            "buyer_state_confidence": (
                state.buyer_state_machine.state_confidence
                if self.buyer_state_mode in {"shadow", "active", "active_repair", "active_v2"} else None
            ),
            "buyer_state_action": (
                state.buyer_state_machine.suggested_action()
                if self.buyer_state_mode in {"shadow", "active", "active_repair", "active_v2"} else None
            ),
            "buyer_state_transitions": (
                list(state.buyer_state_machine.transitions)
                if self.buyer_state_mode in {"shadow", "active", "active_repair", "active_v2"} else []
            ),
            "category": state.category,
            "active_slots": {key: list(values) for key, values in state.slots.items()},
            "excluded_slots": {key: list(values) for key, values in state.exclusions.items()},
            "slot_provenance": {key: list(values) for key, values in state.slot_provenance.items()},
            "declined_attributes": sorted(state.declined),
            "asked_attributes": list(state.asked),
            "override_count": state.override_count,
            "dissatisfaction_count": state.dissatisfaction_count,
            "candidate_pool_size": state.candidate_pool_size,
            "confidence": state.confidence,
            "ranker_version": self.ranker.version if self.ranker else None,
            "field_signal_profile": self.field_signal_profile if self.ranker else "off",
            "candidate_backfill_mode": self.candidate_backfill_mode,
            "override_cooldown_mode": self.override_cooldown_mode,
            "buyer_state_mode": self.buyer_state_mode,
            "personalization_mode": self.personalization_mode,
            "diversity_mode": self.diversity_mode,
            "constraint_query_mode": self.constraint_query_mode,
            "structured_constraint_mode": self.structured_constraint_mode,
            "override_history_mode": self.override_history_mode,
            "preserve_override_context": bool(state.preserve_override_context),
            "retrieval_cache": self.index.cache_stats() if hasattr(self.index, "cache_stats") else None,
            "candidate_k": self.candidate_k,
            "hybrid_enabled": self.index.hybrid is not None,
            "heuristic_policy": PLACEHOLDER_HEURISTICS["version"] if self.enable_late_exploration else None,
            "candidate_history_size": len(self._candidate_history.get(session_id, {})),
            "ranker_load_error": self.ranker_load_error,
            "semantic_reranker_version": self.semantic_reranker.version if self.semantic_reranker and hasattr(self.index, "_document") else None,
            "semantic_load_error": self.semantic_load_error,
            "neural_shadow_version": (
                self.neural_shadow_reranker.version
                if self.neural_shadow_reranker is not None else None
            ),
            "neural_shadow_load_error": self.neural_shadow_load_error,
            "neural_shadow_turns": list(self._neural_shadow_history.get(session_id, [])),
        }

    def _search_query(self, state: SessionState, user_message: str) -> str:
        query = state.query(
            user_message,
            label_constraints=self.constraint_query_mode == "labels",
            marker_constraints=(
                self.constraint_query_mode == "marker"
                or (
                    self.constraint_query_mode == "marker_safe"
                    and state.override_count == 0
                    and state.initial_intent in {"buying", "browsing"}
                )
                or (
                    self.constraint_query_mode == "marker_after_override"
                    and state.override_count > 0
                )
                or (
                    self.constraint_query_mode == "marker_safe_plus"
                    and (
                        state.initial_intent in {"buying", "browsing"}
                        or state.override_count > 0
                    )
                )
            ),
        )
        if self.personalization_mode != "soft_tags":
            return query

        tags = [
            str(value).strip().lower()
            for value in state.user_profile.get("preference_tags", [])
            if str(value).strip()
        ]
        # Keep profile evidence soft: no colon means query_parts does not treat
        # it as an explicit constraint, while FTS/semantic retrieval may still
        # use the extra terms to break otherwise ambiguous ties.
        return f"{query} Profile preferences {' '.join(dict.fromkeys(tags))}." if tags else query

    def _search_index(
        self,
        state: SessionState,
        message: str,
        top_k: int,
        **kwargs: object,
    ) -> object:
        """Search with structured slot metadata when supported by the index."""
        use_structured = self.structured_constraint_mode == "slots" or (
            self.structured_constraint_mode == "slots_buying"
            and state.initial_intent == "buying"
        )
        if isinstance(self.index, CatalogIndex) and use_structured:
            constraints, exclusions = state.structured_constraints()
            return self.index.search(
                message,
                top_k,
                **kwargs,
                constraint_values=constraints,
                excluded_values=exclusions,
            )
        # Lightweight test/dummy indexes implement the older search contract.
        return self.index.search(message, top_k, **kwargs)

    def _diversify_hits(
        self,
        primary: list[RetrievalHit],
        fallback: list[RetrievalHit],
        top_k: int,
    ) -> list[RetrievalHit]:
        candidates: list[RetrievalHit] = []
        seen_ids: set[str] = set()
        for hit in [*primary, *fallback]:
            if hit.parent_asin not in seen_ids:
                candidates.append(hit)
                seen_ids.add(hit.parent_asin)
        selected: list[RetrievalHit] = []
        deferred: list[RetrievalHit] = []
        brand_counts: dict[str, int] = {}
        for hit in candidates:
            document = self.index._document(hit.parent_asin) if hasattr(self.index, "_document") else None
            brand = document[3].strip().lower() if document and len(document) > 3 else ""
            if brand and brand_counts.get(brand, 0) >= 2:
                deferred.append(hit)
                continue
            selected.append(hit)
            if brand:
                brand_counts[brand] = brand_counts.get(brand, 0) + 1
            if len(selected) >= top_k:
                return selected
        selected.extend(deferred[: max(0, top_k - len(selected))])
        return selected[:top_k]

    def _record_candidates(self, session_id: str, hits: list[RetrievalHit], turn: int) -> None:
        history = self._candidate_history[session_id]
        depth = max(
            int(PLACEHOLDER_HEURISTICS["history_depth"]),
            int(PLACEHOLDER_HEURISTICS["late_candidate_k"]) if turn >= int(PLACEHOLDER_HEURISTICS["raw_fallback_turn"]) else 0,
        )
        for rank, hit in enumerate(hits[:depth], start=1):
            previous = history.get(hit.parent_asin)
            if previous is None:
                history[hit.parent_asin] = (rank, turn, hit, 1)
            elif rank < previous[0]:
                history[hit.parent_asin] = (rank, turn, hit, previous[3] + 1)
            else:
                history[hit.parent_asin] = (previous[0], turn, previous[2], previous[3] + 1)

    def _dissatisfaction_hits(
        self,
        session_id: str,
        learned_hits: list[RetrievalHit],
        top_k: int,
    ) -> list[RetrievalHit]:
        """Return the next learned page after the user rejects prior pages."""
        recommended = self._recommended_history[session_id]
        unseen = [hit for hit in learned_hits if hit.parent_asin not in recommended]
        if len(unseen) >= top_k:
            return unseen[:top_k]
        selected = list(unseen)
        selected_ids = {hit.parent_asin for hit in selected}
        for hit in learned_hits:
            if hit.parent_asin not in selected_ids:
                selected.append(hit)
                selected_ids.add(hit.parent_asin)
            if len(selected) >= top_k:
                break
        return selected[:top_k]

    def _exploration_hits(
        self,
        session_id: str,
        raw_hits: list[RetrievalHit],
        learned_hits: list[RetrievalHit],
        top_k: int,
        turn: int,
    ) -> list[RetrievalHit]:
        if not self.enable_late_exploration:
            return learned_hits
        if turn < int(PLACEHOLDER_HEURISTICS["raw_fallback_turn"]):
            return learned_hits
        if (
            len(learned_hits) >= 2
            and learned_hits[0].score - learned_hits[1].score
            >= float(PLACEHOLDER_HEURISTICS["exploration_confidence_margin"])
        ):
            return learned_hits

        recommended = self._recommended_history[session_id]
        history = self._candidate_history[session_id]
        novel = sorted(
            (value for parent_asin, value in history.items() if parent_asin not in recommended),
            key=lambda value: (
                -(
                    (value[3] ** float(PLACEHOLDER_HEURISTICS["history_frequency_weight"]))
                    / (float(PLACEHOLDER_HEURISTICS["history_rank_offset"]) + value[0])
                ),
                value[0],
                -value[1],
                value[2].parent_asin,
            ),
        )
        selected: list[RetrievalHit] = [
            RetrievalHit(
                hit.parent_asin,
                round(
                    (frequency ** float(PLACEHOLDER_HEURISTICS["history_frequency_weight"]))
                    / (float(PLACEHOLDER_HEURISTICS["history_rank_offset"]) + best_rank),
                    8,
                ),
                hit.signals,
            )
            for best_rank, _, hit, frequency in novel[:top_k]
        ]
        selected_ids = {hit.parent_asin for hit in selected}
        for hit in learned_hits:
            if hit.parent_asin not in selected_ids:
                selected.append(hit)
                selected_ids.add(hit.parent_asin)
            if len(selected) >= top_k:
                break
        return selected[:top_k]
