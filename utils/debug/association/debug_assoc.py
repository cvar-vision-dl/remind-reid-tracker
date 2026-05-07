# utils/debug/association/debug_assoc.py

from __future__ import annotations

from .debug_assoc_core import (
    _alt_candidate_summary,
    _candidate_local_context_diagnostics,
    _candidate_sets_trace,
    _candidate_veto_diagnostics,
    _column_all_close_to,
    _drop_uninformative_assoc_cols,
    _expected_neighbors_for_object,
    _label_list,
    _neighbor_sets_context_cfg,
    _report_decision_summary,
    _resolve_neighbor_sets_ctx,
)
from .debug_assoc_dataframes import (
    assoc_diagnostics_output_to_dataframe,
    assoc_output_to_dataframe,
    assoc_similarity_details_to_dataframe,
    context_veto_candidates_to_dataframe,
    local_context_candidates_to_dataframe,
    neighbor_sets_candidates_to_dataframe,
    neighbor_sets_class_options_to_dataframes,
    neighbor_sets_output_to_dataframes,
)
from .debug_assoc_print import (
    print_assoc_diagnostics_table,
    print_assoc_similarity_details_table,
    print_assoc_table,
    print_known_set_distance_disambiguation_table,
    print_neighbor_sets_table,
    print_postcreate_temporal_table,
)

__all__ = [
    '_alt_candidate_summary',
    '_candidate_local_context_diagnostics',
    '_candidate_sets_trace',
    '_candidate_veto_diagnostics',
    '_column_all_close_to',
    '_drop_uninformative_assoc_cols',
    '_expected_neighbors_for_object',
    '_label_list',
    '_neighbor_sets_context_cfg',
    '_report_decision_summary',
    '_resolve_neighbor_sets_ctx',
    'assoc_diagnostics_output_to_dataframe',
    'assoc_output_to_dataframe',
    'assoc_similarity_details_to_dataframe',
    'context_veto_candidates_to_dataframe',
    'local_context_candidates_to_dataframe',
    'neighbor_sets_candidates_to_dataframe',
    'neighbor_sets_class_options_to_dataframes',
    'neighbor_sets_output_to_dataframes',
    'print_assoc_diagnostics_table',
    'print_assoc_similarity_details_table',
    'print_assoc_table',
    'print_known_set_distance_disambiguation_table',
    'print_neighbor_sets_table',
    'print_postcreate_temporal_table',
]
