import {
  DET_COLORS,
  OUTCOME_COLORS,
  STATUS_COLORS,
  DOOR_COLORS,
  NODE_LABELS,
  MODULE_BY_NODE_ID,
  DEFAULT_NODE_W,
  DEFAULT_NODE_H,
  MAX_NODE_W,
  GRAPH_W,
  NODE_ROAD_CLEARANCE,
  ROAD_GRID_MARGIN,
  ROAD_TURN_PENALTY,
  ROAD_DET_SPREAD,
  VISUAL_SCORE_COLUMNS,
  NODE_GENERAL_DESCRIPTIONS,
  TREE_PHASES,
  NODE_LAYOUT_OFFSETS,
  VISUAL_GRAPH_EDGES,
} from './modules/viewer_config.js';

const state = {
  runs: [],
  schema: null,
  manifest: null,
  trace: null,
  memorySnapshot: null,
  selectedRunId: null,
  selectedFrameId: null,
  selectedClassId: null,
  selectedNodeId: null,
  selectedDetId: null,
  activeTabId: 'overview',
  openTabs: [],
  transform: { x: 0, y: 0, scale: 1 },
  frameNavigationBusy: false,
};

let currentNodeMatrixInfoMap = new Map();

const runSelect = document.getElementById('run-select');
const frameSelect = document.getElementById('frame-select');
const classSelect = document.getElementById('class-select');
const traceTitle = document.getElementById('trace-title');
const graphSvg = document.getElementById('graph-svg');
const graphRoot = document.getElementById('graph-root');
const framePreviewTitle = document.getElementById('frame-preview-title');
const framePreviewImage = document.getElementById('frame-preview-image');
const framePreviewEmpty = document.getElementById('frame-preview-empty');
const tabsStrip = document.getElementById('tabs-strip');
const overviewPane = document.getElementById('overview-pane');
const detailPane = document.getElementById('detail-pane');
const detailTabKind = document.getElementById('detail-tab-kind');
const detailTabTitle = document.getElementById('detail-tab-title');
const detailTabBadges = document.getElementById('detail-tab-badges');
const detailTabContent = document.getElementById('detail-tab-content');
const memoryOverview = document.getElementById('memory-overview');
const openObjectsTabButton = document.getElementById('open-objects-tab');
const graphPanel = document.querySelector('.graph-panel');

let currentGraphLayoutInfo = null;

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ error: response.statusText }));
    throw new Error(payload.error || response.statusText);
  }
  return response.json();
}

function detColor(detId) {
  const index = Math.abs(Number(detId)) % DET_COLORS.length;
  return DET_COLORS[index];
}

function outcomeColor(outcome) {
  return OUTCOME_COLORS[String(outcome || 'UNASSIGNED')] || OUTCOME_COLORS.UNASSIGNED;
}

function nodeLabel(nodeId) {
  return NODE_LABELS[String(nodeId)] || String(nodeId)
    .split('.')
    .slice(-1)[0]
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function displayNodeLabel(nodeId) {
  const synthetic = syntheticNode(String(nodeId));
  if (synthetic?.label) return synthetic.label;
  return nodeLabel(nodeId);
}

function moduleSpecForNode(nodeId) {
  return MODULE_BY_NODE_ID[String(nodeId)] || null;
}

function moduleLabelForNode(nodeId) {
  return moduleSpecForNode(nodeId)?.label || 'Módulo no clasificado';
}

function generalDescription(nodeId) {
  return NODE_GENERAL_DESCRIPTIONS[String(nodeId)] || 'Paso del pipeline de asociación.';
}

const NODE_WHY_TEXT = {
  'prepare.class_partition': 'Existe para que el resto del pipeline trabaje solo con la clase relevante del frame y no mezcle detecciones ni objetos ajenos.',
  'prepare.reliable_visual_anchors': 'Existe para dar a Hipótesis de sets una base visual muy fiable antes de construir contexto relacional.',
  'prepare.valid_detections': 'Existe para separar la rama de matching conocido de los casos que no pueden compararse limpiamente por falta de features.',
  'visual.build_candidates': 'Existe para abrir un espacio de alternativas visuales antes de aplicar contexto, gating y resolución global.',
  'visual.report_diagnosis': 'Existe para medir si la evidencia visual ya viene limpia o si hará falta más ayuda contextual para resolver.',
  'context.neighbor_sets_hypotheses': 'Existe para proponer un contexto relacional global del frame en vez de mirar cada detección de forma aislada.',
  'context.sets_activation': 'Existe para evitar que un contexto relacional débil o ruidoso contamine Uso de contexto por reporte, Shaping por candidato y Tablas finales de score.',
  'shape.allow_for_report': 'Existe para que no todas las detecciones usen contexto por igual cuando su lectura visual aún no lo soporta.',
  'shape.context_veto': 'Existe para decidir, candidato a candidato, qué sigue plausible y operativo antes de construir las tablas finales.',
  'shape.final_score_tables': 'Existe para reunir en una sola tabla operativa todo lo que sigue vivo antes de la resolución global.',
  'resolve.locks': 'Existe para cerrar coincidencias muy claras sin gastar una resolución global innecesaria.',
  'resolve.hungarian': 'Existe para resolver de forma conjunta las detecciones y objetos que todavía compiten entre sí.',
  'post.assignment_ambiguity': 'Existe para detectar si la asignación elegida sigue siendo discutible aunque ya haya pasado la resolución global.',
  'post.identity_stability': 'Existe para impedir cambios de identidad demasiado frágiles o poco consistentes en el tiempo.',
  'post.known_set_distance_disambiguation': 'Existe para romper ambigüedades conocidas usando memoria y señales adicionales tras la asignación principal.',
  'post.create_competition': 'Existe para decidir si un caso debe quedarse como conocido o si merece abrir la puerta a una identidad nueva.',
  'post.ambiguous_track_candidates': 'Existe para materializar qué detecciones pasan de verdad a la bolsa ambigua antes de la resolución temporal, y desde qué fuente llegan.',
  'post.provisional_reconciliation': 'Existe para traducir casos dudosos a salidas temporales coherentes en vez de forzar una decisión demasiado dura.',
  'post.final_decision_pack': 'Existe para aplicar la precedencia final entre matches, creates, ambiguos y provisionales antes de anotar el outcome legible.',
  'outcome.final_ambiguity': 'Existe para recalcular la claridad final del caso usando score_final antes de anotar el outcome legible.',
  'outcome.finalize': 'Existe para materializar una única salida legible por detección antes de pasar a update.',
};

const NODE_PREPARES_TEXT = {
  'prepare.class_partition': 'Deja la unidad de análisis frame+class ya delimitada, con sus detecciones y objetos de memoria visibles.',
  'prepare.reliable_visual_anchors': 'Deja referencias visuales muy fiables que alimentan directamente Hipótesis de sets y, desde ahí, el kernel contextual.',
  'prepare.valid_detections': 'Deja qué detecciones siguen a la rama de matching conocido y cuáles salen hacia create o rutas sin matching.',
  'visual.build_candidates': 'Deja un ranking visual base por detección que alimenta diagnóstico, veto, score final y resolución.',
  'visual.report_diagnosis': 'Deja una lectura compacta de fuerza visual que condiciona cuánto contexto merece usar cada detección.',
  'context.neighbor_sets_hypotheses': 'Deja hipótesis retenidas, shortlist, priors y soporte por objeto para que Activación de sets decida si el contexto entra en juego y para que después puedan usarlo Uso de contexto por reporte, Shaping por candidato y Tablas finales de score.',
  'context.sets_activation': 'Deja el contexto en estado activo, degradado o inactivo antes de que afecte a bonus, rescates o vetos.',
  'shape.allow_for_report': 'Deja marcado qué detecciones sí podrán usar contexto de sets y cuáles seguirán solo con evidencia visual.',
  'shape.context_veto': 'Deja decidido, por candidato, qué pares siguen plausibles y cuáles quedan bloqueados antes de construir la tabla final de score.',
  'shape.final_score_tables': 'Deja el ranking operativo por detección que usarán locks y Hungarian.',
  'resolve.locks': 'Deja cerrados los matches evidentes y reduce el problema que llega a Hungarian.',
  'resolve.hungarian': 'Deja una propuesta global de asignación coherente a nivel de clase.',
  'post.assignment_ambiguity': 'Deja señalizado si la salida todavía necesita guards o reinterpretación temporal.',
  'post.identity_stability': 'Deja aceptados o frenados los cambios de identidad más frágiles.',
  'post.known_set_distance_disambiguation': 'Deja ambigüedades conocidas reducidas cuando hay evidencia suficiente para romperlas.',
  'post.create_competition': 'Deja resuelta la competencia entre crear nuevo y conservar alternativas conocidas.',
  'post.ambiguous_track_candidates': 'Deja la bolsa efectiva de entradas ambiguas que se pasará a la resolución temporal y a la reconciliación posterior.',
  'post.provisional_reconciliation': 'Deja el caso traducido a una salida temporal consistente con la incertidumbre real.',
  'post.final_decision_pack': 'Deja un único bucket final por detección tras aplicar la precedencia entre match, create, ambiguous y provisional.',
  'outcome.final_ambiguity': 'Deja una lectura final strong/ambiguous/weak basada en score_final, ya sobre el resultado semántico final del caso.',
  'outcome.finalize': 'Deja la decisión final lista para que el pipeline de update la consuma sin reinterpretaciones extra.',
};

function nodeWhyText(nodeId) {
  return NODE_WHY_TEXT[String(nodeId)] || 'Existe para aportar una decisión intermedia necesaria antes del siguiente tramo del pipeline.';
}

function nodePreparesText(nodeId) {
  return NODE_PREPARES_TEXT[String(nodeId)] || 'Deja información preparada para el tramo siguiente del pipeline.';
}

function nodeHasCustomNarrative(nodeId) {
  return [
    'prepare.reliable_visual_anchors',
    'context.neighbor_sets_hypotheses',
    'context.sets_activation',
    'shape.allow_for_report',
    'shape.context_veto',
    'shape.final_score_tables',
    'resolve.hungarian',
    'post.assignment_ambiguity',
    'post.identity_stability',
    'post.create_competition',
    'post.ambiguous_track_candidates',
    'post.known_set_distance_disambiguation',
    'post.provisional_reconciliation',
    'visual.build_candidates',
    'visual.report_diagnosis',
    'post.final_decision_pack',
    'outcome.final_ambiguity',
    'outcome.finalize',
  ].includes(String(nodeId));
}

function syntheticNode(nodeId) {
  if (nodeId === 'synthetic.input_detections') {
    const detIds = state.trace?.det_ids || [];
    return {
      id: nodeId,
      label: 'Detecciones posibles',
      moduleLabel: 'Entradas',
      description: detIds.length
        ? 'Detecciones visibles de esta clase al comenzar el flujo.'
        : 'No hay detecciones visibles de esta clase en el frame.',
      listItems: detIds.length
        ? detIds.map((detId) => ({ label: `det ${detId}`, color: detColor(detId) }))
        : [{ label: 'sin detecciones', color: '#6a7779' }],
      color: '#7f6d44',
    };
  }
  if (nodeId === 'synthetic.input_memory') {
    const snapshotIds = new Set((state.trace?.snapshot_object_ids || []).map((value) => Number(value)));
    const rows = objectSnapshotRows().filter((row) => snapshotIds.has(Number(row.object_id)));
    return {
      id: nodeId,
      label: 'Objetos guardados',
      moduleLabel: 'Entradas',
      description: rows.length
        ? 'Objetos ya guardados en memoria que podrían explicar estas detecciones.'
        : 'No hay objetos persistidos relevantes para esta clase en este frame.',
      listItems: rows.length
        ? rows.map((row) => ({ label: `${row.label} · ID ${row.object_id}` }))
        : [{ label: 'sin objetos relevantes' }],
      color: '#7f6d44',
    };
  }
  return null;
}

function objectLabelForId(objectId) {
  const numericObjectId = Number(objectId);
  const row = objectSnapshotRows().find((item) => Number(item.object_id) === numericObjectId);
  if (row?.label) return row.label;
  return `obj_${numericObjectId}`;
}

function objectLabelsForIds(objectIds) {
  const ids = Array.isArray(objectIds) ? objectIds : [];
  return ids
    .map((objectId) => Number(objectId))
    .filter((objectId) => Number.isFinite(objectId))
    .map((objectId) => objectLabelForId(objectId));
}

function objectLabelsText(objectIds) {
  const labels = objectLabelsForIds(objectIds);
  return labels.length ? labels.join(', ') : '—';
}

function objectScoreMapText(value) {
  if (!value || typeof value !== 'object') return pretty(value);
  const entries = Object.entries(value)
    .map(([objectId, score]) => {
      const numericObjectId = Number(objectId);
      const objLabel = Number.isFinite(numericObjectId) ? objectLabelForId(numericObjectId) : String(objectId);
      return `${objLabel}: ${pretty(score)}`;
    });
  return entries.length ? entries.join(', ') : '—';
}

function detLabelsText(detIds) {
  const ids = Array.isArray(detIds) ? detIds : [];
  const labels = ids
    .map((detId) => Number(detId))
    .filter((detId) => Number.isFinite(detId))
    .map((detId) => `det ${detId}`);
  return labels.length ? labels.join(', ') : '—';
}

function assignmentText(value) {
  if (!value || typeof value !== 'object') return pretty(value);
  const entries = Object.entries(value)
    .map(([detId, objectId]) => {
      const numericDetId = Number(detId);
      const numericObjectId = Number(objectId);
      const detLabel = Number.isFinite(numericDetId) ? `det ${numericDetId}` : String(detId);
      const objLabel = Number.isFinite(numericObjectId) ? objectLabelForId(numericObjectId) : pretty(objectId);
      return `${detLabel} -> ${objLabel}`;
    });
  return entries.length ? entries.join(', ') : '—';
}

function formatValueForDisplay(key, value) {
  const normalizedKey = String(key || '');
  if (normalizedKey === 'object_id' || normalizedKey === 'best_object_id' || normalizedKey === 'second_object_id'
    || normalizedKey === 'final_object_id' || normalizedKey === 'locked_object_id' || normalizedKey === 'parent_oid'
    || normalizedKey === 'assigned_object_id' || normalizedKey === 'top_supported_object_id'
    || normalizedKey === 'committed_new_object_id') {
    return value == null ? '—' : objectLabelForId(value);
  }
  if (normalizedKey === 'det_id' || normalizedKey === 'create_det_id' || normalizedKey === 'selected_anchor_det_id') {
    return value == null ? '—' : `det ${pretty(value)}`;
  }
  if (normalizedKey === 'object_ids' || normalizedKey === 'shortlist_object_ids' || normalizedKey === 'anchor_object_ids'
    || normalizedKey === 'prior_object_ids' || normalizedKey === 'snapshot_object_ids' || normalizedKey === 'candidate_union'
    || normalizedKey === 'component_object_ids' || normalizedKey === 'class_object_overlap' || normalizedKey === 'global_anchor_object_ids'
    || normalizedKey === 'anchor_object_ids_global' || normalizedKey === 'support_known_ids'
    || normalizedKey === 'blocked_known_ids' || normalizedKey === 'provisional_support_ids'
    || normalizedKey === 'provisional_blocked_known_ids' || normalizedKey === 'provisional_related_known_ids'
    || normalizedKey === 'ambiguous_candidate_ids' || normalizedKey === 'candidate_ids'
    || normalizedKey === 'related_known_ids'
    || normalizedKey === 'committed_new_parent_ids' || normalizedKey === 'anchor_pair') {
    return objectLabelsText(value);
  }
  if (normalizedKey === 'det_ids' || normalizedKey === 'det_ids_explained' || normalizedKey === 'det_ids_unexplained'
    || normalizedKey === 'class_det_overlap' || normalizedKey === 'ambiguous_det_ids' || normalizedKey === 'component_det_ids'
    || normalizedKey === 'create_det_ids' || normalizedKey === 'pass_input_det_ids'
    || normalizedKey === 'pass_resolved_det_ids' || normalizedKey === 'pass_remaining_det_ids'
    || normalizedKey === 'det_pair') {
    return detLabelsText(value);
  }
  if (normalizedKey === 'ambiguous_candidate_scores' || normalizedKey === 'provisional_support_scores'
    || normalizedKey === 'provisional_blocked_known_scores' || normalizedKey === 'provisional_related_known_scores'
    || normalizedKey === 'candidate_scores' || normalizedKey === 'committed_new_parent_scores'
    || normalizedKey === 'support_known_scores' || normalizedKey === 'blocked_known_scores'
    || normalizedKey === 'related_known_scores') {
    return objectScoreMapText(value);
  }
  if (normalizedKey === 'current_assignment' || normalizedKey === 'stable_det_assignments') {
    return assignmentText(value);
  }
  return pretty(value);
}

function nodeSpecificListItems(nodeId) {
  if (String(nodeId) === 'visual.build_candidates') {
    const nodeRun = getNodeRun(nodeId);
    const rows = nodeRun?.detection_rows || [];
    const activeRows = state.selectedDetId == null
      ? rows
      : rows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));
    const noCandidateRows = activeRows
      .filter((row) => Number(row.candidate_count ?? 0) <= 0)
      .sort((a, b) => Number(a.det_id ?? 0) - Number(b.det_id ?? 0));
    if (noCandidateRows.length) {
      return noCandidateRows.map((row) => ({
        label: `det ${Number(row.det_id)} - sin candidatos`,
        color: detColor(row.det_id),
      }));
    }
    return null;
  }

  if (String(nodeId) === 'prepare.reliable_visual_anchors') {
    const nodeRun = getNodeRun(nodeId);
    const globalAnchorPairs = nodeRun?.values?.global_anchor_pairs || [];
    if (globalAnchorPairs.length) {
      return globalAnchorPairs.map((row) => ({
        label: `det ${Number(row.det_id)} -> ${objectLabelForId(row.object_id)}`,
        color: detColor(row.det_id),
      }));
    }
    const anchorPairs = nodeRun?.values?.anchor_pairs || [];
    if (anchorPairs.length) {
      return anchorPairs.map((row) => ({
        label: `det ${Number(row.det_id)} -> ${objectLabelForId(row.object_id)}`,
        color: detColor(row.det_id),
      }));
    }
    const globalAnchorObjectIds = nodeRun?.values?.global_anchor_object_ids || [];
    if (globalAnchorObjectIds.length) {
      return globalAnchorObjectIds.map((objectId) => ({
        label: objectLabelForId(objectId),
      }));
    }
    const anchorObjectIds = nodeRun?.values?.anchor_object_ids || [];
    if (anchorObjectIds.length) {
      return anchorObjectIds.map((objectId) => ({
        label: objectLabelForId(objectId),
      }));
    }
    return [{ label: 'sin anchors fiables' }];
  }

  if (String(nodeId) === 'context.neighbor_sets_hypotheses') {
    const nodeRun = getNodeRun(nodeId);
    const rows = sortContextObjectRows((nodeRun?.global_rows || []).filter((row) => row?.row_type === 'object_support'));
    if (rows.length) {
      return rows.slice(0, 5).map((row) => ({
        label: `${objectLabelForId(row.object_id)} · p=${pretty(row.prior)} · s=${pretty(row.support_sum)}`,
      }));
    }
    const shortlist = nodeRun?.values?.shortlist_object_ids || [];
    if (shortlist.length) {
      return shortlist.slice(0, 5).map((objectId) => ({
        label: objectLabelForId(objectId),
      }));
    }
    return [{ label: 'sin objetos plausibles visibles' }];
  }

  if (String(nodeId) === 'context.sets_activation') {
    const nodeRun = getNodeRun(nodeId);
    const values = nodeRun?.values || {};
    if (!nodeRun?.entered) return null;
    return [
      { label: `estado ${nodeRun?.decision?.branch || 'inactive'}` },
      { label: `quality ${pretty(values.quality)}` },
      { label: `shortlist ${pretty(values.shortlist_size ?? 0)}` },
    ];
  }

  if (String(nodeId) === 'shape.allow_for_report') {
    const nodeRun = getNodeRun(nodeId);
    const values = nodeRun?.values || {};
    if (!nodeRun?.entered) return null;
    return [
      { label: `permitidas ${pretty(values.allowed_count ?? 0)}` },
      { label: `bloqueadas ${pretty(values.blocked_count ?? 0)}` },
    ];
  }

  if (String(nodeId) === 'shape.context_veto') {
    const nodeRun = getNodeRun(nodeId);
    const rows = nodeRun?.candidate_rows || [];
    if (!nodeRun?.entered) return null;
    const kept = rows.filter((row) => Number(row.decision_keep) === 1).length;
    const vetoed = rows.filter((row) => Number(row.decision_keep) !== 1).length;
    return [
      { label: `sobreviven ${pretty(kept)}` },
      { label: `caen ${pretty(vetoed)}` },
    ];
  }

  if (String(nodeId) === 'shape.final_score_tables') {
    const nodeRun = getNodeRun(nodeId);
    const rows = nodeRun?.detection_rows || [];
    if (!nodeRun?.entered) return null;
    return rows.slice(0, 3).map((row) => ({
      label: `det ${pretty(row.det_id)} -> ${row.best_object_id != null ? objectLabelForId(row.best_object_id) : '—'}`,
    }));
  }
  return null;
}

function nodeOverviewDescription(nodeId) {
  const node = orderedNodes().find((item) => item.id === nodeId);
  const nodeRun = getNodeRun(nodeId);
  if (node && nodeRun) return summarizeNodeRun(node, nodeRun);
  return generalDescription(nodeId);
}

function displayNode(nodeId) {
  const synthetic = syntheticNode(nodeId);
  if (synthetic) return synthetic;
  const node = orderedNodes().find((item) => item.id === nodeId);
  if (!node) return null;
  const specificListItems = nodeSpecificListItems(nodeId);
  return {
    ...node,
    label: nodeLabel(nodeId),
    moduleLabel: moduleLabelForNode(nodeId),
    description: nodeOverviewDescription(nodeId),
    listItems: specificListItems || node.listItems || [],
    color: moduleSpecForNode(nodeId)?.color || '#6a7779',
  };
}

function pretty(value) {
  if (value == null) return '—';
  if (typeof value === 'number') return Number.isInteger(value) ? String(value) : value.toFixed(4);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (Array.isArray(value)) return value.length ? `[${value.join(', ')}]` : '[]';
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

function expression(check) {
  return `${pretty(check.lhs)} ${check.op} ${pretty(check.rhs)}`;
}

function clearChildren(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function orderedNodes() {
  return state.schema?.nodes || [];
}

function getNodeRun(nodeId) {
  return (state.trace?.node_runs || []).find((nodeRun) => nodeRun.node_id === nodeId) || null;
}

function nodeEntered(nodeRun) {
  return Boolean(nodeRun && nodeRun.entered);
}

function filteredRows(nodeRun, detId = state.selectedDetId) {
  if (!nodeRun) return { detectionRows: [], candidateRows: [], globalRows: [] };
  if (detId == null) {
    return {
      detectionRows: nodeRun.detection_rows || [],
      candidateRows: nodeRun.candidate_rows || [],
      globalRows: nodeRun.global_rows || [],
    };
  }
  const numeric = Number(detId);
  return {
    detectionRows: (nodeRun.detection_rows || []).filter((row) => Number(row.det_id) === numeric),
    candidateRows: (nodeRun.candidate_rows || []).filter((row) => Number(row.det_id) === numeric),
    globalRows: (nodeRun.global_rows || []).filter((row) => row.det_id == null || Number(row.det_id) === numeric),
  };
}

function gatherRelevantChecks(nodeRun, detId = state.selectedDetId) {
  if (!nodeRun) return [];
  const checks = [];
  const pushChecks = (rawChecks, meta) => {
    for (const [index, check] of (rawChecks || []).entries()) {
      const logicGroup = String(check?.logic_group || '').trim();
      const logicGroupLabel = String(check?.logic_group_label || '').trim();
      const logicOrderRaw = Number(check?.logic_order);
      const logicOrder = Number.isFinite(logicOrderRaw) ? logicOrderRaw : (index + 1);
      const baseGroupKey = meta.groupKey || 'node:global';
      const baseGroupLabel = meta.groupLabel || 'Checks globales del nodo';
      checks.push({
        ...check,
        __sourceKind: meta.sourceKind || 'node',
        __rowKind: meta.rowKind || null,
        __detId: meta.detId ?? null,
        __objectId: meta.objectId ?? null,
        __associationLabel: meta.associationLabel || 'nodo completo',
        __logicGroup: logicGroup || null,
        __logicGroupLabel: logicGroupLabel || null,
        __groupKey: logicGroup ? `${baseGroupKey}::${logicGroup}` : baseGroupKey,
        __groupLabel: logicGroupLabel ? `${baseGroupLabel} · ${logicGroupLabel}` : baseGroupLabel,
        __sourceOrder: index + 1,
        __logicOrder: logicOrder,
      });
    }
  };

  pushChecks(nodeRun.checks || [], {
    sourceKind: 'node',
    groupKey: 'node:global',
    groupLabel: 'Checks globales del nodo',
    associationLabel: 'nodo completo',
  });

  const rows = filteredRows(nodeRun, detId);
  for (const row of rows.detectionRows) {
    const numericDetId = Number(row.det_id);
    pushChecks(row.checks || [], {
      sourceKind: 'row',
      rowKind: 'detection',
      detId: numericDetId,
      groupKey: `detection:${numericDetId}`,
      groupLabel: `Checks de det ${numericDetId}`,
      associationLabel: `det ${numericDetId}`,
    });
  }
  for (const row of rows.candidateRows) {
    const numericDetId = Number(row.det_id);
    const numericObjectId = Number(row.object_id);
    const objectLabel = Number.isFinite(numericObjectId) ? objectLabelForId(numericObjectId) : 'obj_—';
    pushChecks(row.checks || [], {
      sourceKind: 'row',
      rowKind: 'candidate',
      detId: numericDetId,
      objectId: numericObjectId,
      groupKey: `candidate:${numericDetId}:${numericObjectId}`,
      groupLabel: `Checks de det ${numericDetId} -> ${objectLabel}`,
      associationLabel: `det ${numericDetId} -> ${objectLabel}`,
    });
  }
  for (const row of rows.globalRows) {
    const numericDetId = Number(row.det_id);
    const hasDetId = Number.isFinite(numericDetId);
    pushChecks(row.checks || [], {
      sourceKind: 'row',
      rowKind: 'global',
      detId: hasDetId ? numericDetId : null,
      groupKey: hasDetId ? `global-det:${numericDetId}` : 'global-aggregate',
      groupLabel: hasDetId ? `Checks globales ligados a det ${numericDetId}` : 'Checks globales agregados',
      associationLabel: hasDetId ? `det ${numericDetId} (ámbito global)` : 'ámbito global del nodo',
    });
  }

  const groupCounts = new Map();
  checks.forEach((check) => {
    groupCounts.set(check.__groupKey, Number(groupCounts.get(check.__groupKey) || 0) + 1);
  });

  const groupedChecks = new Map();
  checks.forEach((check) => {
    const bucket = groupedChecks.get(check.__groupKey) || [];
    bucket.push(check);
    groupedChecks.set(check.__groupKey, bucket);
  });

  const enriched = [];
  groupedChecks.forEach((bucket, groupKey) => {
    const orderedBucket = [...bucket].sort((left, right) => {
      const leftOrder = Number(left.__logicOrder ?? left.__sourceOrder ?? 0);
      const rightOrder = Number(right.__logicOrder ?? right.__sourceOrder ?? 0);
      if (leftOrder !== rightOrder) return leftOrder - rightOrder;
      return Number(left.__sourceOrder ?? 0) - Number(right.__sourceOrder ?? 0);
    });
    orderedBucket.forEach((check, index) => {
      enriched.push({
        ...check,
        __groupIndex: index + 1,
        __groupSize: Number(groupCounts.get(groupKey) || 1),
      });
    });
  });
  return enriched;
}

function nodeStatusClass(nodeRun) {
  if (!nodeRun) return 'na';
  if (!nodeRun.entered) return 'na';
  const decisionStatus = (nodeRun.decision || {}).status;
  if (decisionStatus === 'FAIL') return 'fail';
  if (decisionStatus === 'PASS') return 'pass';
  if (decisionStatus === 'N/A') return 'soft';
  const checks = gatherRelevantChecks(nodeRun, state.selectedDetId);
  if (!checks.length) return 'pass';
  if (checks.some((check) => check.passed === false)) return 'fail';
  if (checks.some((check) => check.passed == null)) return 'soft';
  return 'pass';
}

function nodeDoorState(nodeRun) {
  if (!nodeRun) return 'missing';
  if (!nodeRun.entered) return 'closed';
  const branch = (nodeRun.decision || {}).branch || '';
  const status = (nodeRun.decision || {}).status || '';
  if (status === 'FAIL') return 'blocked';
  if (branch === 'filtered') return 'filtered';
  if (branch === 'inactive' || branch === 'shortlist_empty' || status === 'N/A') return 'inactive';
  if (nodeStatusClass(nodeRun) === 'soft') return 'soft';
  return 'open';
}

function doorStateColor(doorState) {
  return DOOR_COLORS[doorState] || DOOR_COLORS.missing;
}

function detectionTouchesNode(nodeRun, detId, { requireEntered = true } = {}) {
  if (!nodeRun || detId == null) return false;
  if (requireEntered && !nodeEntered(nodeRun)) return false;
  const numericDetId = Number(detId);
  const participants = nodeRun.participants || {};
  if ((participants.det_ids || []).includes(numericDetId)) return true;
  if ((nodeRun.detection_rows || []).some((row) => Number(row.det_id) === numericDetId)) return true;
  if ((nodeRun.candidate_rows || []).some((row) => Number(row.det_id) === numericDetId)) return true;
  if ((nodeRun.global_rows || []).some((row) => Number(row.det_id) === numericDetId)) return true;
  return false;
}

function syntheticNodeTouchesDet(nodeId, detId) {
  const detIds = (state.trace?.det_ids || []).map((value) => Number(value));
  const numericDetId = Number(detId);
  if (nodeId === 'synthetic.input_detections') {
    return detIds.includes(numericDetId);
  }
  if (nodeId === 'synthetic.input_memory') {
    return detIds.includes(numericDetId);
  }
  return false;
}

function nodeTouchesDetId(nodeId, detId, options) {
  if (String(nodeId).startsWith('synthetic.')) {
    return syntheticNodeTouchesDet(nodeId, detId);
  }
  return detectionTouchesNode(getNodeRun(nodeId), detId, options);
}

function graphEdges() {
  return VISUAL_GRAPH_EDGES.map((edge) => ({
    ...edge,
    skipNodes: [...(edge.skipNodes || [])],
  }));
}

const MATRIX_CELL_SIZE = 34;
const MATRIX_CELL_GAP = 8;
const MATRIX_LABEL_COL_W = 76;
const MATRIX_HEADER_H = 92;
const MATRIX_TITLE_H = 14;
const MATRIX_TOP_GAP = 14;

const MATRIX_STATE_COLORS = {
  keep: '#4f8f57',
  warn: '#d89a2d',
  drop: '#c85a46',
  none: '#dfe4de',
};

const MATRIX_STATE_LABELS = {
  keep: 'activa',
  warn: 'dudosa',
  drop: 'eliminada',
  none: 'sin señal directa',
};

function traceDetIds() {
  return [...new Set((state.trace?.det_ids || []).map((value) => Number(value)))]
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);
}

function traceObjectIds() {
  return [...new Set((state.trace?.snapshot_object_ids || []).map((value) => Number(value)))]
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);
}

function shouldRenderDecisionMatrix(nodeId) {
  return !String(nodeId).startsWith('synthetic.')
    && String(nodeId) !== 'prepare.reliable_visual_anchors'
    && traceDetIds().length > 0
    && traceObjectIds().length > 0
    && Boolean(currentNodeMatrixInfoMap.get(String(nodeId))?.visible);
}

function decisionMatrixMetrics(nodeId) {
  if (!shouldRenderDecisionMatrix(nodeId)) return null;
  const detIds = traceDetIds();
  const objIds = traceObjectIds();
  const step = MATRIX_CELL_SIZE + MATRIX_CELL_GAP;
  const maxLabelChars = objIds.reduce(
    (maxChars, objectId) => Math.max(maxChars, String(objectLabelForId(objectId)).length),
    0
  );
  const headerHeight = Math.max(MATRIX_HEADER_H, Math.min(156, 34 + maxLabelChars * 4));
  return {
    detIds,
    objIds,
    step,
    titleHeight: MATRIX_TITLE_H,
    headerHeight,
    labelColWidth: MATRIX_LABEL_COL_W,
    cellSize: MATRIX_CELL_SIZE,
    width: MATRIX_LABEL_COL_W + objIds.length * step,
    height: MATRIX_TITLE_H + headerHeight + detIds.length * step,
  };
}

function matrixCellKey(detId, objectId) {
  return `${Number(detId)}::${Number(objectId)}`;
}

function createDecisionMatrix(defaultState = 'keep') {
  const detIds = traceDetIds();
  const objIds = traceObjectIds();
  const cells = new Map();
  detIds.forEach((detId) => {
    objIds.forEach((objectId) => {
      cells.set(matrixCellKey(detId, objectId), String(defaultState));
    });
  });
  return { detIds, objIds, cells };
}

function cloneDecisionMatrix(matrix) {
  return {
    detIds: [...(matrix?.detIds || [])],
    objIds: [...(matrix?.objIds || [])],
    cells: new Map(matrix?.cells || []),
  };
}

function matrixState(matrix, detId, objectId) {
  return matrix?.cells?.get(matrixCellKey(detId, objectId)) || 'none';
}

function setMatrixState(matrix, detId, objectId, stateName) {
  if (!matrix?.cells) return;
  matrix.cells.set(matrixCellKey(detId, objectId), String(stateName));
}

function setMatrixRowState(matrix, detId, stateName) {
  (matrix?.objIds || []).forEach((objectId) => setMatrixState(matrix, detId, objectId, stateName));
}

function downgradeAliveRowToWarn(matrix, detId) {
  (matrix?.objIds || []).forEach((objectId) => {
    const current = matrixState(matrix, detId, objectId);
    if (current === 'keep') setMatrixState(matrix, detId, objectId, 'warn');
  });
}

function markObjectsInRow(matrix, detId, objectIds, stateName) {
  [...new Set((objectIds || []).map((value) => Number(value)).filter((value) => Number.isFinite(value)))]
    .forEach((objectId) => setMatrixState(matrix, detId, objectId, stateName));
}

function candidateRowsByDet(candidateRows) {
  const byDet = new Map();
  (candidateRows || []).forEach((row) => {
    const detId = Number(row.det_id);
    if (!Number.isFinite(detId)) return;
    const bucket = byDet.get(detId) || [];
    bucket.push(row);
    byDet.set(detId, bucket);
  });
  return byDet;
}

function sortCandidateRowsForMatrix(rows) {
  return [...(rows || [])].sort((a, b) => {
    const rankA = Number(a.rank);
    const rankB = Number(b.rank);
    const hasRankA = Number.isFinite(rankA);
    const hasRankB = Number.isFinite(rankB);
    if (hasRankA || hasRankB) {
      if (!hasRankA) return 1;
      if (!hasRankB) return -1;
      if (rankA !== rankB) return rankA - rankB;
    }
    const scoreA = Number(a.score_final ?? a.score_assign ?? a.score_sim ?? 0);
    const scoreB = Number(b.score_final ?? b.score_assign ?? b.score_sim ?? 0);
    if (scoreA !== scoreB) return scoreB - scoreA;
    return Number(a.object_id ?? 0) - Number(b.object_id ?? 0);
  });
}

function candidateRowColorState(row, index) {
  const keep = Number(row.decision_keep ?? row.ctx_keep ?? 1) === 1;
  if (!keep) return 'drop';
  return index === 0 ? 'keep' : 'warn';
}

function applyExplicitCandidateRows(matrix, nodeRun, resolveState) {
  const byDet = candidateRowsByDet(nodeRun?.candidate_rows || []);
  byDet.forEach((rows, detId) => {
    setMatrixRowState(matrix, detId, 'drop');
    sortCandidateRowsForMatrix(rows).forEach((row, index) => {
      const objectId = Number(row.object_id);
      if (!Number.isFinite(objectId)) return;
      setMatrixState(matrix, detId, objectId, resolveState(row, index, rows));
    });
  });
}

function normalizeAssignmentMap(candidate) {
  const out = new Map();
  if (!candidate || typeof candidate !== 'object') return out;
  Object.entries(candidate).forEach(([detId, objectId]) => {
    const detNumeric = Number(detId);
    const objectNumeric = Number(objectId);
    if (Number.isFinite(detNumeric) && Number.isFinite(objectNumeric)) {
      out.set(detNumeric, objectNumeric);
    }
  });
  return out;
}

function matricesDiffer(left, right) {
  const detIds = left?.detIds || [];
  const objIds = left?.objIds || [];
  for (const detId of detIds) {
    for (const objectId of objIds) {
      if (matrixState(left, detId, objectId) !== matrixState(right, detId, objectId)) return true;
    }
  }
  return false;
}

function buildNodeDecisionMatrixMap() {
  const detIds = traceDetIds();
  const objIds = traceObjectIds();
  const matrixMap = new Map();
  if (!detIds.length || !objIds.length) return matrixMap;

  let current = createDecisionMatrix('keep');
  orderedNodes().forEach((node) => {
    const nodeId = String(node.id);
    const nodeRun = getNodeRun(nodeId);
    let next = cloneDecisionMatrix(current);
    let hasDirectSignal = false;

    if (nodeRun?.entered) {
      if (nodeId === 'prepare.class_partition') {
        next = createDecisionMatrix('keep');
        hasDirectSignal = true;
      } else if (nodeId === 'visual.build_candidates') {
        const byDet = candidateRowsByDet(nodeRun?.candidate_rows || []);
        byDet.forEach((rows, detId) => {
          setMatrixRowState(next, detId, 'none');
          sortCandidateRowsForMatrix(rows).forEach((row, index) => {
            const objectId = Number(row.object_id);
            if (!Number.isFinite(objectId)) return;
            setMatrixState(next, detId, objectId, index === 0 ? 'keep' : 'warn');
          });
        });
        (nodeRun.detection_rows || []).forEach((row) => {
          if (Number(row.candidate_count ?? 0) <= 0) setMatrixRowState(next, Number(row.det_id), 'drop');
        });
        hasDirectSignal = Boolean((nodeRun.candidate_rows || []).length || (nodeRun.detection_rows || []).length);
      } else if (nodeId === 'visual.report_diagnosis') {
        (nodeRun.detection_rows || []).forEach((row) => {
          const status = String(row.status || '').toUpperCase();
          if (status === 'AMBIGUOUS' || status === 'WEAK') downgradeAliveRowToWarn(next, Number(row.det_id));
        });
        hasDirectSignal = Boolean((nodeRun.detection_rows || []).some((row) => {
          const status = String(row.status || '').toUpperCase();
          return status === 'AMBIGUOUS' || status === 'WEAK';
        }));
      } else if (nodeId === 'prepare.valid_detections') {
        (nodeRun.detection_rows || []).forEach((row) => {
          if (!Boolean(row.valid)) setMatrixRowState(next, Number(row.det_id), 'drop');
        });
        hasDirectSignal = Boolean((nodeRun.detection_rows || []).some((row) => !Boolean(row.valid)));
      } else if (nodeId === 'shape.allow_for_report') {
        (nodeRun.detection_rows || []).forEach((row) => {
          if (!Boolean(row.allowed)) downgradeAliveRowToWarn(next, Number(row.det_id));
        });
        hasDirectSignal = Boolean((nodeRun.detection_rows || []).some((row) => !Boolean(row.allowed)));
      } else if (nodeId === 'shape.context_veto') {
        applyExplicitCandidateRows(next, nodeRun, candidateRowColorState);
        hasDirectSignal = Boolean((nodeRun.candidate_rows || []).length);
      } else if (nodeId === 'shape.final_score_tables') {
        applyExplicitCandidateRows(next, nodeRun, (_row, index) => (index === 0 ? 'keep' : 'warn'));
        (nodeRun.detection_rows || []).forEach((row) => {
          if (Number(row.candidate_count ?? 0) <= 0) setMatrixRowState(next, Number(row.det_id), 'drop');
        });
        hasDirectSignal = Boolean((nodeRun.candidate_rows || []).length || (nodeRun.detection_rows || []).some((row) => Number(row.candidate_count ?? 0) <= 0));
      } else if (nodeId === 'resolve.locks') {
        (nodeRun.detection_rows || []).forEach((row) => {
          const detId = Number(row.det_id);
          const objectId = Number(row.locked_object_id);
          if (!Boolean(row.locked) || !Number.isFinite(detId)) return;
          setMatrixRowState(next, detId, 'drop');
          if (Number.isFinite(objectId)) setMatrixState(next, detId, objectId, 'keep');
        });
        hasDirectSignal = Boolean((nodeRun.detection_rows || []).some((row) => Boolean(row.locked)));
      } else if (nodeId === 'resolve.hungarian') {
        (nodeRun.global_rows || []).forEach((row) => {
          const detId = Number(row.det_id);
          if (!Number.isFinite(detId)) return;
          setMatrixRowState(next, detId, 'drop');
          if (String(row.kind || '') === 'match') {
            const objectId = Number(row.object_id);
            if (Number.isFinite(objectId)) setMatrixState(next, detId, objectId, 'keep');
          }
        });
        hasDirectSignal = Boolean((nodeRun.global_rows || []).length);
      } else if (nodeId === 'post.assignment_ambiguity') {
        (nodeRun.global_rows || []).forEach((row) => {
          const ambiguousDetIds = (row.ambiguous_det_ids || []).map((value) => Number(value)).filter((value) => Number.isFinite(value));
          ambiguousDetIds.forEach((detId) => downgradeAliveRowToWarn(next, detId));
          const currentAssignment = normalizeAssignmentMap(row.current_assignment);
          currentAssignment.forEach((objectId, detId) => {
            if (Boolean(row.is_ambiguous)) setMatrixState(next, detId, objectId, 'warn');
          });
        });
        hasDirectSignal = Boolean((nodeRun.global_rows || []).some((row) => Boolean(row.is_ambiguous)));
      } else if (nodeId === 'post.identity_stability') {
        (nodeRun.detection_rows || []).forEach((row) => {
          const detId = Number(row.det_id);
          const finalObjectId = Number(row.final_object_id);
          if (!Number.isFinite(detId)) return;
          if (!Number.isFinite(finalObjectId)) {
            setMatrixRowState(next, detId, 'drop');
            return;
          }
          setMatrixRowState(next, detId, 'drop');
          setMatrixState(next, detId, finalObjectId, String(row.state || '') === 'kept' ? 'keep' : 'warn');
        });
        hasDirectSignal = Boolean((nodeRun.detection_rows || []).length);
      } else if (nodeId === 'post.create_competition') {
        (nodeRun.global_rows || []).forEach((row) => {
          const detId = Number(row.create_det_id);
          const parentOid = Number(row.parent_oid);
          if (!Number.isFinite(detId)) return;
          setMatrixRowState(next, detId, 'drop');
          if (Number.isFinite(parentOid)) setMatrixState(next, detId, parentOid, Boolean(row.selected) ? 'warn' : 'drop');
        });
        hasDirectSignal = Boolean((nodeRun.global_rows || []).length);
      } else if (nodeId === 'post.known_set_distance_disambiguation') {
        (nodeRun.global_rows || []).forEach((row) => {
          const resolvedAssignments = normalizeAssignmentMap(row.stable_det_assignments);
          const status = String(row.status || '').toLowerCase();
          const resolvedState = status === 'resolved' ? 'keep' : 'warn';
          if (resolvedAssignments.size) {
            resolvedAssignments.forEach((objectId, detId) => {
              setMatrixRowState(next, detId, 'drop');
              setMatrixState(next, detId, objectId, resolvedState);
            });
            return;
          }
          const detGroup = (row.det_ids || row.component_det_ids || []).map((value) => Number(value)).filter((value) => Number.isFinite(value));
          const objectGroup = (row.candidate_union || row.component_object_ids || []).map((value) => Number(value)).filter((value) => Number.isFinite(value));
          detGroup.forEach((detId) => {
            objectGroup.forEach((objectId) => {
              const currentState = matrixState(next, detId, objectId);
              if (currentState === 'drop') return;
              setMatrixState(next, detId, objectId, resolvedState);
            });
          });
        });
        hasDirectSignal = Boolean((nodeRun.global_rows || []).length);
      } else if (nodeId === 'post.provisional_reconciliation') {
        (nodeRun.detection_rows || []).forEach((row) => {
          const detId = Number(row.det_id);
          if (!Number.isFinite(detId)) return;
          const supportIds = [
            ...(row.support_known_ids || []),
            ...(row.blocked_known_ids || []),
            row.best_object_id,
            row.top_supported_object_id,
          ].map((value) => Number(value)).filter((value) => Number.isFinite(value));
          if (!supportIds.length) return;
          setMatrixRowState(next, detId, 'drop');
          markObjectsInRow(next, detId, supportIds, 'warn');
        });
        hasDirectSignal = Boolean((nodeRun.detection_rows || []).some((row) => {
          const supportIds = [
            ...(row.support_known_ids || []),
            ...(row.blocked_known_ids || []),
            row.best_object_id,
            row.top_supported_object_id,
          ].map((value) => Number(value)).filter((value) => Number.isFinite(value));
          return supportIds.length > 0;
        }));
      } else if (nodeId === 'outcome.finalize') {
        (nodeRun.detection_rows || []).forEach((row) => {
          const detId = Number(row.det_id);
          if (!Number.isFinite(detId)) return;
          const finalDecision = String(row.final_decision || '').toUpperCase();
          setMatrixRowState(next, detId, 'drop');
          if (finalDecision === 'MATCH') {
            const objectId = Number(row.final_object_id);
            if (Number.isFinite(objectId)) setMatrixState(next, detId, objectId, 'keep');
            return;
          }
          if (finalDecision === 'AMBIGUOUS_TRACK') {
            markObjectsInRow(next, detId, row.ambiguous_candidate_ids || [], 'warn');
            return;
          }
          if (finalDecision.startsWith('PROVISIONAL')) {
            markObjectsInRow(next, detId, row.provisional_related_known_ids || [], 'warn');
          }
        });
        hasDirectSignal = Boolean((nodeRun.detection_rows || []).length);
      }
    }

    const changed = matricesDiffer(current, next);
    const forceVisible = nodeId === 'visual.build_candidates' && hasDirectSignal;
    matrixMap.set(nodeId, {
      matrix: next,
      visible: Boolean(nodeRun?.entered) && hasDirectSignal && (changed || forceVisible),
      changed,
      hasDirectSignal,
    });
    current = next;
  });

  return matrixMap;
}

function edgeIsBypass(edge) {
  return Array.isArray(edge?.skipNodes) && edge.skipNodes.length > 0;
}

function nodeActiveForDet(nodeId, detId) {
  if (String(nodeId).startsWith('synthetic.')) return syntheticNodeTouchesDet(nodeId, detId);
  return nodeTouchesDetId(nodeId, detId, { requireEntered: true });
}

function edgeFollowedByDet(edge, detId) {
  if (!nodeActiveForDet(edge.from, detId) || !nodeActiveForDet(edge.to, detId)) return false;
  if (!edgeIsBypass(edge)) return true;
  return (edge.skipNodes || []).every((nodeId) => !nodeActiveForDet(nodeId, detId));
}

function buildBypassLaneMap(edges, layout) {
  const laneMap = new Map();
  const groupCounts = new Map();
  const sideCounts = new Map();
  const destinationOrder = new Map(
    [...new Set(
      edges
        .filter((edge) => edgeIsBypass(edge))
        .map((edge) => String(edge.to))
    )]
      .sort((a, b) => {
        const nodeA = layout?.[String(a)] || {};
        const nodeB = layout?.[String(b)] || {};
        const yA = Number(nodeA.y ?? Number.MAX_SAFE_INTEGER);
        const yB = Number(nodeB.y ?? Number.MAX_SAFE_INTEGER);
        if (yA !== yB) return yA - yB;
        const xA = Number(nodeA.x ?? Number.MAX_SAFE_INTEGER);
        const xB = Number(nodeB.x ?? Number.MAX_SAFE_INTEGER);
        if (xA !== xB) return xA - xB;
        return String(a).localeCompare(String(b));
      })
      .map((nodeId, index) => [String(nodeId), index])
  );

  edges.forEach((edge) => {
    if (!edgeIsBypass(edge)) return;
    const groupKey = String(edge.to);
    const laneIndex = groupCounts.get(groupKey) || 0;
    const preferredSide = ((destinationOrder.get(groupKey) || 0) % 2 === 0) ? 'right' : 'left';
    const alternateSide = preferredSide === 'right' ? 'left' : 'right';
    const side = (Math.floor(laneIndex / 2) % 2 === 0) ? preferredSide : alternateSide;
    const sideKey = `${groupKey}::${side}`;
    const sideIndex = sideCounts.get(sideKey) || 0;
    laneMap.set(`${edge.from}->${edge.to}::${(edge.skipNodes || []).join('|')}`, {
      laneIndex,
      side,
      sideIndex,
    });
    groupCounts.set(groupKey, laneIndex + 1);
    sideCounts.set(sideKey, sideIndex + 1);
  });
  return laneMap;
}

function getFrameEntries(frameId) {
  return (state.manifest?.class_entries || []).filter((entry) => Number(entry.frame_id) === Number(frameId));
}

function manifestHasFramePreview(frameId) {
  return (state.manifest?.frame_previews || []).some((item) => Number(item.frame_id) === Number(frameId));
}

function manifestHasMemorySnapshot(frameId) {
  return (state.manifest?.memory_snapshots || []).some((item) => Number(item.frame_id) === Number(frameId));
}

function getOutcomeRows() {
  return getNodeRun('outcome.finalize')?.detection_rows || [];
}

function getOutcomeRow(detId) {
  return getOutcomeRows().find((row) => Number(row.det_id) === Number(detId)) || null;
}

function candidateRowsForFocus(nodeRun, detId = state.selectedDetId) {
  if (!nodeRun) return [];
  const rows = filteredRows(nodeRun, detId).candidateRows;
  if (rows.length) return rows;
  if (detId == null) return nodeRun.candidate_rows || [];
  return [];
}

function allCandidateRows(nodeId) {
  return getNodeRun(nodeId)?.candidate_rows || [];
}

function isDroppedCandidate(row) {
  if (row.decision_keep != null) return Number(row.decision_keep) !== 1;
  if (row.alive_after != null) return !Boolean(row.alive_after);
  return false;
}

function candidateImpact(nodeRun, detId = state.selectedDetId) {
  if (!nodeRun) return null;
  const sourceRows = candidateRowsForFocus(nodeRun, detId);
  if (!sourceRows.length) return null;

  let keep = 0;
  let drop = 0;
  for (const row of sourceRows) {
    if (isDroppedCandidate(row)) drop += 1;
    else keep += 1;
  }
  return { total: sourceRows.length, keep, drop };
}

function candidateImpactText(nodeRun, detId = state.selectedDetId) {
  const impact = candidateImpact(nodeRun, detId);
  if (!impact) return null;
  if (impact.drop > 0) return `cand ${impact.total} -> ${impact.keep} · drop ${impact.drop}`;
  return `cand ${impact.total} activos`;
}

function candidateDropPreview(nodeRun, detId = state.selectedDetId) {
  const rows = candidateRowsForFocus(nodeRun, detId).filter((row) => isDroppedCandidate(row));
  if (!rows.length) return null;
  const preview = rows.slice(0, 2).map((row) => `obj ${row.object_id} x ${row.gate_reason || row.veto_reason || row.reason || 'DROP'}`);
  const suffix = rows.length > 2 ? ` +${rows.length - 2}` : '';
  return `${preview.join(' · ')}${suffix}`;
}

function candidateKeepPreview(nodeRun, detId = state.selectedDetId) {
  const rows = candidateRowsForFocus(nodeRun, detId).filter((row) => !isDroppedCandidate(row));
  if (!rows.length) return null;
  const preview = rows.slice(0, 3).map((row) => `obj ${row.object_id}`);
  const suffix = rows.length > 3 ? ` +${rows.length - 3}` : '';
  return `${preview.join(', ')}${suffix}`;
}

function traceFocusDetIds() {
  if (state.selectedDetId != null) return [Number(state.selectedDetId)];
  return [...new Set((state.trace?.det_ids || []).map((value) => Number(value)).filter((value) => Number.isFinite(value)))]
    .sort((a, b) => a - b);
}

function pathReasonLabel(reason) {
  const labels = {
    NO_VALID_DETECTIONS: 'ninguna detección válida llegó a esta fase, normalmente porque faltan features',
    NO_SCORE_ROWS: 'no sobrevivió ninguna fila det -> objeto tras el shaping y el gating',
    NO_FEATURES: 'la detección no tenía features comparables para entrar al matching',
    ALL_RESOLVED_BY_LOCKS: 'los locks resolvieron todo antes de llegar a Hungarian',
    no_candidates: 'no se construyeron candidatos visuales útiles en esta clase',
    candidates_built: 'sí se construyeron candidatos visuales y el flujo sigue por la secuencia mostrada en la traza',
    has_valid_detections: 'hay detecciones válidas para seguir evaluando matching',
    no_valid_detections: 'no queda ninguna detección válida para matching',
    has_reliable_anchors: 'aparecen anchors fiables que pueden apoyar el contexto',
    no_reliable_anchors: 'no aparece ningún anchor suficientemente claro y el contexto sigue sin esos apoyos',
    diagnosed: 'el bloque deja el diagnóstico visual ya clasificado para que lo consuman Filtro de detecciones válidas y Uso de contexto por reporte',
    shortlist_ready: 'hay shortlist, rescate o prior contextual disponible',
    shortlist_empty: 'no hay shortlist contextual útil y el bloque queda sin efecto práctico',
    filtered: 'el bloque ha filtrado candidatos o detecciones para la siguiente fase',
    packed: 'el bloque consolida los buckets finales sin conflicto antes del outcome legible',
    finalized: 'este nodo ya materializa la salida final',
    evaluated: 'este guard post-assignment se evaluó con la información disponible',
    NO_DETECTIONS_IN_CLASS: 'no hay detecciones trazables en esta clase para materializar el paso',
    NO_INITIAL_MATCHES: 'no llegaron matches iniciales a esta guard de estabilidad',
    NO_MATCHES_TO_COMPARE: 'no había matches iniciales comparables para ejecutar esta guard',
    NO_CREATES_OR_MATCHES: 'no llegaron creates ni matches a esta fase para esta clase',
    NO_DISAMBIGUATION_COMPONENTS: 'no quedaron componentes ambiguos que pudieran entrar en known-set-distance',
    NO_POSTCREATE_DEBUG: 'no hubo decisiones postcreate temporales trazables en esta clase',
    NO_AMBIGUOUS_CANDIDATES: 'no se materializó ninguna entrada ambigua para pasar a la resolución temporal',
    policy_disabled: 'la policy correspondiente está desactivada y el nodo solo deja constancia de ello',
    no_components_to_compare: 'no había componentes suficientes para comparar asignaciones completas',
    no_ambiguous_components: 'se compararon componentes, pero ninguno quedó ambiguo',
    ambiguous_components_found: 'al menos un componente sigue siendo ambiguo tras la resolución principal',
    no_competitions: 'no apareció ninguna competición create-vs-known en esta clase',
    competitions_found: 'sí aparecieron competiciones create-vs-known relevantes',
    candidates_built: 'se materializó la bolsa ambigua real que entra en la resolución temporal',
  };
  return labels[String(reason)] || pretty(reason);
}

function outgoingVisualEdges(nodeId) {
  return VISUAL_GRAPH_EDGES.filter((edge) => String(edge.from) === String(nodeId));
}

function incomingVisualEdges(nodeId) {
  return VISUAL_GRAPH_EDGES.filter((edge) => String(edge.to) === String(nodeId));
}

function skippedReasonForNode(nodeId) {
  return String(getNodeRun(nodeId)?.skipped_reason || '');
}

function resolvedBuildCandidatesPath(nodeRun) {
  if (!nodeRun) return '';
  const skipReasons = [
    skippedReasonForNode('shape.allow_for_report'),
    skippedReasonForNode('shape.context_veto'),
    skippedReasonForNode('shape.final_score_tables'),
    skippedReasonForNode('resolve.locks'),
    skippedReasonForNode('resolve.hungarian'),
  ].filter(Boolean);
  if (skipReasons.includes('NO_VALID_DETECTIONS')) return 'NO_VALID_DETECTIONS';
  if (skipReasons.includes('NO_SCORE_ROWS')) return 'NO_SCORE_ROWS';

  const filtered = filteredRows(nodeRun, state.selectedDetId);
  const detRows = filtered.detectionRows.length
    ? filtered.detectionRows
    : (nodeRun.detection_rows || []);
  const totalCandidates = detRows.reduce((acc, row) => acc + Number(row.candidate_count || 0), 0);
  if (totalCandidates <= 0) return 'NO_CANDIDATES_UNRESOLVED';
  return 'CANDIDATES_BUILT';
}

function resolvedBuildCandidatesBranchNode(nodeRun) {
  const resolvedPath = resolvedBuildCandidatesPath(nodeRun);
  if (resolvedPath === 'NO_VALID_DETECTIONS') {
    return 'prepare.valid_detections';
  }
  if (resolvedPath === 'NO_SCORE_ROWS') {
    return 'shape.final_score_tables';
  }
  return '';
}

function semanticPathOptionsForNode(nodeId, nodeRun) {
  if (!nodeRun) return [];
  const stringNodeId = String(nodeId);

  if (stringNodeId === 'prepare.valid_detections') {
    const rows = filteredRows(nodeRun, state.selectedDetId).detectionRows.length
      ? filteredRows(nodeRun, state.selectedDetId).detectionRows
      : (nodeRun.detection_rows || []);
    const validCount = rows.filter((row) => Boolean(row.valid)).length;
    const invalidCount = rows.filter((row) => !Boolean(row.valid)).length;
    return [
      {
        label: 'Seguir hacia matching',
        condition: 'La detección tiene features y entra como válida en el matching.',
        state: validCount > 0 ? 'tomado' : 'no tomado',
        note: validCount > 0
          ? 'Estas detecciones pueden seguir hacia allow_for_report y las tablas finales de score.'
          : 'No hay detecciones válidas en el foco actual.',
      },
      {
        label: 'Desvío por NO_FEATURES',
        condition: 'La detección no tiene features comparables.',
        state: invalidCount > 0 ? 'tomado' : 'no tomado',
        note: invalidCount > 0
          ? 'Estas detecciones salen del matching conocido, pero todavía siguen por la rama post-assignment de create.'
          : 'No aparece este caso en el foco actual.',
      },
    ];
  }

  if (stringNodeId === 'shape.final_score_tables') {
    const rows = filteredRows(nodeRun, state.selectedDetId).detectionRows.length
      ? filteredRows(nodeRun, state.selectedDetId).detectionRows
      : (nodeRun.detection_rows || []);
    const totalCandidates = rows.reduce((acc, row) => acc + Number(row.candidate_count || 0), 0);
    return [
      {
        label: 'Resolver con locks y Hungarian',
        condition: 'Las tablas finales conservan al menos una fila det -> objeto.',
        state: totalCandidates > 0 ? 'tomado' : 'no tomado',
        note: totalCandidates > 0
          ? 'Hay score final suficiente para intentar resolución global.'
          : 'No se puede abrir resolución global porque las tablas quedan vacías.',
      },
      {
        label: 'Caída a la rama create',
        condition: 'Las tablas finales quedan vacías y se serializa NO_SCORE_ROWS.',
        state: totalCandidates <= 0 ? 'tomado' : 'no tomado',
        note: totalCandidates <= 0
          ? 'Locks y Hungarian se saltan porque ya no hay filas de score para resolver, y el flujo continúa por post-assignment.'
          : 'No aplica porque sí sobreviven filas de score.',
      },
    ];
  }

  if (stringNodeId === 'resolve.locks') {
    const rows = filteredRows(nodeRun, state.selectedDetId).detectionRows.length
      ? filteredRows(nodeRun, state.selectedDetId).detectionRows
      : (nodeRun.detection_rows || []);
    const lockedCount = rows.filter((row) => Boolean(row.locked)).length;
    const unlockedCount = rows.length - lockedCount;
    return [
      {
        label: 'Cerrar aquí la resolución',
        condition: 'Los locks resuelven todas las detecciones restantes.',
        state: lockedCount > 0 && unlockedCount <= 0 ? 'tomado' : 'no tomado',
        note: lockedCount > 0 && unlockedCount <= 0
          ? 'Hungarian se salta con el motivo ALL_RESOLVED_BY_LOCKS.'
          : 'No basta con locks para resolverlo todo en esta traza.',
      },
      {
        label: 'Seguir a Hungarian',
        condition: 'Queda al menos una detección sin cerrar tras aplicar locks.',
        state: unlockedCount > 0 ? 'tomado' : 'no tomado',
        note: unlockedCount > 0
          ? 'Todavía queda conflicto global por resolver.'
          : 'No queda trabajo para Hungarian en el foco actual.',
      },
    ];
  }

  return [];
}

function edgeStateForCurrentFocus(edge) {
  const focusDetIds = traceFocusDetIds();
  const relevantDetIds = focusDetIds.filter((detId) => nodeActiveForDet(edge.from, detId));
  const followedDetIds = relevantDetIds.filter((detId) => edgeFollowedByDet(edge, detId));
  const targetRun = getNodeRun(edge.to);
  if (followedDetIds.length) {
    return {
      state: 'tomado',
      note: state.selectedDetId != null
        ? `camino activo para det ${state.selectedDetId}`
        : `lo siguen ${followedDetIds.length} detecciones en este frame/clase`,
    };
  }
  if (targetRun && !targetRun.entered && targetRun.skipped_reason) {
    return {
      state: 'no tomado',
      note: `destino saltado: ${pathReasonLabel(targetRun.skipped_reason)}`,
    };
  }
  return {
    state: 'inactivo',
    note: 'no aparece como camino activo en este foco',
  };
}

function centerNodeInGraphPanel(nodeId) {
  if (!graphPanel) return;
  const targetNodeId = String(nodeId || '');
  if (!targetNodeId) return;
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      const escapedNodeId = targetNodeId.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
      const nodeGroup = graphRoot.querySelector(`[data-node-id="${escapedNodeId}"]`);
      if (!(nodeGroup instanceof SVGGElement)) return;
      const nodeRect = nodeGroup.getBoundingClientRect();
      const panelRect = graphPanel.getBoundingClientRect();
      const targetLeft = graphPanel.scrollLeft
        + (nodeRect.left - panelRect.left)
        - (panelRect.width - nodeRect.width) / 2;
      const targetTop = graphPanel.scrollTop
        + (nodeRect.top - panelRect.top)
        - (panelRect.height - nodeRect.height) / 2;
      graphPanel.scrollTo({
        left: Math.max(0, targetLeft),
        top: Math.max(0, targetTop),
        behavior: 'smooth',
      });
    });
  });
}

function focusNodeInGraph(nodeId) {
  const targetNodeId = String(nodeId || '');
  if (!targetNodeId) return;
  state.selectedNodeId = targetNodeId;
  if (state.activeTabId !== 'overview') {
    state.activeTabId = 'overview';
    renderTabStrip();
    renderVisiblePane();
  }
  renderGraph();
  centerNodeInGraphPanel(targetNodeId);
}

function nearestUpstreamConvergence(nodeId) {
  const immediateEdges = incomingVisualEdges(nodeId);
  if (immediateEdges.length !== 1) {
    return {
      immediateEdges,
      frontierNodeId: null,
      frontierEdges: [],
      viaChainNodeIds: [],
    };
  }

  const chain = [];
  const visited = new Set([String(nodeId)]);
  let currentNodeId = String(nodeId);

  while (true) {
    const incoming = incomingVisualEdges(currentNodeId);
    if (incoming.length !== 1) break;
    const edge = incoming[0];
    const parentNodeId = String(edge.from);
    if (visited.has(parentNodeId)) break;
    visited.add(parentNodeId);
    chain.push(parentNodeId);

    const parentIncoming = incomingVisualEdges(parentNodeId);
    if (parentIncoming.length > 1) {
      return {
        immediateEdges,
        frontierNodeId: parentNodeId,
        frontierEdges: parentIncoming,
        viaChainNodeIds: chain.slice().reverse(),
      };
    }
    if (parentIncoming.length === 0) break;
    currentNodeId = parentNodeId;
  }

  return {
    immediateEdges,
    frontierNodeId: null,
    frontierEdges: [],
    viaChainNodeIds: [],
  };
}

function summarizeCurrentPath(nodeId, nodeRun) {
  if (!nodeRun) return 'Nodo no presente en esta traza.';
  if (!nodeRun.entered) return `Este bloque se salta porque ${pathReasonLabel(nodeRun.skipped_reason || 'sin razón')}.`;

  const participants = nodeRun.participants || {};
  if (String(nodeId) === 'visual.build_candidates') {
    const filtered = filteredRows(nodeRun, state.selectedDetId);
    const detectionRows = filtered.detectionRows.length
      ? filtered.detectionRows
      : (nodeRun.detection_rows || []);
    const totalCandidates = detectionRows.reduce((acc, row) => acc + Number(row.candidate_count || 0), 0);
    const resolvedPath = resolvedBuildCandidatesPath(nodeRun);
    if (resolvedPath === 'NO_SCORE_ROWS') {
      return 'Este nodo no abre candidatos y el caso acaba materializándose más tarde como NO_SCORE_ROWS. El desvío real del grafo aparece en Tablas finales de score, no aquí.';
    }
    if (resolvedPath === 'NO_VALID_DETECTIONS') {
      return 'Este nodo no abre candidatos y, más tarde, la traza confirma que no quedan detecciones válidas para matching. El punto de desvío real del grafo es Filtro de detecciones válidas.';
    }
    if (totalCandidates <= 0) {
      return 'Este nodo no abre candidatos visuales para el foco actual. Eso no cierra todavía la ruta final: hay que leer los nodos posteriores para ver en qué motivo downstream acaba.';
    }
  }

  const status = nodeRun?.decision?.status;
  const branch = nodeRun?.decision?.branch;
  if (branch || status) {
    const pieces = [];
    if (status) pieces.push(`estado ${status}`);
    if (branch) pieces.push(pathReasonLabel(branch));
    return pieces.join(' · ');
  }
  return summarizeNodeRun(orderedNodes().find((item) => item.id === nodeId), nodeRun);
}

function renderPathDecisionSection(nodeId, nodeRun, container) {
  const outgoingEdges = outgoingVisualEdges(nodeId);
  if (outgoingEdges.length <= 1) return;
  const semanticOptions = semanticPathOptionsForNode(nodeId, nodeRun);
  const section = createDetailSection('Camino y ramas', {
    open: false,
  });
  const branchNodeId = String(nodeId) === 'visual.build_candidates'
    ? resolvedBuildCandidatesBranchNode(nodeRun)
    : '';

  const currentCard = document.createElement('div');
  currentCard.className = 'detail-card';
  currentCard.innerHTML = `
    <div class="detail-card-head"><strong>Camino tomado en esta traza</strong></div>
    <p>${summarizeCurrentPath(nodeId, nodeRun)}</p>
    ${branchNodeId ? `<p><strong>Punto de desvío real en el grafo:</strong> ${nodeLabel(branchNodeId)}.</p>` : ''}
  `;
  section.body.appendChild(currentCard);

  if (semanticOptions.length) {
    const semanticCard = document.createElement('div');
    semanticCard.className = 'detail-card';
    const rows = semanticOptions.map((item) => `
      <tr>
        <td>${item.label}</td>
        <td>${item.condition}</td>
        <td>${item.state}</td>
        <td>${item.note}</td>
      </tr>
    `).join('');
    semanticCard.innerHTML = `
      <div class="detail-card-head"><strong>Resoluciones posibles de este bloque</strong></div>
      <p>
        Aquí no solo se ve el camino tomado, sino también las otras salidas
        semánticas que este bloque puede desencadenar y la condición típica
        que activa cada una.
      </p>
      <div class="table-wrap">
        <table class="detail-table">
          <thead>
            <tr>
              <th>salida</th>
              <th>condición</th>
              <th>estado</th>
              <th>lectura</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;
    section.body.appendChild(semanticCard);
  }

  if (outgoingEdges.length > 1) {
    const branchCard = document.createElement('div');
    branchCard.className = 'detail-card';
    const rows = outgoingEdges.map((edge) => {
      const stateInfo = edgeStateForCurrentFocus(edge);
      const edgeType = edgeIsBypass(edge)
        ? `ruta que omite ${(edge.skipNodes || []).map((skipNodeId) => nodeLabel(skipNodeId)).join(', ')}`
        : 'ruta mostrada entre nodos consecutivos';
      return `
        <tr>
          <td>${displayNodeLabel(edge.to)}</td>
          <td>${edgeType}</td>
          <td>${stateInfo.state}</td>
          <td>${stateInfo.note}</td>
        </tr>
      `;
    }).join('');
    branchCard.innerHTML = `
      <div class="detail-card-head"><strong>Salidas posibles desde este bloque</strong></div>
      <p>
        Este bloque puede abrir varios caminos. Aquí se ve cuál ha quedado
        activo en esta traza y por qué los otros no aparecen como camino
        tomado.
      </p>
      <div class="table-wrap">
        <table class="detail-table">
          <thead>
            <tr>
              <th>destino</th>
              <th>tipo</th>
              <th>estado</th>
              <th>lectura</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    `;
    section.body.appendChild(branchCard);
  }

  container.appendChild(section.section);
}

function renderIncomingDependenciesSection(nodeId, container) {
  const incomingEdges = incomingVisualEdges(nodeId)
    .filter((edge) => !String(edge.from).startsWith('synthetic.'));
  if (!incomingEdges.length) return;

  const section = createDetailSection('Bloques que alimentan este nodo', {
    badge: `${incomingEdges.length} origen${incomingEdges.length === 1 ? '' : 'es'}`,
    open: false,
  });

  const intro = document.createElement('div');
  intro.className = 'detail-card';
  intro.innerHTML = `
    <div class="detail-card-head"><strong>Dependencias directas reales</strong></div>
    <p>
      Estas etiquetas representan solo los bloques reales que entran
      directamente en este nodo dentro del DAG del visor. No incluyen entradas
      sintéticas ni ancestros lejanos.
    </p>
  `;
  section.body.appendChild(intro);

  const buttonWrap = document.createElement('div');
  buttonWrap.className = 'upstream-grid';

  incomingEdges.forEach((edge) => {
    const stateInfo = edgeStateForCurrentFocus(edge);
    const stateClass = String(stateInfo.state || 'inactivo').replace(/\s+/g, '-');
    const button = document.createElement('div');
    button.className = `upstream-button state-${stateClass}`;

    const title = document.createElement('div');
    title.className = 'upstream-button-title';
    title.textContent = displayNodeLabel(edge.from);
    button.appendChild(title);

    const meta = document.createElement('div');
    meta.className = 'upstream-button-meta';

    const status = document.createElement('span');
    status.className = `status-pill ${stateInfo.state === 'tomado' ? 'pass' : stateInfo.state === 'no tomado' ? 'fail' : 'na'}`;
    status.textContent = stateInfo.state;
    meta.appendChild(status);

    const moduleBadge = document.createElement('span');
    moduleBadge.className = 'badge upstream-module-badge';
    moduleBadge.textContent = String(edge.from).startsWith('synthetic.') ? 'Entradas' : moduleLabelForNode(edge.from);
    meta.appendChild(moduleBadge);

    button.appendChild(meta);

    const note = document.createElement('div');
    note.className = 'upstream-button-note';
    note.textContent = stateInfo.note;
    button.appendChild(note);

    if (edgeIsBypass(edge)) {
      const bypass = document.createElement('div');
      bypass.className = 'upstream-button-bypass';
      bypass.textContent = `Bypass visual: omite ${(edge.skipNodes || []).map((skipNodeId) => nodeLabel(skipNodeId)).join(', ')}`;
      button.appendChild(bypass);
    }

    buttonWrap.appendChild(button);
  });
  section.body.appendChild(buttonWrap);
  container.appendChild(section.section);
}

function summarizeNodeRun(node, nodeRun, detId = state.selectedDetId) {
  if (!nodeRun) return 'Nodo no presente en esta traza.';
  if (!nodeRun.entered) return `Puerta cerrada: ${nodeRun.skipped_reason || 'sin razón'}.`;

  const rows = filteredRows(nodeRun, detId);
  const detectionRows = rows.detectionRows;
  const candidateRows = rows.candidateRows;
  const globalRows = rows.globalRows;
  const firstCheck = gatherRelevantChecks(nodeRun, detId)[0] || null;

  if (node.id === 'outcome.finalize') {
    const row = detectionRows[0] || (nodeRun.detection_rows || [])[0];
    if (row) return `Outcome final: ${row.final_decision || '—'} · ${row.final_reason || 'sin razón'}.`;
  }
  if (node.id === 'outcome.final_ambiguity') {
    const row = detectionRows[0] || (nodeRun.detection_rows || [])[0];
    if (row) return `Diagnóstico final: ${row.status || '—'} · ${row.reason || 'sin razón'}.`;
  }
  if (node.id === 'prepare.class_partition') {
    const values = nodeRun?.values || {};
    return `La clase queda delimitada con ${pretty(values.detection_count ?? 0)} detecciones y ${pretty(values.snapshot_object_count ?? 0)} objetos de memoria comparables.`;
  }
  if (node.id === 'prepare.reliable_visual_anchors') {
    const values = nodeRun?.values || {};
    const row = detectionRows[0] || (nodeRun.detection_rows || [])[0];
    const anchorCount = Number(values.anchor_count ?? 0);
    if (row) {
      if (row.selected_as_anchor) {
        return `La detección queda como anchor visual fiable para ${objectLabelForId(row.best_object_id)}.`;
      }
      return `La detección no entra como anchor: ${reliableAnchorReasonLabel(row.reason)}.`;
    }
    if (anchorCount > 0) {
      return `Se seleccionan ${anchorCount} anchors visuales fiables para esta clase antes de activar el contexto.`;
    }
    return 'No aparece ningún anchor visual suficientemente fiable en esta clase.';
  }
  if (node.id === 'visual.report_diagnosis') {
    const row = detectionRows[0] || (nodeRun.detection_rows || [])[0];
    if (row) return `Diagnóstico visual: ${row.status || '—'} · ${row.reason || 'sin razón'}.`;
  }
  if (node.id === 'context.neighbor_sets_hypotheses') {
    const values = nodeRun?.values || {};
    const nHypotheses = Number(values.n_hypotheses ?? 0);
    if (nHypotheses > 0) {
      return `Se retienen ${nHypotheses} hipótesis tras una búsqueda acotada; no representan todo el espacio posible del frame.`;
    }
    return 'No aparecen hipótesis relacionales útiles para esta clase en este frame.';
  }
  if (node.id === 'context.sets_activation') {
    const values = nodeRun?.values || {};
    if (!Boolean(values.enabled)) {
      return 'El contexto de sets no llega a activarse para esta clase.';
    }
    return Boolean(values.global_ok)
      ? 'El contexto de sets queda activo y puede influir en Uso de contexto por reporte, Shaping por candidato y Tablas finales de score.'
      : 'El contexto de sets existe, pero queda degradado antes de influir de lleno en la asociación.';
  }
  if (node.id === 'shape.allow_for_report') {
    const row = detectionRows[0] || (nodeRun.detection_rows || [])[0];
    if (row) return row.allowed ? 'La detección puede usar contexto adicional.' : 'La detección no puede usar contexto en esta fase.';
  }
  if (node.id === 'prepare.valid_detections') {
    const row = detectionRows[0] || (nodeRun.detection_rows || [])[0];
    if (row) return row.valid ? 'La detección conserva features utilizables y sigue a la rama de matching conocido.' : `La detección sale del matching conocido: ${row.reason || 'sin razón'}.`;
  }
  if (node.id === 'shape.context_veto' && candidateRows.length) {
    const kept = candidateRows.filter((row) => Number(row.decision_keep) === 1).length;
    return `${kept} candidatos sobreviven al veto contextual para el foco actual.`;
  }
  if (node.id === 'visual.build_candidates') {
    const impact = candidateImpact(nodeRun, detId);
    const participants = nodeRun?.participants || {};
    const hasClassMemory = Boolean((participants.object_ids || []).length);
    const rows = filteredRows(nodeRun, detId);
    const detRows = rows.detectionRows.length ? rows.detectionRows : (nodeRun?.detection_rows || []);
    const totalCandidates = detRows.reduce((acc, row) => acc + Number(row.candidate_count || 0), 0);
    if (!hasClassMemory) {
      return 'No hay objetos guardados de esta clase; el matching contra memoria conocida ya no aporta y el flujo se orienta a NEW.';
    }
    if (totalCandidates <= 0) {
      return 'Este nodo no abre candidatos visuales para el foco actual; el motivo final se decide más tarde, normalmente cuando se separan las detecciones sin features comparables.';
    }
    if (impact) return `Se abren ${impact.total} candidatos visuales para esta detección.`;
  }
  if (node.id === 'shape.final_score_tables') {
    const impact = candidateImpact(nodeRun, detId);
    if (impact) return `Tras el shaping quedan ${impact.keep} candidatos útiles para score final.`;
  }
  if (node.id === 'resolve.locks') {
    const row = detectionRows[0] || (nodeRun.detection_rows || [])[0];
    if (row?.locked) return `Lock directo: det ${pretty(row.det_id)} queda cerrada con ${objectLabelForId(row.locked_object_id)}.`;
    const values = nodeRun?.values || {};
    return `Locks cierra ${pretty(values.locked_count ?? 0)} matches claros antes de Hungarian.`;
  }
  if (node.id === 'resolve.hungarian' && globalRows.length) {
    const detRow = detectionRows[0] || (nodeRun.detection_rows || [])[0];
    if (detRow) {
      return `Hungarian propone ${detRow.final_action || '—'} para la detección tras competir contra objetos y dummies.`;
    }
    return `Resolución global con ${globalRows.length} salidas registradas en Hungarian.`;
  }
  if (node.id === 'resolve.hungarian') {
    const values = nodeRun?.values || {};
    return `Hungarian resuelve ${pretty(values.participant_det_ids?.length ?? 0)} detecciones frente a ${pretty(values.object_column_count ?? 0)} objetos y ${pretty(values.dummy_column_count ?? 0)} dummies.`;
  }
  if (node.id === 'post.assignment_ambiguity') {
    const values = nodeRun?.values || {};
    return `Assignment ambiguity detecta ${pretty(values.ambiguous_component_count ?? 0)} componentes ambiguos entre ${pretty(values.component_count ?? 0)} grupos comparables.`;
  }
  if (node.id === 'post.identity_stability') {
    const row = detectionRows[0] || (nodeRun.detection_rows || [])[0];
    if (row) return `Identity stability deja la detección en estado ${row.state || '—'}.`;
  }
  if (node.id === 'post.known_set_distance_disambiguation') {
    return 'Known-set-distance intenta romper empates conocidos cuando la asignación principal todavía deja ambigüedad útil.';
  }
  if (node.id === 'post.create_competition') {
    const values = nodeRun?.values || {};
    return `Create competition encuentra ${pretty(values.competition_count ?? 0)} competiciones entre crear nuevo y conservar alternativas conocidas.`;
  }
  if (node.id === 'post.ambiguous_track_candidates') {
    const row = detectionRows[0] || (nodeRun.detection_rows || [])[0];
    if (row) return `La detección entra en la bolsa ambigua desde ${row.selected_source || '—'} con ${pretty(row.candidate_count ?? 0)} alternativas.`;
    const values = nodeRun?.values || {};
    return `Se materializan ${pretty(values.candidate_count ?? 0)} entradas ambiguas para la resolución temporal.`;
  }
  if (node.id === 'post.provisional_reconciliation') {
    const row = detectionRows[0] || (nodeRun.detection_rows || [])[0];
    if (row) return `Reconciliación temporal: ${row.decision_kind || '—'} · ${row.reason || 'sin razón'}.`;
  }
  if (node.id === 'post.final_decision_pack') {
    const row = detectionRows[0] || (nodeRun.detection_rows || [])[0];
    if (row) return `Final pack deja la detección en bucket ${row.final_bucket || '—'} · ${row.reason || 'sin razón'}.`;
    const values = nodeRun?.values || {};
    return `Final pack arbitra matches=${pretty(values.final_match_count ?? 0)}, creates=${pretty(values.final_create_count ?? 0)}, ambiguos=${pretty(values.final_ambiguous_count ?? 0)} y provisionales=${pretty(values.final_provisional_count ?? 0)}.`;
  }
  if (firstCheck) return `${firstCheck.label || firstCheck.id}: ${expression(firstCheck)}.`;
  if (detectionRows.length) return `Hay ${detectionRows.length} filas por detección relevantes en este nodo.`;
  if (candidateRows.length) return `Hay ${candidateRows.length} filas por candidato relevantes en este nodo.`;
  if (globalRows.length) return `Hay ${globalRows.length} filas globales relevantes en este nodo.`;
  return `Nodo ${nodeLabel(node.id)} ejecutado sin detalle narrativo específico.`;
}

function previewExpression(nodeRun, detId = state.selectedDetId) {
  const dropPreview = candidateDropPreview(nodeRun, detId);
  if (dropPreview) return `drop: ${dropPreview}`;
  const keepPreview = candidateKeepPreview(nodeRun, detId);
  if (keepPreview) return `keep: ${keepPreview}`;
  const impactText = candidateImpactText(nodeRun, detId);
  if (impactText) return impactText;
  const firstCheck = gatherRelevantChecks(nodeRun, detId)[0] || null;
  if (!firstCheck) return 'Sin checks visibles en este foco.';
  return expression(firstCheck);
}

function metricChipCount(nodeRun, detId = state.selectedDetId) {
  const rows = filteredRows(nodeRun, detId);
  return {
    checks: gatherRelevantChecks(nodeRun, detId).length,
    det: rows.detectionRows.length || (nodeRun?.detection_rows?.length || 0),
    cand: rows.candidateRows.length || (nodeRun?.candidate_rows?.length || 0),
    global: rows.globalRows.length || (nodeRun?.global_rows?.length || 0),
  };
}

function currentTab() {
  if (state.activeTabId === 'overview') {
    return { id: 'overview', type: 'overview', label: 'Vista general' };
  }
  return state.openTabs.find((tab) => tab.id === state.activeTabId) || { id: 'overview', type: 'overview', label: 'Vista general' };
}

function setActiveTab(tabId) {
  state.activeTabId = tabId;
  renderTabStrip();
  renderVisiblePane();
}

function openTab(tab) {
  const existing = state.openTabs.find((it) => it.id === tab.id);
  if (!existing) state.openTabs.push(tab);
  state.activeTabId = tab.id;
  renderTabStrip();
  renderVisiblePane();
}

function closeTab(tabId) {
  state.openTabs = state.openTabs.filter((tab) => tab.id !== tabId);
  if (state.activeTabId === tabId) state.activeTabId = 'overview';
  renderTabStrip();
  renderVisiblePane();
}

function openTabForNode(nodeId) {
  openTab({
    id: `node:${nodeId}`,
    type: 'node',
    label: nodeLabel(nodeId),
    nodeId,
  });
}

function openObjectsTab() {
  openTab({
    id: 'objects_snapshot',
    type: 'objects',
    label: 'Objetos',
  });
}

function renderTabStrip() {
  clearChildren(tabsStrip);
  const tabs = [{ id: 'overview', type: 'overview', label: 'Vista general' }, ...state.openTabs];
  for (const tab of tabs) {
    const wrapper = document.createElement('div');
    wrapper.className = `tab-chip ${state.activeTabId === tab.id ? 'active' : ''}`;

    const tabButton = document.createElement('button');
    tabButton.type = 'button';
    tabButton.className = 'tab-main';
    tabButton.textContent = tab.label;
    tabButton.onclick = () => setActiveTab(tab.id);
    wrapper.appendChild(tabButton);

    if (tab.id !== 'overview') {
      const closeButton = document.createElement('button');
      closeButton.type = 'button';
      closeButton.className = 'tab-close';
      closeButton.textContent = '×';
      closeButton.onclick = (event) => {
        event.stopPropagation();
        closeTab(tab.id);
      };
      wrapper.appendChild(closeButton);
    }
    tabsStrip.appendChild(wrapper);
  }
}

function renderRuns() {
  clearChildren(runSelect);
  for (const run of state.runs) {
    const option = document.createElement('option');
    option.value = run.run_id;
    option.textContent = `${run.run_id} · ${run.frame_count} frames · ${run.class_entry_count} trazas`;
    runSelect.appendChild(option);
  }
  if (state.selectedRunId) runSelect.value = state.selectedRunId;
}

function renderFrames() {
  clearChildren(frameSelect);
  const frameIds = state.manifest?.frame_ids || [];
  for (const frameId of frameIds) {
    const option = document.createElement('option');
    option.value = String(frameId);
    option.textContent = `Frame ${String(frameId).padStart(3, '0')}`;
    frameSelect.appendChild(option);
  }
  if (state.selectedFrameId != null) frameSelect.value = String(state.selectedFrameId);
}

function renderClasses() {
  clearChildren(classSelect);
  for (const entry of getFrameEntries(state.selectedFrameId)) {
    const option = document.createElement('option');
    option.value = String(entry.class_id);
    option.textContent = `${entry.class_name || 'class'}`;
    classSelect.appendChild(option);
  }
  if (state.selectedClassId != null) classSelect.value = String(state.selectedClassId);
}

function resolveSelectedClassIdForFrame(frameId, preferredClassId = state.selectedClassId) {
  const entries = getFrameEntries(frameId);
  if (!entries.length) return null;
  const preferred = Number(preferredClassId);
  if (Number.isFinite(preferred)) {
    const matchingEntry = entries.find((entry) => Number(entry.class_id) === preferred);
    if (matchingEntry) return matchingEntry.class_id;
  }
  return entries[0]?.class_id ?? null;
}

function renderHeader() {
  if (!state.trace) return;
  traceTitle.textContent = `Run ${state.trace.run_id} · Frame ${String(state.trace.frame_id).padStart(3, '0')} · ${state.trace.class_name || state.trace.class_id}`;
}

function renderFramePreview() {
  if (!state.trace) return;

  framePreviewTitle.textContent = `Frame ${String(state.trace.frame_id).padStart(3, '0')} · ${state.trace.class_name || state.trace.class_id}`;

  if (!manifestHasFramePreview(state.selectedFrameId)) {
    framePreviewImage.style.display = 'none';
    framePreviewEmpty.style.display = 'flex';
    framePreviewEmpty.textContent = 'Esta traza no incluye previews de frame. Genera una nueva ejecución si quieres disponer de ellos.';
    framePreviewImage.removeAttribute('src');
    return;
  }

  const previewUrl = `/api/run/${encodeURIComponent(state.selectedRunId)}/preview?frame_id=${state.selectedFrameId}&_=${encodeURIComponent(state.selectedRunId)}_${state.selectedFrameId}`;
  framePreviewImage.onload = () => {
    framePreviewImage.style.display = 'block';
    framePreviewEmpty.style.display = 'none';
  };
  framePreviewImage.onerror = () => {
    framePreviewImage.style.display = 'none';
    framePreviewEmpty.style.display = 'flex';
    framePreviewEmpty.textContent = 'Esta traza aún no tiene preview visual para este frame. Genera una nueva ejecución para obtenerla.';
  };
  framePreviewImage.src = previewUrl;
}

function objectSnapshotRows() {
  const rows = state.memorySnapshot?.object_rows || [];
  return rows.map((row) => ({
    label: String(row.label || row.label_raw || `id_${row.object_id}`),
    object_id: Number(row.object_id),
    hits: Number(row.hits || 0),
    last_seen: row.last_seen,
    obj_desc_count: Number(row.obj_desc_count || 0),
    bg_desc_count: Number(row.bg_desc_count || 0),
    parts_desc_count: Number(row.parts_desc_count || 0),
    state: String(row.state || ''),
  }));
}

function renderMemoryOverview() {
  clearChildren(memoryOverview);
  if (!state.trace) return;
  if (!manifestHasMemorySnapshot(state.selectedFrameId)) {
    memoryOverview.innerHTML = '<div class="empty-state">Esta traza no incluye snapshots de memoria para este frame.</div>';
    return;
  }
  const rows = objectSnapshotRows();
  if (!rows.length) {
    memoryOverview.innerHTML = '<div class="empty-state">No se encontró snapshot de memoria para este frame concreto.</div>';
    return;
  }
  const card = document.createElement('button');
  card.type = 'button';
  card.className = 'memory-card memory-card-summary';
  card.innerHTML = `
    <div class="eyebrow">objetos persistentes</div>
    <strong>${rows.length}</strong>
    <span>doble click o “Abrir detalle” para ver el snapshot completo</span>
  `;
  card.ondblclick = () => openObjectsTab();
  memoryOverview.appendChild(card);
}

function buildLayout() {
  const layout = {};
  const phases = [];
  const bandLeft = 56;
  const bandWidth = GRAPH_W - 112;
  const rowGap = 134;
  const phaseGap = 182;
  const topPadding = 54;
  const bottomPadding = 96;
  const innerPaddingX = 90;
  const topInset = 116;
  const bottomInset = 72;
  const nodeOffset = (nodeId) => NODE_LAYOUT_OFFSETS[String(nodeId)] || {};

  const phaseHeights = TREE_PHASES.map((phase) => {
    const rowHeights = phase.rows
      .filter((row) => row.length)
      .map((row) => {
        const displayNodes = row.map((nodeId) => displayNode(nodeId)).filter(Boolean);
        return Math.max(
          ...displayNodes.map((node) => {
            const dims = nodeDimensions(node);
            const offset = nodeOffset(node.id);
            return dims.h + Number(offset.y || 0);
          }),
          DEFAULT_NODE_H
        );
      });
    const rowCount = rowHeights.length || 1;
    return topInset + bottomInset + rowHeights.reduce((acc, value) => acc + value, 0) + Math.max(0, rowCount - 1) * rowGap;
  });

  const graphHeight = topPadding + bottomPadding + phaseHeights.reduce((acc, value) => acc + value, 0) + phaseGap * (TREE_PHASES.length - 1);
  let cursorY = topPadding;

  TREE_PHASES.forEach((phase, phaseIndex) => {
    const phaseHeight = phaseHeights[phaseIndex];
    const y = cursorY;
    const phaseInfo = {
      id: phase.id,
      label: phase.label,
      caption: phase.caption,
      color: phase.color,
      x: bandLeft,
      y,
      width: bandWidth,
      height: phaseHeight,
    };
    phases.push(phaseInfo);

    const innerLeft = bandLeft + innerPaddingX;
    const innerWidth = bandWidth - innerPaddingX * 2;
    let rowCursorY = y + topInset;

    phase.rows.forEach((row) => {
      const rowNodes = row.map((nodeId) => displayNode(nodeId));
      const displayNodes = rowNodes.filter(Boolean);
      if (!displayNodes.length) return;
      const dimensions = displayNodes.map((node) => nodeDimensions(node));
      const rowHeight = Math.max(
        ...displayNodes.map((node, index) => dimensions[index].h + Number(nodeOffset(node.id).y || 0)),
        DEFAULT_NODE_H
      );
      const slotCount = Math.max(1, row.length);
      const slotWidth = slotCount === 1 ? innerWidth : innerWidth / slotCount;
      let displayIndex = 0;
      rowNodes.forEach((node, columnIndex) => {
        if (!node) return;
        const dims = dimensions[displayIndex];
        const offset = nodeOffset(node.id);
        const nodeX = slotCount === 1
          ? innerLeft + (innerWidth - dims.w) / 2
          : innerLeft + columnIndex * slotWidth + (slotWidth - dims.w) / 2 + Number(offset.x || 0);
        layout[node.id] = {
          ...node,
          x: nodeX,
          y: rowCursorY + Number(offset.y || 0),
          w: dims.w,
          h: dims.h,
          phaseId: phase.id,
        };
        displayIndex += 1;
      });
      rowCursorY += rowHeight + rowGap;
    });

    cursorY = y + phaseHeight + phaseGap;
  });

  return { nodes: layout, phases, graphHeight };
}

function edgeMidpoint(from, to) {
  return {
    x: (from.x + from.w / 2 + to.x + to.w / 2) / 2,
    y: (from.y + from.h + to.y) / 2,
  };
}

function createSvg(tag, attrs = {}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, String(value));
  return el;
}

function uniqueSortedNumbers(values) {
  return [...values]
    .map((value) => Number(Number(value).toFixed(2)))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b)
    .filter((value, index, array) => index === 0 || Math.abs(value - array[index - 1]) > 0.5);
}

function expandedRoadObstacle(box, clearance = NODE_ROAD_CLEARANCE) {
  return {
    left: box.x - clearance,
    right: box.x + box.w + clearance,
    top: box.y - clearance,
    bottom: box.y + box.h + clearance,
  };
}

function pointInsideObstacle(point, obstacle) {
  return point.x > obstacle.left && point.x < obstacle.right && point.y > obstacle.top && point.y < obstacle.bottom;
}

function pointInsideAnyObstacle(point, obstacles) {
  return obstacles.some((obstacle) => pointInsideObstacle(point, obstacle));
}

function intervalsOverlapStrict(a1, a2, b1, b2) {
  const left = Math.max(Math.min(a1, a2), Math.min(b1, b2));
  const right = Math.min(Math.max(a1, a2), Math.max(b1, b2));
  return right - left > 0.5;
}

function segmentCrossesObstacle(a, b, obstacle) {
  if (Math.abs(a.x - b.x) < 0.5) {
    const x = a.x;
    if (!(x > obstacle.left && x < obstacle.right)) return false;
    return intervalsOverlapStrict(a.y, b.y, obstacle.top, obstacle.bottom);
  }
  if (Math.abs(a.y - b.y) < 0.5) {
    const y = a.y;
    if (!(y > obstacle.top && y < obstacle.bottom)) return false;
    return intervalsOverlapStrict(a.x, b.x, obstacle.left, obstacle.right);
  }
  return true;
}

function segmentCrossesAnyObstacle(a, b, obstacles) {
  return obstacles.some((obstacle) => segmentCrossesObstacle(a, b, obstacle));
}

function simplifyPolyline(points) {
  const deduped = [];
  points.forEach((point) => {
    const last = deduped[deduped.length - 1];
    if (!last || Math.abs(last.x - point.x) > 0.5 || Math.abs(last.y - point.y) > 0.5) deduped.push(point);
  });
  if (deduped.length <= 2) return deduped;
  const simplified = [deduped[0]];
  for (let index = 1; index < deduped.length - 1; index += 1) {
    const prev = simplified[simplified.length - 1];
    const current = deduped[index];
    const next = deduped[index + 1];
    const collinearX = Math.abs(prev.x - current.x) < 0.5 && Math.abs(current.x - next.x) < 0.5;
    const collinearY = Math.abs(prev.y - current.y) < 0.5 && Math.abs(current.y - next.y) < 0.5;
    if (!collinearX && !collinearY) simplified.push(current);
  }
  simplified.push(deduped[deduped.length - 1]);
  return simplified;
}

function routeStateKey(key, dir) {
  return `${key}::${dir || 'none'}`;
}

function directionalRoute(start, end, obstacles, bounds, preferredX = null) {
  const xs = uniqueSortedNumbers([
    bounds.left,
    bounds.right,
    start.x,
    end.x,
    preferredX,
    ...obstacles.flatMap((obstacle) => [obstacle.left, obstacle.right]),
  ]);
  const ys = uniqueSortedNumbers([
    bounds.top,
    bounds.bottom,
    start.y,
    end.y,
    ...obstacles.flatMap((obstacle) => [obstacle.top, obstacle.bottom]),
  ]);

  const points = new Map();
  xs.forEach((x) => {
    ys.forEach((y) => {
      const point = { x, y };
      if (!pointInsideAnyObstacle(point, obstacles)) {
        points.set(`${x}|${y}`, point);
      }
    });
  });

  const startKey = `${start.x}|${start.y}`;
  const endKey = `${end.x}|${end.y}`;
  if (!points.has(startKey)) points.set(startKey, { ...start });
  if (!points.has(endKey)) points.set(endKey, { ...end });

  const adjacency = new Map();
  const ensureAdjacency = (key) => {
    if (!adjacency.has(key)) adjacency.set(key, []);
    return adjacency.get(key);
  };

  xs.forEach((x) => {
    const column = ys
      .map((y) => `${x}|${y}`)
      .filter((key) => points.has(key));
    for (let index = 0; index < column.length - 1; index += 1) {
      const currentKey = column[index];
      const nextKey = column[index + 1];
      const current = points.get(currentKey);
      const next = points.get(nextKey);
      if (segmentCrossesAnyObstacle(current, next, obstacles)) continue;
      const distance = Math.abs(next.y - current.y);
      ensureAdjacency(currentKey).push({ key: nextKey, dir: 'v', distance });
      ensureAdjacency(nextKey).push({ key: currentKey, dir: 'v', distance });
    }
  });

  ys.forEach((y) => {
    const row = xs
      .map((x) => `${x}|${y}`)
      .filter((key) => points.has(key));
    for (let index = 0; index < row.length - 1; index += 1) {
      const currentKey = row[index];
      const nextKey = row[index + 1];
      const current = points.get(currentKey);
      const next = points.get(nextKey);
      if (segmentCrossesAnyObstacle(current, next, obstacles)) continue;
      const distance = Math.abs(next.x - current.x);
      ensureAdjacency(currentKey).push({ key: nextKey, dir: 'h', distance });
      ensureAdjacency(nextKey).push({ key: currentKey, dir: 'h', distance });
    }
  });

  const queue = [{ stateKey: routeStateKey(startKey, 'none'), nodeKey: startKey, dir: 'none', cost: 0 }];
  const costs = new Map([[routeStateKey(startKey, 'none'), 0]]);
  const previous = new Map();

  while (queue.length) {
    queue.sort((a, b) => a.cost - b.cost);
    const current = queue.shift();
    if (!current) break;
    const knownCost = costs.get(current.stateKey);
    if (knownCost == null || current.cost > knownCost + 0.001) continue;
    if (current.nodeKey === endKey) {
      const nodePath = [];
      let cursor = current.stateKey;
      while (cursor) {
        const [nodeKey] = cursor.split('::');
        nodePath.push(points.get(nodeKey));
        cursor = previous.get(cursor);
      }
      return simplifyPolyline(nodePath.reverse());
    }
    const neighbors = adjacency.get(current.nodeKey) || [];
    neighbors.forEach((neighbor) => {
      const turnPenalty = current.dir !== 'none' && current.dir !== neighbor.dir ? ROAD_TURN_PENALTY : 0;
      const nextPoint = points.get(neighbor.key);
      const preferencePenalty = preferredX != null && Math.abs(nextPoint.x - preferredX) > 0.5 ? 8 : 0;
      const nextCost = current.cost + neighbor.distance + turnPenalty + preferencePenalty;
      const nextStateKey = routeStateKey(neighbor.key, neighbor.dir);
      if (nextCost + 0.001 >= (costs.get(nextStateKey) ?? Number.POSITIVE_INFINITY)) return;
      costs.set(nextStateKey, nextCost);
      previous.set(nextStateKey, current.stateKey);
      queue.push({
        stateKey: nextStateKey,
        nodeKey: neighbor.key,
        dir: neighbor.dir,
        cost: nextCost,
      });
    });
  }

  return null;
}

function obstacleAwareEdgePolyline(from, to, fromPort, toPort, layout, graphHeight, options = {}) {
  const detOffset = Number(options.detOffset || 0);
  const clearance = NODE_ROAD_CLEARANCE + Math.abs(detOffset) + 2;
  const startX = fromPort.x + detOffset;
  const endX = toPort.x + detOffset;
  const start = {
    x: startX,
    y: from.y + from.h + clearance,
  };
  const end = {
    x: endX,
    y: to.y - clearance,
  };
  const allObstacles = Object.values(layout || {})
    .filter(Boolean)
    .map((box) => expandedRoadObstacle(box, clearance));
  const bounds = {
    left: ROAD_GRID_MARGIN,
    right: GRAPH_W - ROAD_GRID_MARGIN,
    top: ROAD_GRID_MARGIN,
    bottom: graphHeight - ROAD_GRID_MARGIN,
  };
  const routed = directionalRoute(start, end, allObstacles, bounds, options.preferredX);
  if (!routed?.length) {
    const cruiseY = start.y <= end.y
      ? start.y + Math.max(34, Math.min(end.y - start.y - 34, (end.y - start.y) * 0.5))
      : Math.min(start.y, end.y) - 34;
    return simplifyPolyline([
      { x: startX, y: from.y + from.h },
      start,
      { x: start.x, y: cruiseY },
      { x: end.x, y: cruiseY },
      end,
      { x: endX, y: to.y },
    ]);
  }
  return simplifyPolyline([
    { x: startX, y: from.y + from.h },
    start,
    ...routed,
    end,
    { x: endX, y: to.y },
  ]);
}

function orthogonalEdgePolylineToMerge(from, fromPort, mergeX, mergeY, detOffset = 0) {
  const fromBottom = from.y + from.h;
  const startX = fromPort.x + detOffset;
  const endX = mergeX + detOffset;
  // dropY is where we make the horizontal jog.
  // Always stay >= fromBottom + 28 so we never travel back up through the source node.
  const dropY = Math.max(fromBottom + 28, mergeY - 16);
  return [
    { x: startX, y: fromBottom },
    { x: startX, y: dropY },
    { x: endX, y: dropY },
    { x: endX, y: mergeY },
  ];
}

function mergeTrunkPolyline(mergeX, mergeY, toTop, detOffset = 0) {
  const x = mergeX + detOffset;
  const entryY = Math.max(mergeY + 10, toTop - 26);
  return [
    { x, y: mergeY },
    { x, y: entryY },
    { x, y: toTop },
  ];
}

function polylinePath(points) {
  if (!points.length) return '';
  return points.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`).join(' ');
}

function finalArrowSegment(points) {
  if (points.length < 2) return '';
  const a = points[points.length - 2];
  const b = points[points.length - 1];
  return `M ${a.x} ${a.y} L ${b.x} ${b.y}`;
}

function mergeArrowSegment(points, segmentLength = 12) {
  if (points.length < 3) return '';
  const mergePoint = points[points.length - 2];
  let previousPoint = null;
  for (let index = points.length - 3; index >= 0; index -= 1) {
    const candidate = points[index];
    if (Math.abs(candidate.x - mergePoint.x) > 0.5 || Math.abs(candidate.y - mergePoint.y) > 0.5) {
      previousPoint = candidate;
      break;
    }
  }
  if (!previousPoint) return '';
  const dx = mergePoint.x - previousPoint.x;
  const dy = mergePoint.y - previousPoint.y;
  const length = Math.hypot(dx, dy);
  if (length < 2) return '';
  const scale = Math.min(segmentLength, length) / length;
  const start = {
    x: mergePoint.x - dx * scale,
    y: mergePoint.y - dy * scale,
  };
  return `M ${start.x} ${start.y} L ${mergePoint.x} ${mergePoint.y}`;
}

function splitArrowSegment(points, segmentLength = 12) {
  if (points.length < 3) return '';
  const splitPoint = points[1];
  let nextPoint = null;
  for (let index = 2; index < points.length; index += 1) {
    const candidate = points[index];
    if (Math.abs(candidate.x - splitPoint.x) > 0.5 || Math.abs(candidate.y - splitPoint.y) > 0.5) {
      nextPoint = candidate;
      break;
    }
  }
  if (!nextPoint) return '';
  const dx = nextPoint.x - splitPoint.x;
  const dy = nextPoint.y - splitPoint.y;
  const length = Math.hypot(dx, dy);
  if (length < 2) return '';
  const scale = Math.min(segmentLength, length) / length;
  const end = {
    x: splitPoint.x + dx * scale,
    y: splitPoint.y + dy * scale,
  };
  return `M ${splitPoint.x} ${splitPoint.y} L ${end.x} ${end.y}`;
}

function splitHubPoint(points) {
  if (points.length < 3) return null;
  return points[1];
}

function mergeHubPoint(points) {
  if (points.length < 3) return null;
  return points[points.length - 2];
}

function renderFlowHub(x, y, kind, count) {
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(count) || count <= 1) {
    return createSvg('g', {});
  }
  const normalizedKind = kind === 'split' ? 'split' : 'merge';
  const group = createSvg('g', {
    class: `flow-hub-group flow-hub-group-${normalizedKind}`,
    transform: `translate(${x} ${y})`,
  });
  const title = createSvg('title');
  title.textContent = normalizedKind === 'split'
    ? `${count} caminos salen realmente de este bloque`
    : `${count} caminos entran realmente en este bloque`;
  group.appendChild(title);

  const hub = createSvg('circle', {
    cx: 0,
    cy: 0,
    r: normalizedKind === 'split' ? 8 : 7,
    class: `flow-hub flow-hub-${normalizedKind}`,
  });
  group.appendChild(hub);

  if (normalizedKind === 'split') {
    group.appendChild(createSvg('path', {
      d: 'M -3 -3 L 0 0 L -3 3 M 0 0 L 4 0',
      class: 'flow-hub-glyph',
    }));
  } else {
    group.appendChild(createSvg('path', {
      d: 'M -4 0 L -1 0 M 1 0 L 4 0 M 0 -4 L 0 4',
      class: 'flow-hub-glyph',
    }));
  }

  const badgeWidth = 52;
  const badgeHeight = 18;
  const badgeOffsetY = normalizedKind === 'split' ? -24 : 24;
  const badge = createSvg('g', {
    class: `flow-badge flow-badge-${normalizedKind}`,
    transform: `translate(${-badgeWidth / 2} ${badgeOffsetY - badgeHeight / 2})`,
  });
  badge.appendChild(createSvg('rect', {
    x: 0,
    y: 0,
    width: badgeWidth,
    height: badgeHeight,
    rx: 9,
    class: 'flow-badge-rect',
  }));
  const badgeText = createSvg('text', {
    x: badgeWidth / 2,
    y: 12,
    class: 'flow-badge-text',
    'text-anchor': 'middle',
  });
  badgeText.textContent = normalizedKind === 'split' ? `salen ${count}` : `entran ${count}`;
  badge.appendChild(badgeText);
  group.appendChild(badge);

  return group;
}

function clusteredPortX(box, index, count) {
  if (!box) return 0;
  if (count <= 1) return box.x + box.w / 2;
  const maxSpread = Math.min(box.w * 0.42, 96);
  const spacing = count === 2 ? Math.min(42, maxSpread) : Math.min(34, maxSpread / Math.max(1, count - 1));
  const center = box.x + box.w / 2;
  return center + (index - (count - 1) / 2) * spacing;
}

function nodePreferenceMap(layout) {
  const ordered = Object.values(layout || {}).sort((a, b) => {
    if (a.y !== b.y) return a.y - b.y;
    if (a.x !== b.x) return a.x - b.x;
    return String(a.id).localeCompare(String(b.id));
  });
  return new Map(
    ordered.map((node, index) => [String(node.id), index % 2 === 0 ? 'right' : 'left'])
  );
}

function alternatingSidePattern(count, preferredSide) {
  const pattern = [];
  let side = preferredSide === 'left' ? 'left' : 'right';
  while (pattern.length < count) {
    const take = Math.min(2, count - pattern.length);
    for (let i = 0; i < take; i += 1) pattern.push(side);
    side = side === 'right' ? 'left' : 'right';
  }
  return pattern;
}

function distributedPortXs(box, count, preferredSide) {
  if (!box) return [];
  if (count <= 1) return [box.x + box.w / 2];
  const center = box.x + box.w / 2;
  const sidePadding = 42;
  const slotStep = 46;
  const maxSideSlots = Math.max(1, Math.ceil(count / 2));
  const maxOffset = Math.max(26, box.w / 2 - sidePadding);
  const sideOffsets = Array.from({ length: maxSideSlots }, (_, index) => Math.min(maxOffset, sidePadding + index * slotStep));
  const leftXs = sideOffsets.map((offset) => center - offset);
  const rightXs = sideOffsets.map((offset) => center + offset);
  let leftIndex = 0;
  let rightIndex = 0;
  return alternatingSidePattern(count, preferredSide).map((side) => {
    if (side === 'left') {
      const x = leftXs[Math.min(leftIndex, leftXs.length - 1)];
      leftIndex += 1;
      return x;
    }
    const x = rightXs[Math.min(rightIndex, rightXs.length - 1)];
    rightIndex += 1;
    return x;
  });
}

function distributedPortXsAwayFromCenter(box, count, preferredSide) {
  if (!box || count <= 0) return [];
  const center = box.x + box.w / 2;
  const sidePadding = 42;
  const slotStep = 46;
  const maxOffset = Math.max(26, box.w / 2 - sidePadding);
  const offsets = Array.from({ length: Math.max(1, count) }, (_, index) => Math.min(maxOffset, sidePadding + index * slotStep));
  let leftIndex = 0;
  let rightIndex = 0;
  return alternatingSidePattern(count, preferredSide).map((side) => {
    if (side === 'left') {
      const offset = offsets[Math.min(leftIndex, offsets.length - 1)];
      leftIndex += 1;
      return center - offset;
    }
    const offset = offsets[Math.min(rightIndex, offsets.length - 1)];
    rightIndex += 1;
    return center + offset;
  });
}

function assignPortPositions(box, nodeEdges, preferredSide, keyFn) {
  if (!box || !nodeEdges.length) return new Map();
  const centerX = box.x + box.w / 2;
  if (nodeEdges.length === 1) return new Map([[keyFn(nodeEdges[0]), centerX]]);

  const canonicalEdges = nodeEdges.filter((edge) => !edgeIsBypass(edge));
  if (canonicalEdges.length === 1) {
    const assignments = new Map([[keyFn(canonicalEdges[0]), centerX]]);
    const remainingEdges = nodeEdges.filter((edge) => edge !== canonicalEdges[0]);
    const xs = distributedPortXsAwayFromCenter(box, remainingEdges.length, preferredSide);
    remainingEdges.forEach((edge, index) => {
      assignments.set(keyFn(edge), xs[index] ?? centerX);
    });
    return assignments;
  }

  const xs = distributedPortXs(box, nodeEdges.length, preferredSide);
  return new Map(nodeEdges.map((edge, index) => [keyFn(edge), xs[index] ?? centerX]));
}

function assignSingleCenterPortPositions(box, nodeEdges, keyFn) {
  if (!box || !nodeEdges.length) return new Map();
  const centerX = box.x + box.w / 2;
  return new Map(nodeEdges.map((edge) => [keyFn(edge), centerX]));
}

function buildEdgePortMaps(edges, layout) {
  const outgoing = new Map();
  const incoming = new Map();
  const preferenceByNodeId = nodePreferenceMap(layout);

  edges.forEach((edge) => {
    const fromKey = String(edge.from);
    const toKey = String(edge.to);
    outgoing.set(fromKey, [...(outgoing.get(fromKey) || []), edge]);
    incoming.set(toKey, [...(incoming.get(toKey) || []), edge]);
  });

  const fromPortMap = new Map();
  const toPortMap = new Map();

  for (const [nodeId, nodeEdges] of outgoing.entries()) {
    const box = layout[nodeId];
    if (!box) continue;
    const assignments = assignSingleCenterPortPositions(box, nodeEdges, (edge) => `${edge.from}->${edge.to}`);
    nodeEdges.forEach((edge, index) => {
      const x = assignments.get(`${edge.from}->${edge.to}`) ?? clusteredPortX(box, index, nodeEdges.length);
      fromPortMap.set(`${edge.from}->${edge.to}`, { x });
    });
  }

  for (const [nodeId, nodeEdges] of incoming.entries()) {
    const box = layout[nodeId];
    if (!box) continue;
    const assignments = assignSingleCenterPortPositions(box, nodeEdges, (edge) => `${edge.from}->${edge.to}`);
    nodeEdges.forEach((edge, index) => {
      const x = assignments.get(`${edge.from}->${edge.to}`) ?? clusteredPortX(box, index, nodeEdges.length);
      toPortMap.set(`${edge.from}->${edge.to}`, { x });
    });
  }

  return { fromPortMap, toPortMap };
}

function buildIncomingGroupMap(edges) {
  const incoming = new Map();
  edges.forEach((edge) => {
    const key = String(edge.to);
    incoming.set(key, [...(incoming.get(key) || []), edge]);
  });
  return incoming;
}

function buildOutgoingGroupMap(edges) {
  const outgoing = new Map();
  edges.forEach((edge) => {
    const key = String(edge.from);
    outgoing.set(key, [...(outgoing.get(key) || []), edge]);
  });
  return outgoing;
}

function applyTransform() {
  graphRoot.setAttribute('transform', `translate(${state.transform.x} ${state.transform.y}) scale(${state.transform.scale})`);
}

function fitGraphToViewport() {
  state.transform.scale = 1;
  state.transform.x = 0;
  state.transform.y = 0;
  applyTransform();
}

function truncate(text, max = 58) {
  const str = String(text || '');
  return str.length <= max ? str : `${str.slice(0, max - 1)}…`;
}

function renderStageRegions(layoutInfo) {
  for (const phase of layoutInfo.phases) {
    const group = createSvg('g', { class: 'stage-region' });
    group.appendChild(createSvg('rect', {
      x: phase.x,
      y: phase.y,
      width: phase.width,
      height: phase.height,
      class: 'phase-band',
      stroke: phase.color,
    }));

    const label = createSvg('text', { x: phase.x + 18, y: phase.y + 30, class: 'phase-label stage-title' });
    label.textContent = phase.label;
    group.appendChild(label);

    const caption = createSvg('text', { x: phase.x + 18, y: phase.y + 52, class: 'phase-caption stage-caption' });
    caption.textContent = phase.caption;
    group.appendChild(caption);

    graphRoot.appendChild(group);
  }
}

function lineWrap(text, maxChars = 44, maxLines = 3) {
  const words = String(text || '').split(/\s+/).filter(Boolean);
  if (!words.length) return ['—'];
  const lines = [];
  let current = '';
  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length <= maxChars) {
      current = next;
      continue;
    }
    lines.push(current || word);
    current = current && current !== word ? word : '';
    if (lines.length === maxLines - 1) break;
  }
  if (current && lines.length < maxLines) lines.push(current);
  const remaining = words.join(' ');
  const used = lines.join(' ');
  if (remaining.length > used.length && lines.length) {
    lines[lines.length - 1] = truncate(lines[lines.length - 1], maxChars);
  }
  return lines.slice(0, maxLines);
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function nodeDescriptionMaxChars(width) {
  return clamp(Math.floor((width - 48) / 7.2), 42, 60);
}

function estimateNodeWidth(node) {
  const title = String(node?.label || nodeLabel(node?.id) || '');
  const subtitle = String(node?.moduleLabel || moduleLabelForNode(node?.id) || '');
  const description = String(node?.description || generalDescription(node?.id) || '');
  const listItems = Array.isArray(node?.listItems) ? node.listItems : [];
  const longestListLabel = listItems.reduce((max, item) => Math.max(max, String(item?.label || '').length), 0);
  const longestSample = Math.max(title.length, subtitle.length, longestListLabel, Math.min(description.length, 72));
  const estimated = DEFAULT_NODE_W + Math.max(0, longestSample - 28) * 3.6;
  return clamp(Math.round(estimated), DEFAULT_NODE_W, MAX_NODE_W);
}

function nodeDimensions(node) {
  const matrixMetrics = decisionMatrixMetrics(node?.id);
  const originInfo = graphCardOriginInfo(node?.id);
  const width = Math.max(
    estimateNodeWidth(node),
    matrixMetrics ? matrixMetrics.width + 42 : 0
  );
  const listItems = Array.isArray(node?.listItems) ? node.listItems : [];
  const descriptionMaxChars = nodeDescriptionMaxChars(width);
  const descriptionMaxLines = listItems.length ? 3 : 4;
  const descriptionLines = lineWrap(
    node.description || generalDescription(node.id),
    descriptionMaxChars,
    descriptionMaxLines
  ).length;
  const originHeight = originInfo.items.length ? 18 + estimateGraphCardOriginRows(width, originInfo.items) * 24 : 0;
  const listCount = listItems.length;
  const matrixHeight = matrixMetrics ? MATRIX_TOP_GAP + matrixMetrics.height : 0;
  const height = 102 + originHeight + descriptionLines * 19 + listCount * 18 + (listCount ? 20 : 12) + matrixHeight;
  return {
    w: Math.min(width, MAX_NODE_W),
    h: Math.max(DEFAULT_NODE_H, height),
  };
}

function renderMultilineText(group, x, y, className, lines, lineHeight = 16) {
  const text = createSvg('text', { x, y, class: className });
  lines.forEach((line, index) => {
    const tspan = createSvg('tspan', { x, y: y + index * lineHeight });
    tspan.textContent = line;
    text.appendChild(tspan);
  });
  group.appendChild(text);
}

function renderNodeList(group, x, y, items, overflowText = null) {
  items.forEach((item, index) => {
    const rowY = y + index * 17;
    if (item.color) {
      group.appendChild(createSvg('rect', {
        x,
        y: rowY - 9,
        width: 10,
        height: 10,
        rx: 2,
        fill: item.color,
        class: 'det-legend-swatch',
      }));
    }
    const label = createSvg('text', { x: x + (item.color ? 16 : 0), y: rowY, class: 'node-list-label' });
    label.textContent = item.label;
    group.appendChild(label);
  });
  if (overflowText) {
    const more = createSvg('text', { x, y: y + items.length * 17, class: 'node-list-more' });
    more.textContent = overflowText;
    group.appendChild(more);
  }
}

function graphCardOriginInfo(nodeId) {
  const immediateRealEdges = incomingVisualEdges(nodeId)
    .filter((edge) => !String(edge.from).startsWith('synthetic.'));
  if (!immediateRealEdges.length) return { caption: '', items: [] };
  return {
    caption: 'depende de',
    items: immediateRealEdges.map((edge) => ({
      sourceNodeId: String(edge.from),
      label: displayNodeLabel(edge.from),
      state: edgeStateForCurrentFocus(edge).state,
    })),
  };
}

function estimateGraphCardOriginRows(width, items) {
  if (!items.length) return 0;
  const available = Math.max(180, width - 36);
  let rows = 1;
  let used = 0;
  items.forEach((item) => {
    const pillWidth = Math.min(Math.max(84, 34 + String(item.label || '').length * 7), Math.max(110, available));
    if (used > 0 && used + pillWidth > available) {
      rows += 1;
      used = 0;
    }
    used += pillWidth + 8;
  });
  return rows;
}

function renderNodeOriginChips(group, nodeId, box) {
  const info = graphCardOriginInfo(nodeId);
  if (!info.items.length) return 0;

  const startX = 18;
  const captionY = 78;
  const rowHeight = 24;
  const pillHeight = 18;
  const available = Math.max(180, box.w - 36);

  const caption = createSvg('text', {
    x: startX,
    y: captionY,
    class: 'node-origin-caption',
  });
  caption.textContent = info.caption;
  group.appendChild(caption);

  let cursorX = startX;
  let rowIndex = 0;
  info.items.forEach((item) => {
    const pillWidth = Math.min(Math.max(84, 34 + String(item.label || '').length * 7), Math.max(110, available));
    if (cursorX > startX && cursorX - startX + pillWidth > available) {
      rowIndex += 1;
      cursorX = startX;
    }
    const pillY = captionY + 10 + rowIndex * rowHeight;
    const pillGroup = createSvg('g', {
      class: `node-origin-chip state-${String(item.state || 'inactivo').replace(/\s+/g, '-')}`,
      transform: `translate(${cursorX} ${pillY})`,
    });

    pillGroup.appendChild(createSvg('rect', {
      x: 0,
      y: 0,
      width: pillWidth,
      height: pillHeight,
      rx: 9,
      class: 'node-origin-chip-rect',
    }));
    pillGroup.appendChild(createSvg('circle', {
      cx: 10,
      cy: 9,
      r: 3,
      class: 'node-origin-chip-dot',
    }));
    const text = createSvg('text', {
      x: 18,
      y: 12.5,
      class: 'node-origin-chip-text',
    });
    text.textContent = truncate(item.label, Math.max(10, Math.floor((pillWidth - 24) / 7)));
    pillGroup.appendChild(text);
    const title = createSvg('title');
    title.textContent = `${item.label} · ${item.state}`;
    pillGroup.appendChild(title);
    group.appendChild(pillGroup);
    cursorX += pillWidth + 8;
  });

  return 18 + estimateGraphCardOriginRows(box.w, info.items) * rowHeight;
}

function nodeHasOpenTabControl(node) {
  return Boolean(node && String(node.phaseId || '') !== 'inputs');
}

function renderNodeOpenTabButton(group, node, box) {
  if (!nodeHasOpenTabControl(node)) return;
  const buttonSize = 28;
  const buttonX = box.w - buttonSize - 14;
  const buttonY = 18;
  const buttonGroup = createSvg('g', { class: 'node-open-tab-button' });
  const buttonRect = createSvg('rect', {
    x: buttonX,
    y: buttonY,
    width: buttonSize,
    height: buttonSize,
    rx: 8,
    class: 'node-open-tab-button-rect',
  });
  const buttonText = createSvg('text', {
    x: buttonX + buttonSize / 2,
    y: buttonY + 19,
    class: 'node-open-tab-button-text',
    'text-anchor': 'middle',
  });
  buttonText.textContent = '+';
  const handleOpen = (event) => {
    event.stopPropagation();
    openTabForNode(node.id);
  };
  buttonRect.addEventListener('click', handleOpen);
  buttonText.addEventListener('click', handleOpen);
  const title = createSvg('title');
  title.textContent = 'Abrir detalle';
  buttonGroup.appendChild(title);
  buttonGroup.appendChild(buttonRect);
  buttonGroup.appendChild(buttonText);
  group.appendChild(buttonGroup);
}

function renderDecisionMatrix(group, x, y, nodeId, matrix) {
  const metrics = decisionMatrixMetrics(nodeId);
  if (!metrics || !matrix) return 0;
  const selectedDetId = state.selectedDetId == null ? null : Number(state.selectedDetId);
  const caption = createSvg('text', {
    x: x + metrics.width / 2,
    y: y + 10,
    class: 'node-matrix-caption',
    'text-anchor': 'middle',
  });
  caption.textContent = 'estado det × obj';
  group.appendChild(caption);

  const topY = y + metrics.titleHeight;
  metrics.objIds.forEach((objectId, columnIndex) => {
    const cellX = x + metrics.labelColWidth + columnIndex * metrics.step;
    const labelX = cellX + 4;
    const labelY = topY + metrics.headerHeight - 10;
    const label = createSvg('text', {
      x: labelX,
      y: labelY,
      class: 'node-matrix-label',
      'text-anchor': 'start',
      transform: `rotate(-20 ${labelX} ${labelY})`,
    });
    label.textContent = objectLabelForId(objectId);
    group.appendChild(label);
  });

  metrics.detIds.forEach((detId, rowIndex) => {
    const rowY = topY + metrics.headerHeight + rowIndex * metrics.step;
    const detLabel = createSvg('text', {
      x,
      y: rowY + metrics.cellSize * 0.78,
      class: `node-matrix-label ${selectedDetId != null && detId === selectedDetId ? 'selected' : ''}`,
    });
    detLabel.textContent = `d${detId}`;
    if (selectedDetId != null && detId === selectedDetId) detLabel.setAttribute('fill', detColor(detId));
    group.appendChild(detLabel);

    metrics.objIds.forEach((objectId, columnIndex) => {
      const stateName = matrixState(matrix, detId, objectId);
      const rect = createSvg('rect', {
        x: x + metrics.labelColWidth + columnIndex * metrics.step,
        y: rowY,
        width: metrics.cellSize,
        height: metrics.cellSize,
        rx: 2,
        class: 'node-matrix-cell',
        fill: MATRIX_STATE_COLORS[stateName] || MATRIX_STATE_COLORS.none,
      });
      const title = createSvg('title');
      title.textContent = `det ${detId} · ${objectLabelForId(objectId)} · ${MATRIX_STATE_LABELS[stateName] || stateName}`;
      rect.appendChild(title);
      group.appendChild(rect);
    });
  });

  return metrics.height;
}

function targetEntryOffset(from, to, detIndex, detCount) {
  const spread = ROAD_DET_SPREAD;
  const base = detIndex - (detCount - 1) / 2;
  if (to.y >= from.y + from.h) return base * spread;
  if (to.x >= from.x + from.w) return base * spread;
  return base * spread;
}

function bypassEdgeKey(edge) {
  return `${edge.from}->${edge.to}::${(edge.skipNodes || []).join('|')}`;
}

function bypassPreferredX(edge, from, to, layout, bypassLaneMap, detOffset = 0) {
  const laneInfo = bypassLaneMap.get(bypassEdgeKey(edge)) || { laneIndex: 0, side: 'right', sideIndex: 0 };
  const side = String(laneInfo.side || 'right');
  const sideIndex = Number(laneInfo.sideIndex || 0);
  const laneXOffset = 72 + sideIndex * 68;
  const routeNodeIds = [String(edge.from), String(edge.to), ...((edge.skipNodes || []).map((nodeId) => String(nodeId)))];
  const routeBoxes = routeNodeIds
    .map((nodeId) => layout?.[nodeId])
    .filter(Boolean);
  const boxes = routeBoxes.length ? routeBoxes : [from, to];
  const minLeft = Math.min(...boxes.map((box) => box.x));
  const maxRight = Math.max(...boxes.map((box) => box.x + box.w));
  const rawX = side === 'left'
    ? minLeft - laneXOffset + detOffset
    : maxRight + laneXOffset + detOffset;
  return Math.max(ROAD_GRID_MARGIN, Math.min(GRAPH_W - ROAD_GRID_MARGIN, rawX));
}

function bypassEdgePolyline(edge, from, to, fromPort, toPort, layout, bypassLaneMap, graphHeight, detOffset = 0) {
  return obstacleAwareEdgePolyline(from, to, fromPort, toPort, layout, graphHeight, {
    detOffset,
    preferredX: bypassPreferredX(edge, from, to, layout, bypassLaneMap, detOffset),
  });
}

function renderGraph() {
  clearChildren(graphRoot);
  currentNodeMatrixInfoMap = new Map();
  if (!state.schema || !state.trace) return;

  currentNodeMatrixInfoMap = buildNodeDecisionMatrixMap();
  const layoutInfo = buildLayout();
  currentGraphLayoutInfo = layoutInfo;
  const layout = layoutInfo.nodes;
  graphSvg.setAttribute('viewBox', `0 0 ${GRAPH_W} ${layoutInfo.graphHeight}`);
  graphSvg.setAttribute('width', String(GRAPH_W));
  graphSvg.setAttribute('height', String(layoutInfo.graphHeight));
  graphSvg.style.width = `${GRAPH_W}px`;
  graphSvg.style.height = `${layoutInfo.graphHeight}px`;
  renderStageRegions(layoutInfo);

  const allEdges = graphEdges();
  const bypassLaneMap = buildBypassLaneMap(allEdges, layout);
  const { fromPortMap, toPortMap } = buildEdgePortMaps(allEdges, layout);
  const incomingGroupMap = buildIncomingGroupMap(allEdges);
  const outgoingGroupMap = buildOutgoingGroupMap(allEdges);
  const renderedSplitHubs = new Set();
  const renderedMergeHubs = new Set();
  for (const edge of allEdges) {
    const from = layout[edge.from];
    const to = layout[edge.to];
    if (!from || !to) continue;
    const edgeKey = `${edge.from}->${edge.to}`;
    const fromPort = fromPortMap.get(edgeKey) || { x: from.x + from.w / 2 };
    const toPort = toPortMap.get(edgeKey) || { x: to.x + to.w / 2 };
    const hasMergeIntoTarget = (incomingGroupMap.get(String(edge.to)) || []).length > 1;
    const hasSplitFromSource = (outgoingGroupMap.get(String(edge.from)) || []).length > 1;

    const basePoints = edgeIsBypass(edge)
      ? bypassEdgePolyline(edge, from, to, fromPort, toPort, layout, bypassLaneMap, layoutInfo.graphHeight, 0)
      : obstacleAwareEdgePolyline(from, to, fromPort, toPort, layout, layoutInfo.graphHeight, { detOffset: 0 });
    const path = createSvg('path', {
      d: polylinePath(basePoints),
      class: `edge-road ${edgeIsBypass(edge) ? 'edge-road-bypass' : ''}`,
      'marker-end': 'url(#edgeArrow)',
    });
    graphRoot.appendChild(path);
    if (hasSplitFromSource) {
      const splitRoad = createSvg('path', {
        d: splitArrowSegment(basePoints),
        class: `edge-road ${edgeIsBypass(edge) ? 'edge-road-bypass' : ''}`,
        'marker-end': 'url(#edgeArrowTiny)',
      });
      graphRoot.appendChild(splitRoad);
      const splitNodeKey = String(edge.from);
      if (!renderedSplitHubs.has(splitNodeKey)) {
        const splitPoint = splitHubPoint(basePoints);
        const splitCount = (outgoingGroupMap.get(splitNodeKey) || []).length;
        if (splitPoint && splitCount > 1) {
          graphRoot.appendChild(renderFlowHub(splitPoint.x, splitPoint.y, 'split', splitCount));
          renderedSplitHubs.add(splitNodeKey);
        }
      }
    }
    if (hasMergeIntoTarget) {
      const mergeRoad = createSvg('path', {
        d: mergeArrowSegment(basePoints),
        class: `edge-road ${edgeIsBypass(edge) ? 'edge-road-bypass' : ''}`,
        'marker-end': 'url(#edgeArrowTiny)',
      });
      graphRoot.appendChild(mergeRoad);
      const mergeNodeKey = String(edge.to);
      if (!renderedMergeHubs.has(mergeNodeKey)) {
        const mergePoint = mergeHubPoint(basePoints);
        const mergeCount = (incomingGroupMap.get(mergeNodeKey) || []).length;
        if (mergePoint && mergeCount > 1) {
          graphRoot.appendChild(renderFlowHub(mergePoint.x, mergePoint.y, 'merge', mergeCount));
          renderedMergeHubs.add(mergeNodeKey);
        }
      }
    }

    const detIds = (state.trace?.det_ids || []).map((value) => Number(value));
    const activeDetIds = detIds.filter((detId) => edgeFollowedByDet(edge, detId));
    activeDetIds.forEach((detId, index) => {
      const offset = targetEntryOffset(from, to, index, activeDetIds.length);
      const detPoints = edgeIsBypass(edge)
        ? bypassEdgePolyline(edge, from, to, fromPort, toPort, layout, bypassLaneMap, layoutInfo.graphHeight, offset)
        : obstacleAwareEdgePolyline(from, to, fromPort, toPort, layout, layoutInfo.graphHeight, { detOffset: offset });
      const detPath = createSvg('path', {
        d: polylinePath(detPoints),
        class: `edge-det ${edgeIsBypass(edge) ? 'edge-det-bypass' : ''}`,
        stroke: detColor(detId),
        'marker-end': 'url(#edgeArrowColor)',
      });
      graphRoot.appendChild(detPath);
      if (hasSplitFromSource) {
        const splitDet = createSvg('path', {
          d: splitArrowSegment(detPoints),
          class: `edge-det ${edgeIsBypass(edge) ? 'edge-det-bypass' : ''}`,
          stroke: detColor(detId),
          'marker-end': 'url(#edgeArrowColorTiny)',
        });
        graphRoot.appendChild(splitDet);
      }
      if (hasMergeIntoTarget) {
        const mergeDet = createSvg('path', {
          d: mergeArrowSegment(detPoints),
          class: `edge-det ${edgeIsBypass(edge) ? 'edge-det-bypass' : ''}`,
          stroke: detColor(detId),
          'marker-end': 'url(#edgeArrowColorTiny)',
        });
        graphRoot.appendChild(mergeDet);
      }
    });
  }

  const displayNodes = Object.values(layout);
  for (const node of displayNodes) {
    const box = layout[node.id];
    const nodeRun = getNodeRun(node.id);
    const group = createSvg('g', {
      transform: `translate(${box.x} ${box.y})`,
      'data-node-id': String(node.id),
    });
    const topColor = node.color || doorStateColor(nodeDoorState(nodeRun));

    const rect = createSvg('rect', {
      width: box.w,
      height: box.h,
      class: `node-card ${nodeRun?.entered || node.id.startsWith('synthetic.') ? 'entered' : 'skipped'} ${state.selectedNodeId === node.id ? 'selected' : ''}`,
    });
    rect.addEventListener('click', () => {
      state.selectedNodeId = node.id;
      renderGraph();
    });
    group.appendChild(rect);

    group.appendChild(createSvg('rect', {
      x: 0,
      y: 0,
      width: box.w,
      height: 10,
      class: 'node-topbar',
      fill: topColor,
    }));

    const title = createSvg('text', { x: 18, y: 36, class: 'node-title' });
    title.textContent = node.label || nodeLabel(node.id);
    group.appendChild(title);

    const subtitle = createSvg('text', { x: 18, y: 56, class: 'node-subtitle' });
    subtitle.textContent = node.moduleLabel || moduleLabelForNode(node.id);
    group.appendChild(subtitle);

    renderNodeOpenTabButton(group, node, box);
    const originHeight = renderNodeOriginChips(group, node.id, box);

    const listItems = node.listItems || [];
    const descriptionLines = lineWrap(
      node.description || generalDescription(node.id),
      nodeDescriptionMaxChars(box.w),
      listItems.length ? 3 : 4
    );
    renderMultilineText(
      group,
      18,
      84 + originHeight,
      'node-summary',
      descriptionLines,
      17
    );

    if (listItems.length) {
      const listY = 84 + originHeight + descriptionLines.length * 17 + 8;
      renderNodeList(group, 18, listY, listItems, null);
    }

    const matrixInfo = currentNodeMatrixInfoMap.get(String(node.id));
    if (!node.id.startsWith('synthetic.') && matrixInfo?.visible) {
      const matrixY = 84 + originHeight + descriptionLines.length * 17 + (listItems.length ? (8 + listItems.length * 17 + 14) : 14);
      const matrixMetrics = decisionMatrixMetrics(node.id);
      const matrixX = matrixMetrics ? Math.max(18, Math.round((box.w - matrixMetrics.width) / 2)) : 18;
      renderDecisionMatrix(group, matrixX, matrixY, node.id, matrixInfo.matrix);
    }

    graphRoot.appendChild(group);
  }

  applyTransform();
}

function renderRowsAsCards(title, rows, kind, container) {
  const wrap = document.createElement('div');
  wrap.className = 'detail-card';
  wrap.innerHTML = `<div class="detail-card-head"><strong>${title}</strong><span class="badge">${rows.length}</span></div>`;

  for (const row of rows) {
    const entry = document.createElement('div');
    entry.className = 'detail-subcard';
    const summary = kind === 'candidate'
      ? `det ${row.det_id} -> ${row.object_id == null ? '—' : objectLabelForId(row.object_id)} · ${isDroppedCandidate(row) ? 'eliminado' : 'activo'}`
      : JSON.stringify(row);
    const kvs = Object.entries(row)
      .filter(([key]) => key !== 'checks')
      .slice(0, 8)
      .map(([key, value]) => `<span>${key}: ${formatValueForDisplay(key, value)}</span>`)
      .join('');
    entry.innerHTML = `
      <div class="detail-card-head"><strong>${summary}</strong></div>
      <div class="inline-kv">${kvs || '<span>Sin highlights</span>'}</div>
      <details class="raw-details">
        <summary>Ver fila raw</summary>
        <pre class="code-block">${JSON.stringify(row, null, 2)}</pre>
      </details>
    `;
    wrap.appendChild(entry);
  }
  container.appendChild(wrap);
}

function normalizeCheckLogic(check) {
  const raw = check?.logic_op ?? check?.logic ?? check?.group_op ?? check?.operator ?? '';
  const value = String(raw || '').trim();
  return value ? value.toUpperCase() : '';
}

function groupChecksByAssociation(checks) {
  const groups = [];
  const byKey = new Map();
  for (const check of checks) {
    const key = String(check.__groupKey || 'node:global');
    let group = byKey.get(key);
    if (!group) {
      group = {
        key,
        label: check.__groupLabel || 'Checks',
        associationLabel: check.__associationLabel || 'nodo completo',
        checks: [],
      };
      byKey.set(key, group);
      groups.push(group);
    }
    group.checks.push(check);
  }
  groups.forEach((group) => {
    group.checks.sort((left, right) => {
      const leftOrder = Number(left.__logicOrder ?? left.__groupIndex ?? left.__sourceOrder ?? 0);
      const rightOrder = Number(right.__logicOrder ?? right.__groupIndex ?? right.__sourceOrder ?? 0);
      if (leftOrder !== rightOrder) return leftOrder - rightOrder;
      return Number(left.__sourceOrder ?? 0) - Number(right.__sourceOrder ?? 0);
    });
  });
  return groups;
}

function checkGroupCategory(group) {
  const sourceKinds = [...new Set((group?.checks || []).map((check) => String(check.__sourceKind || 'node')))];
  if (sourceKinds.every((kind) => kind === 'node')) return 'block';
  return 'logic';
}

function checkGroupCategoryLabel(group) {
  return checkGroupCategory(group) === 'block'
    ? 'gate o check propio del bloque'
    : 'lógica interna del bloque';
}

function sortCheckGroups(groups) {
  return [...(groups || [])].sort((left, right) => {
    const leftCategory = checkGroupCategory(left);
    const rightCategory = checkGroupCategory(right);
    if (leftCategory !== rightCategory) {
      return leftCategory === 'logic' ? -1 : 1;
    }
    const leftDet = Math.min(...(left.checks || []).map((check) => Number(check.__detId)).filter((value) => Number.isFinite(value)));
    const rightDet = Math.min(...(right.checks || []).map((check) => Number(check.__detId)).filter((value) => Number.isFinite(value)));
    const leftHasDet = Number.isFinite(leftDet);
    const rightHasDet = Number.isFinite(rightDet);
    if (leftHasDet !== rightHasDet) return leftHasDet ? -1 : 1;
    if (leftHasDet && rightHasDet && leftDet !== rightDet) return leftDet - rightDet;
    const leftObject = Math.min(...(left.checks || []).map((check) => Number(check.__objectId)).filter((value) => Number.isFinite(value)));
    const rightObject = Math.min(...(right.checks || []).map((check) => Number(check.__objectId)).filter((value) => Number.isFinite(value)));
    const leftHasObject = Number.isFinite(leftObject);
    const rightHasObject = Number.isFinite(rightObject);
    if (leftHasObject !== rightHasObject) return leftHasObject ? 1 : -1;
    if (leftHasObject && rightHasObject && leftObject !== rightObject) return leftObject - rightObject;
    return String(left.label || '').localeCompare(String(right.label || ''));
  });
}

function checkGroupLogicLabel(checks) {
  const logicOps = [...new Set(
    (checks || [])
      .map((check) => normalizeCheckLogic(check))
      .filter((value) => value)
  )];
  if (logicOps.length === 1) return logicOps[0];
  if (logicOps.length > 1) return logicOps.join(' / ');
  if ((checks || []).length > 1) return 'SECUENCIA';
  return 'INDIVIDUAL';
}

function checkGroupLogicDescription(checks) {
  const logicOps = [...new Set(
    (checks || [])
      .map((check) => normalizeCheckLogic(check))
      .filter((value) => value)
  )];
  if (logicOps.length === 1) return `lógica explícita: ${logicOps[0]}`;
  if (logicOps.length > 1) return `lógicas explícitas mixtas: ${logicOps.join(' / ')}`;
  if ((checks || []).length > 1) return 'sin AND/OR explícito en la traza; se respeta el orden serializado';
  return 'check individual sin composición adicional';
}

function renderCheckCards(checks, container) {
  if (!checks.length) {
    container.innerHTML = '<div class="empty-state">No hay checks visibles para esta vista.</div>';
    return;
  }

  const groupedChecks = sortCheckGroups(groupChecksByAssociation(checks));
  const groupedByCategory = {
    logic: groupedChecks.filter((group) => checkGroupCategory(group) === 'logic'),
    block: groupedChecks.filter((group) => checkGroupCategory(group) === 'block'),
  };

  const renderGroupCollection = (title, groups) => {
    if (!groups.length) return;
    const section = document.createElement('div');
    section.className = 'check-section';
    section.innerHTML = `<div class="check-section-title">${title}</div>`;

    for (const group of groups) {
      const wrap = document.createElement('div');
      wrap.className = 'check-group';
      const orderTrail = group.checks.map((check) => check.__groupIndex).join(' -> ');
      const logicLabel = checkGroupLogicLabel(group.checks);
      const logicDescription = checkGroupLogicDescription(group.checks);
      wrap.innerHTML = `
        <div class="detail-card-head">
          <strong>${group.label}</strong>
          <span class="badge">${group.checks.length} checks</span>
        </div>
        <div class="check-group-meta">
          <span>tipo: ${checkGroupCategoryLabel(group)}</span>
          <span>asociado a: ${group.associationLabel}</span>
          <span>orden: ${orderTrail}</span>
          <span>lógica: ${logicLabel}</span>
          <span>${logicDescription}</span>
        </div>
      `;

      const grid = document.createElement('div');
      grid.className = 'detail-grid check-grid';

      for (const check of group.checks) {
        const statusClass = check.passed === true ? 'pass' : (check.passed === false ? 'fail' : 'soft');
        const card = document.createElement('div');
        card.className = `check-card ${statusClass}`;
        card.innerHTML = `
          <div class="detail-card-head">
            <strong>${check.label || check.id}</strong>
            <span class="status-pill ${statusClass}">${check.passed === true ? 'PASS' : (check.passed === false ? 'FAIL' : 'N/A')}</span>
          </div>
          <div class="check-meta-row">
            <span class="node-small">tipo: ${checkGroupCategoryLabel(group)}</span>
            <span class="node-small">asociado a: ${check.__associationLabel || 'nodo completo'}</span>
            <span class="node-small">orden: ${check.__groupIndex || 1}/${check.__groupSize || 1}</span>
          </div>
          <div class="node-expression">${expression(check)}</div>
          <div class="node-small">razón: ${check.reason || '—'}</div>
          <div class="node-small">efecto: ${check.effect || '—'}</div>
        `;
        grid.appendChild(card);
      }

      wrap.appendChild(grid);
      section.appendChild(wrap);
    }

    container.appendChild(section);
  };

  renderGroupCollection('Lógica interna', groupedByCategory.logic);
  renderGroupCollection('Checks del bloque', groupedByCategory.block);
}

function renderValueCards(values, container) {
  const entries = Object.entries(values || {});
  if (!entries.length) {
    container.innerHTML = '<div class="empty-state">No hay values visibles en esta vista.</div>';
    return;
  }
  for (const [key, value] of entries) {
    const card = document.createElement('div');
    card.className = 'detail-card';
    card.innerHTML = `<div class="detail-card-head"><strong>${key}</strong></div><code>${formatValueForDisplay(key, value)}</code>`;
    container.appendChild(card);
  }
}

function createDetailSection(title, { badge = null, open = false } = {}) {
  const section = document.createElement('details');
  section.className = 'detail-section';
  if (open) section.open = true;

  const summary = document.createElement('summary');
  summary.className = 'detail-section-summary';
  const titleWrap = document.createElement('div');
  titleWrap.className = 'detail-section-title';
  titleWrap.textContent = title;
  summary.appendChild(titleWrap);
  if (badge != null) {
    const badgeEl = document.createElement('span');
    badgeEl.className = 'badge';
    badgeEl.textContent = String(badge);
    summary.appendChild(badgeEl);
  }

  const body = document.createElement('div');
  body.className = 'detail-section-body';
  section.appendChild(summary);
  section.appendChild(body);
  return { section, body };
}

function renderFactRows(container, rows) {
  const filtered = (rows || []).filter((row) => row && row.value != null && String(row.value) !== '');
  if (!filtered.length) {
    container.innerHTML = '<div class="empty-state">Sin información visible en esta sección.</div>';
    return;
  }
  const list = document.createElement('div');
  list.className = 'detail-fact-list';
  filtered.forEach((row) => {
    const entry = document.createElement('div');
    entry.className = 'detail-fact-row';
    entry.innerHTML = `<strong>${row.label}</strong><span>${row.value}</span>`;
    list.appendChild(entry);
  });
  container.appendChild(list);
}

function sortContextObjectRows(rows) {
  return [...(rows || [])].sort((left, right) => {
    const leftSupported = Number(Boolean(left?.supported_hit));
    const rightSupported = Number(Boolean(right?.supported_hit));
    if (leftSupported !== rightSupported) return rightSupported - leftSupported;
    const leftSoft = Number(Boolean(left?.soft_supported_hit));
    const rightSoft = Number(Boolean(right?.soft_supported_hit));
    if (leftSoft !== rightSoft) return rightSoft - leftSoft;
    const leftPrior = Number(left?.prior ?? 0);
    const rightPrior = Number(right?.prior ?? 0);
    if (leftPrior !== rightPrior) return rightPrior - leftPrior;
    const leftSupport = Number(left?.support_sum ?? 0);
    const rightSupport = Number(right?.support_sum ?? 0);
    if (leftSupport !== rightSupport) return rightSupport - leftSupport;
    return String(objectLabelForId(left?.object_id)).localeCompare(String(objectLabelForId(right?.object_id)));
  });
}

function summarizeContextObjects(rows, limit = 4) {
  const objectRows = sortContextObjectRows(
    (rows || []).filter((row) => row?.row_type === 'object_support'),
  ).slice(0, limit);
  if (!objectRows.length) return 'Sin lectura por objeto visible.';
  return objectRows.map((row) => {
    const parts = [];
    if (row.prior != null) parts.push(`prior=${pretty(row.prior)}`);
    if (row.support_sum != null) parts.push(`support=${pretty(row.support_sum)}`);
    return `${objectLabelForId(row.object_id)} (${parts.join(' · ')})`;
  }).join(' · ');
}

function nodeSpecificOutputRows(nodeId, nodeRun) {
  if (String(nodeId) === 'prepare.reliable_visual_anchors') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Selecciona solo pares detección-objeto muy claros para usarlos como referencias visuales fiables en Hipótesis de sets.',
      },
      {
        label: 'Resumen de clase',
        value: `anchors=${pretty(values.anchor_count ?? 0)} · thr score=${pretty(values.strong_score_threshold)} · thr gap=${pretty(values.clear_margin_threshold)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja un mapa objeto -> detección anchor por clase; no resuelve todavía el matching final del frame.',
      },
    ];
  }

  if (String(nodeId) === 'visual.build_candidates') {
    const rows = filteredRows(nodeRun, state.selectedDetId);
    const detRows = rows.detectionRows.length ? rows.detectionRows : (nodeRun?.detection_rows || []);
    return [
      {
        label: 'Score mostrado',
        value: 'score_sim, una similitud visual compuesta detección-objeto.',
      },
      {
        label: 'Candidatos',
        value: detRows.length
          ? 'Lista ordenada de objetos candidatos por detección para que Diagnóstico visual, Shaping por candidato y Tablas finales de score trabajen sobre una base común.'
          : 'Sin candidatos visibles en el foco actual.',
      },
      {
        label: 'Qué deja preparado',
        value: 'Un ranking visual inicial; aquí todavía no se veta por contexto ni se resuelve el matching final.',
      },
    ];
  }

  if (String(nodeId) === 'visual.report_diagnosis') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Clasifica cada detección como STRONG, AMBIGUOUS o WEAK usando solo la evidencia visual ya construida.',
      },
      {
        label: 'Resumen de clase',
        value: `strong=${pretty(values.n_strong ?? 0)} · ambiguous=${pretty(values.n_ambiguous ?? 0)} · weak=${pretty(values.n_weak ?? 0)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja una lectura compacta de la fiabilidad visual antes de entrar en shortlist, soft-gate y matching global.',
      },
    ];
  }

  if (String(nodeId) === 'prepare.class_partition') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Aísla la unidad frame+class para que el resto del flujo no mezcle detecciones u objetos de otras clases.',
      },
      {
        label: 'Resumen de clase',
        value: `detections=${pretty(values.detection_count ?? 0)} · objetos_memoria=${pretty(values.snapshot_object_count ?? 0)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja los participantes exactos que usarán Construcción de candidatos, Anchors visuales fiables y los bloques de matching.',
      },
    ];
  }

  if (String(nodeId) === 'prepare.valid_detections') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Separa las detecciones con features comparables de las que no pueden seguir a la rama de matching conocido.',
      },
      {
        label: 'Resumen de clase',
        value: `valid=${pretty(values.valid_count ?? 0)} · sin_features=${pretty(values.missing_feature_count ?? 0)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja qué detecciones entran en Uso de contexto por reporte y cuáles ya salen de la rama conocida.',
      },
    ];
  }

  if (String(nodeId) === 'context.neighbor_sets_hypotheses') {
    const values = nodeRun?.values || {};
    const objectSummary = summarizeContextObjects(nodeRun?.global_rows || []);
    return [
      {
        label: 'Salida del bloque',
        value: 'Construye varias hipótesis globales de compatibilidad entre las detecciones visibles y objetos ya conocidos antes de decidir si ese contexto se puede usar.',
      },
      {
        label: 'Resumen global',
        value: `retenidas=${pretty(values.retained_hypotheses ?? values.n_hypotheses)} · topk=${pretty(values.topk_sets_limit)} · beam=${pretty(values.beam_width)} · kernel=${pretty(values.context_k)}`,
      },
      {
        label: 'Objetos más plausibles',
        value: objectSummary,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja hipótesis retenidas, shortlist y prior por objeto para Activación de sets; además deja una lectura por objeto que puede reapoyar identidades fuera del top-k mediante afinidad con el kernel contextual.',
      },
    ];
  }

  if (String(nodeId) === 'context.sets_activation') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Decide si el contexto relacional ya construido es suficientemente consistente como para entrar de verdad en la asociación.',
      },
      {
        label: 'Resumen global',
        value: `enabled=${pretty(values.enabled)} · global_ok=${pretty(values.global_ok)} · quality=${pretty(values.quality)} · reason=${pretty(values.reason)}`,
      },
      {
        label: 'Qué deja preparado',
        value: `Deja el contexto de la clase en estado activo, degradado o inactivo, con shortlist=${pretty(values.shortlist_size ?? 0)}, anchors=${pretty(values.anchor_count ?? 0)} y prior=${pretty(values.prior_count ?? 0)} para Uso de contexto por reporte, Shaping por candidato y Tablas finales de score.`,
      },
    ];
  }

  if (String(nodeId) === 'shape.allow_for_report') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Decide, detección a detección, si el contexto de sets puede influir en esta rama o si la evidencia visual aún es demasiado débil o inestable para usarlo.',
      },
      {
        label: 'Resumen de clase',
        value: `allowed=${pretty(values.allowed_count ?? 0)} · blocked=${pretty(values.blocked_count ?? 0)} · context=${pretty(values.class_context_available)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Marca qué detecciones podrán usar contexto al construir candidatos operativos en Shaping por candidato y Tablas finales de score.',
      },
    ];
  }

  if (String(nodeId) === 'shape.context_veto') {
    const rows = filteredRows(nodeRun, state.selectedDetId).candidateRows.length
      ? filteredRows(nodeRun, state.selectedDetId).candidateRows
      : (nodeRun?.candidate_rows || []);
    const kept = rows.filter((row) => Number(row.decision_keep) === 1).length;
    const vetoed = rows.filter((row) => Number(row.decision_keep) !== 1).length;
    return [
      {
        label: 'Salida del bloque',
        value: 'Agrupa el shaping tardío por candidato: plausibilidad conocida, gates por umbral, rescate contextual, veto fuerte y filtros finales antes de las tablas operativas.',
      },
      {
        label: 'Resumen visible',
        value: `kept=${pretty(kept)} · vetoed=${pretty(vetoed)} · filas=${pretty(rows.length)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Reduce el conjunto operativo y deja marcado por qué cada candidato sigue vivo o cae antes de construir las tablas finales de score.',
      },
    ];
  }

  if (String(nodeId) === 'shape.final_score_tables') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Materializa las tres tablas operativas del matching: score_sim, score_assign y score_final, después de todos los gates y bonus ya aplicados.',
      },
      {
        label: 'Resumen de clase',
        value: `detections=${pretty(values.detection_count ?? 0)} · objetos=${pretty(values.candidate_object_count ?? 0)} · filas=${pretty(values.ranked_row_count ?? 0)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja score_sim para lectura visual, score_assign para optimización estable en Hungarian y score_final para aceptación dura, reporting y post-assignment.',
      },
    ];
  }

  if (String(nodeId) === 'resolve.locks') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Cierra matches muy claros sin mandar esos casos a una resolución conjunta innecesaria.',
      },
      {
        label: 'Resumen de clase',
        value: `locked=${pretty(values.locked_count ?? 0)} · dets=${pretty(values.candidate_det_count ?? 0)} · objetos=${pretty(values.candidate_object_count ?? 0)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Reduce el problema que llega a Hungarian y deja solo la parte que todavía requiere resolución global.',
      },
    ];
  }

  if (String(nodeId) === 'resolve.hungarian') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Resuelve conjuntamente las detecciones restantes contra objetos y dummies, y luego somete la asignación elegida a aceptación dura por score.',
      },
      {
        label: 'Resumen global',
        value: `detecciones=${pretty(values.participant_det_ids?.length ?? 0)} · objetos=${pretty(values.object_column_count ?? 0)} · dummies=${pretty(values.dummy_column_count ?? 0)} · matches=${pretty(values.n_matches ?? 0)} · creates=${pretty(values.n_creates ?? 0)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja una propuesta global ya filtrada por umbrales para que entren las guards post-assignment; no anota todavía el outcome final legible.',
      },
    ];
  }

  if (String(nodeId) === 'post.assignment_ambiguity') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Revisa si la asignación ya resuelta sigue dejando componentes ambiguos que merecen tratamiento posterior.',
      },
      {
        label: 'Resumen global',
        value: `componentes=${pretty(values.component_count ?? 0)} · ambiguos=${pretty(values.ambiguous_component_count ?? 0)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja señalizado si Identity stability, known-set-distance o reconciliación temporal deben leer un caso todavía dudoso.',
      },
    ];
  }

  if (String(nodeId) === 'post.identity_stability') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Comprueba si los matches elegidos son suficientemente estables como para mantenerse sin reinterpretación.',
      },
      {
        label: 'Resumen global',
        value: `kept=${pretty(values.kept_count ?? 0)} · remapped=${pretty(values.remapped_count ?? 0)} · diverted=${pretty(values.diverted_count ?? 0)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja los matches frágiles ya filtrados o redirigidos antes de create competition y reconciliación provisional.',
      },
    ];
  }

  if (String(nodeId) === 'post.known_set_distance_disambiguation') {
    return [
      {
        label: 'Salida del bloque',
        value: 'Intenta romper ambigüedades conocidas usando memoria relacional y señales adicionales cuando la asignación principal no basta.',
      },
      {
        label: 'Resumen global',
        value: summarizeNodeRun(orderedNodes().find((item) => item.id === nodeId), nodeRun),
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja menos casos ambiguos antes de la reconciliación provisional y del outcome final.',
      },
    ];
  }

  if (String(nodeId) === 'post.create_competition') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Decide cuándo una alternativa de create realmente compite contra opciones conocidas y cuándo no merece abrir identidad nueva.',
      },
      {
        label: 'Resumen global',
        value: `create_entries=${pretty(values.create_entry_count ?? 0)} · competitions=${pretty(values.competition_count ?? 0)} · selected=${pretty(values.selected_competition_count ?? 0)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja resuelta la competencia create-vs-known antes de la reconciliación provisional.',
      },
    ];
  }

  if (String(nodeId) === 'post.ambiguous_track_candidates') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Fusiona la ambigüedad contextual, la identidad inestable y las competiciones committed-new en una bolsa real de candidatos ambiguos por detección.',
      },
      {
        label: 'Resumen global',
        value: `candidatos=${pretty(values.candidate_count ?? 0)} · policy=${pretty(values.policy_count ?? 0)} · identity=${pretty(values.identity_stability_count ?? 0)} · committed_new=${pretty(values.committed_new_count ?? 0)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja exactamente qué detecciones pasan a known-set-distance o a la reconciliación temporal y desde qué fuente llegó cada ambigüedad.',
      },
    ];
  }

  if (String(nodeId) === 'post.provisional_reconciliation') {
    return [
      {
        label: 'Salida del bloque',
        value: 'Relee creates y ambigüedades residuales para decidir si deben quedarse como create, promoverse a ambiguo o convertirse en provisional_new/provisional_parent.',
      },
      {
        label: 'Resumen global',
        value: summarizeNodeRun(orderedNodes().find((item) => item.id === nodeId), nodeRun),
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja candidatos temporales y ambiguos ya reinterpretados antes del empaquetado final y del outcome legible.',
      },
    ];
  }

  if (String(nodeId) === 'post.final_decision_pack') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Aplica la precedencia final entre matches, creates, ambiguous y provisional para que cada detección quede en un único bucket semántico.',
      },
      {
        label: 'Resumen global',
        value: `in match=${pretty(values.input_match_count ?? 0)} · in create=${pretty(values.input_create_count ?? 0)} · in ambiguous=${pretty(values.input_ambiguous_count ?? 0)} · in provisional=${pretty(values.input_provisional_count ?? 0)} · out match=${pretty(values.final_match_count ?? 0)} · out create=${pretty(values.final_create_count ?? 0)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja el estado final no conflictivo que outcome.final_ambiguity y outcome.finalize leerán como salidas hermanas de cierre.',
      },
    ];
  }

  if (String(nodeId) === 'outcome.finalize') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Anota en cada reporte la decisión final legible, la razón final y los ids de soporte o candidatos ambiguos que correspondan.',
      },
      {
        label: 'Resumen de clase',
        value: pretty(values.decision_counts ?? {}),
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja una decisión final ya consumible por update sin reinterpretaciones extra dentro de association.',
      },
    ];
  }

  if (String(nodeId) === 'outcome.final_ambiguity') {
    const values = nodeRun?.values || {};
    return [
      {
        label: 'Salida del bloque',
        value: 'Recalcula la claridad final del caso usando score_final y candidatos elegibles, ya después del empaquetado semántico final.',
      },
      {
        label: 'Resumen de clase',
        value: `strong=${pretty(values.n_strong ?? 0)} · ambiguous=${pretty(values.n_ambiguous ?? 0)} · weak=${pretty(values.n_weak ?? 0)}`,
      },
      {
        label: 'Qué deja preparado',
        value: 'Deja visible la claridad final del caso; esta lectura convive con outcome.finalize, no le da origen causal.',
      },
    ];
  }

  return [
    {
      label: 'Qué hace',
      value: generalDescription(nodeId),
    },
    {
      label: 'Por qué existe',
      value: nodeWhyText(nodeId),
    },
    {
      label: 'Qué deja preparado',
      value: nodePreparesText(nodeId),
    },
    {
      label: 'Lectura de esta run',
      value: summarizeNodeRun(orderedNodes().find((item) => item.id === nodeId), nodeRun),
    },
  ];
}

function nodeSpecificInputRows(nodeId, nodeRun, participants = nodeRun?.participants || {}) {
  const values = nodeRun?.values || {};
  const baseRows = [
    {
      label: 'Detecciones',
      value: (participants.det_ids || []).length
        ? (participants.det_ids || []).map((detId) => `det ${detId}`).join(', ')
        : 'sin detecciones visibles',
    },
    {
      label: 'Objetos',
      value: (participants.object_ids || []).length
        ? (participants.object_ids || []).map((objectId) => objectLabelForId(objectId)).join(', ')
        : 'sin objetos visibles',
    },
  ];

  if (String(nodeId) === 'prepare.class_partition') {
    return [
      {
        label: 'Entradas externas',
        value: 'Detecciones visibles de la clase y snapshot de objetos persistidos comparables.',
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'visual.build_candidates') {
    return [
      {
        label: 'Material que recibe',
        value: 'La clase ya particionada: detecciones visibles y objetos persistidos comparables de esa misma clase.',
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'prepare.reliable_visual_anchors') {
    return [
      {
        label: 'Base de entrada',
        value: 'Ranking visual ya construido por Construcción de candidatos, usando best_score_sim, second_score_sim y gap por detección.',
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'visual.report_diagnosis') {
    return [
      {
        label: 'Base de entrada',
        value: 'Ranking visual y score_sim por detección generados por Construcción de candidatos.',
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'context.neighbor_sets_hypotheses') {
    return [
      {
        label: 'Anchors disponibles',
        value: objectLabelsText(values.anchor_object_ids),
      },
      {
        label: 'Objetos con prior',
        value: objectLabelsText(values.prior_object_ids),
      },
      {
        label: 'Parámetros de búsqueda',
        value: `beam=${pretty(values.beam_width)} · topk=${pretty(values.topk_sets_limit)} · kernel=${pretty(values.context_k)}`,
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'context.sets_activation') {
    return [
      {
        label: 'Hipótesis retenidas',
        value: `n=${pretty(values.retained_hypotheses ?? values.n_hypotheses)} · best=${pretty(values.best)} · coverage=${pretty(values.coverage_eff)} · density=${pretty(values.density)} · maturity=${pretty(values.maturity)}`,
      },
      {
        label: 'Shortlist de sets',
        value: objectLabelsText(values.shortlist_object_ids),
      },
      {
        label: 'Anchors de clase',
        value: objectLabelsText(values.anchor_object_ids),
      },
      {
        label: 'Objetos con prior',
        value: objectLabelsText(values.prior_object_ids),
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'prepare.valid_detections') {
    return [
      {
        label: 'Base de entrada',
        value: 'Detecciones ya particionadas, antes de separar cuáles conservan features comparables.',
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'shape.allow_for_report') {
    return [
      {
        label: 'Diagnóstico visual',
        value: 'Estados STRONG / AMBIGUOUS / WEAK que llegan desde Diagnóstico visual.',
      },
      {
        label: 'Contexto de clase',
        value: `context_available=${pretty(values.class_context_available)} · shortlist=${pretty(values.shortlist_size)} · enabled=${pretty(values.context_enabled)}`,
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'shape.context_veto') {
    const rows = filteredRows(nodeRun, state.selectedDetId).candidateRows.length
      ? filteredRows(nodeRun, state.selectedDetId).candidateRows
      : (nodeRun?.candidate_rows || []);
    return [
      {
        label: 'Candidatos de entrada',
        value: `filas=${pretty(rows.length)} · contexto=${pretty(values.class_context_available)} · allow_for_report=${pretty(values.allowed_count)}`,
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'shape.final_score_tables') {
    return [
      {
        label: 'Filas candidatas vivas',
        value: `ranked_rows=${pretty(values.ranked_row_count)} · detections=${pretty(values.detection_count)} · objetos=${pretty(values.candidate_object_count)}`,
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'resolve.locks') {
    return [
      {
        label: 'Tabla de entrada',
        value: `detections=${pretty(values.candidate_det_count)} · objetos=${pretty(values.candidate_object_count)} · filas_score=${pretty(values.ranked_row_count)}`,
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'resolve.hungarian') {
    return [
      {
        label: 'Participantes restantes',
        value: `detecciones=${pretty(values.participant_det_ids?.length ?? 0)} · objetos=${pretty(values.object_column_count ?? 0)} · dummies=${pretty(values.dummy_column_count ?? 0)}`,
      },
      {
        label: 'Detecciones en Hungarian',
        value: detLabelsText(values.participant_det_ids),
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'post.assignment_ambiguity') {
    return [
      {
        label: 'Asignación recibida',
        value: 'Matches y creates que llegan desde Locks/Hungarian ya con aceptación dura aplicada.',
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'post.identity_stability') {
    return [
      {
        label: 'Matches iniciales',
        value: `count=${pretty(values.input_match_count ?? values.match_count)}`,
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'post.create_competition') {
    return [
      {
        label: 'Creates en estudio',
        value: `entries=${pretty(values.create_entry_count)} · competitions=${pretty(values.competition_count)}`,
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'post.ambiguous_track_candidates') {
    return [
      {
        label: 'Fuentes de ambigüedad',
        value: `policy=${pretty(values.policy_count)} · identity=${pretty(values.identity_stability_count)} · committed_new=${pretty(values.committed_new_count)}`,
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'post.known_set_distance_disambiguation') {
    return [
      {
        label: 'Componentes ambiguos',
        value: `components=${pretty(values.component_count)} · passes=${pretty(values.pass_count)}`,
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'post.provisional_reconciliation') {
    return [
      {
        label: 'Casos recibidos',
        value: `creates=${pretty(values.input_create_count)} · ambiguos=${pretty(values.input_ambiguous_count)} · provisionales=${pretty(values.input_provisional_count)}`,
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'post.final_decision_pack') {
    return [
      {
        label: 'Buckets de entrada',
        value: `match=${pretty(values.input_match_count)} · create=${pretty(values.input_create_count)} · ambiguous=${pretty(values.input_ambiguous_count)} · provisional=${pretty(values.input_provisional_count)}`,
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'outcome.final_ambiguity') {
    return [
      {
        label: 'Casos evaluables',
        value: 'Buckets finales ya empaquetados y score_final elegible por detección.',
      },
      ...baseRows,
    ];
  }

  if (String(nodeId) === 'outcome.finalize') {
    return [
      {
        label: 'Estado final recibido',
        value: `match=${pretty(values.match_count)} · create=${pretty(values.create_count)} · ambiguous=${pretty(values.ambiguous_count)} · provisional=${pretty(values.provisional_count)}`,
      },
      ...baseRows,
    ];
  }

  return baseRows;
}

function candidateTermScore(row, term) {
  const keyByTerm = {
    obj: 'score_obj',
    bg: 'score_bg',
    bg_partial: 'score_bg_partial',
    parts: 'score_parts',
  };
  const value = row?.[keyByTerm[term]];
  return value == null ? null : Number(value);
}

function candidateRawWeight(row, term) {
  const keyByTerm = {
    obj: 'weight_eff_obj',
    bg: 'weight_eff_bg',
    bg_partial: 'weight_eff_bg_partial',
    parts: 'weight_eff_parts',
  };
  const value = Number(row?.[keyByTerm[term]] ?? 0);
  return Number.isFinite(value) ? value : 0;
}

function candidateNormalizedWeight(row, term) {
  const terms = ['obj', 'bg', 'bg_partial', 'parts'];
  const score = candidateTermScore(row, term);
  if (score == null) return null;
  const total = terms.reduce((acc, currentTerm) => {
    const currentScore = candidateTermScore(row, currentTerm);
    if (currentScore == null) return acc;
    return acc + candidateRawWeight(row, currentTerm);
  }, 0);
  if (total <= 1e-12) return null;
  return candidateRawWeight(row, term) / total;
}

function reliableAnchorReasonLabel(reason) {
  const labels = {
    NO_VISUAL_CANDIDATES: 'no hay candidatos visuales para esta detección',
    BELOW_STRONG_THRESHOLD: 'el mejor candidato no alcanza el umbral fuerte de score',
    BELOW_CLEAR_MARGIN: 'la ventaja sobre el segundo candidato no alcanza el margen mínimo',
    RELIABLE_VISUAL_ANCHOR: 'la detección supera los umbrales y conserva el objeto como anchor',
    SUPERSEDED_BY_BETTER_DETECTION: 'otra detección reclama ese mismo objeto con mejor evidencia visual',
  };
  return labels[String(reason)] || pretty(reason);
}

function reliableAnchorSelectionLabel(row) {
  return row?.selected_as_anchor ? 'anchor' : 'no anchor';
}

function renderReliableVisualAnchorsSection(infoContainer, valueContainer) {
  const anchorRun = getNodeRun('prepare.reliable_visual_anchors');
  const rows = anchorRun?.detection_rows || [];
  const values = anchorRun?.values || {};
  const activeRows = state.selectedDetId == null
    ? rows
    : rows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });

  const whatCard = document.createElement('div');
  whatCard.className = 'detail-card';
  whatCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
    <p>
      Este nodo no cierra el matching del frame. Hace una selección temprana y
      conservadora de pares detección-objeto que ya parecen muy claros solo
      con evidencia visual.
    </p>
    <p>
      La idea es dejar un pequeño conjunto de referencias fiables para que
      <strong>Hipótesis de sets</strong> no trabaje a ciegas. Si aquí no sale
      ningún anchor, el pipeline sigue, pero ese bloque tendrá menos apoyo
      para construir hipótesis y kernel.
    </p>
  `;
  explanationSection.body.appendChild(whatCard);

  const decisionCard = document.createElement('div');
  decisionCard.className = 'detail-card';
  decisionCard.innerHTML = `
    <div class="detail-card-head"><strong>Cómo decide</strong></div>
    <p>
      Para cada detección mira el mejor candidato visual
      (<code>best_score_sim</code>) y el segundo mejor
      (<code>second_score_sim</code>). Solo puede proponer un anchor si el
      mejor supera el umbral fuerte y además la diferencia
      (<code>gap</code>) supera el margen mínimo.
    </p>
    <p>
      Después aplica unicidad por objeto: si varias detecciones apuntan al
      mismo objeto, solo sobrevive como anchor la detección propietaria de ese
      objeto, es decir, la que acaba retenida en
      <code>selected_anchor_det_id</code>.
    </p>
  `;
  explanationSection.body.appendChild(decisionCard);

  const whyCard = document.createElement('div');
  whyCard.className = 'detail-card';
  whyCard.innerHTML = `
    <div class="detail-card-head"><strong>Para qué sirve</strong></div>
    <p>
      La salida de este nodo alimenta sobre todo
      <strong>Hipótesis de sets</strong> y, desde ahí,
      <strong>Activación de sets</strong>. Un anchor no significa “matching
      final confirmado”; significa “referencia visual lo bastante fiable como
      para apoyar el contexto”.
    </p>
    <p>
      En esta traza la clase deja <code>${pretty(values.anchor_count ?? 0)}</code>
      anchors, con umbrales <code>score >= ${pretty(values.strong_score_threshold)}</code>
      y <code>gap >= ${pretty(values.clear_margin_threshold)}</code>.
    </p>
  `;
  explanationSection.body.appendChild(whyCard);
  infoContainer.appendChild(explanationSection.section);

  const perDetectionSection = createDetailSection('Selección por detección', {
    badge: `${rows.length} filas`,
    open: false,
  });

  if (!rows.length) {
    perDetectionSection.body.innerHTML = '<div class="empty-state">La traza no serializa filas por detección para este nodo.</div>';
    valueContainer.appendChild(perDetectionSection.section);
    return;
  }

  for (const [detId, detRows] of groupRowsByDetection(activeRows)) {
    const detRow = detRows[0];
    const card = document.createElement('div');
    card.className = 'detail-card';
    const header = document.createElement('div');
    header.className = 'detail-card-head';
    header.innerHTML = `
      <strong>det ${detId}</strong>
      <span class="badge">${reliableAnchorSelectionLabel(detRow)} · ${reliableAnchorReasonLabel(detRow?.reason)}</span>
    `;
    card.appendChild(header);

    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>mejor objeto</th>
          <th>segundo objeto</th>
          <th>best</th>
          <th>second</th>
          <th>gap</th>
          <th>thr score</th>
          <th>thr gap</th>
          <th>score ok</th>
          <th>gap ok</th>
          <th>owner</th>
          <th>resultado</th>
          <th>razón</th>
        </tr>
      </thead>
      <tbody>
        ${detRows.map((row) => `
          <tr>
            <td>${objectLabelForId(row.best_object_id)}</td>
            <td>${objectLabelForId(row.second_object_id)}</td>
            <td>${pretty(row.best_score_sim)}</td>
            <td>${pretty(row.second_score_sim)}</td>
            <td>${pretty(row.gap)}</td>
            <td>${pretty(row.score_threshold)}</td>
            <td>${pretty(row.margin_threshold)}</td>
            <td>${row.passes_score_threshold ? 'si' : 'no'}</td>
            <td>${row.passes_margin_threshold ? 'si' : 'no'}</td>
            <td>${row.selected_anchor_det_id == null ? '—' : `det ${pretty(row.selected_anchor_det_id)}`}</td>
            <td>${reliableAnchorSelectionLabel(row)}</td>
            <td>${reliableAnchorReasonLabel(row.reason)}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    card.appendChild(tableWrap);
    perDetectionSection.body.appendChild(card);
  }

  valueContainer.appendChild(perDetectionSection.section);
}

function renderNeighborSetsHypothesesSection(infoContainer, valueContainer) {
  const setsRun = getNodeRun('context.neighbor_sets_hypotheses');
  const values = setsRun?.values || {};
  const allRows = setsRun?.global_rows || [];
  const hypothesisRows = allRows.filter((row) => !row?.row_type || row.row_type === 'hypothesis');
  const objectRows = allRows.filter((row) => row?.row_type === 'object_support');

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });

  const whatCard = document.createElement('div');
  whatCard.className = 'detail-card';
  whatCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
    <p>
      Este nodo intenta construir una lectura relacional global del frame.
      En vez de mirar cada detección por separado, propone combinaciones
      plausibles de objetos conocidos que podrían explicar conjuntamente lo que
      aparece en escena.
    </p>
    <p>
      La idea no es decidir todavía el matching final, sino formular varias
      hipótesis de contexto: qué objetos parecen compatibles entre sí, cuáles
      coaparecen de forma creíble y qué subconjunto del inventario conocido
      encaja mejor con el frame actual.
    </p>
  `;
  explanationSection.body.appendChild(whatCard);

  const inputsCard = document.createElement('div');
  inputsCard.className = 'detail-card';
  inputsCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué usa como entrada</strong></div>
    <p>
      Se apoya en tres fuentes: las detecciones visibles del frame, los anchors
      visuales fiables ya confirmados en el paso anterior y la memoria de
      vecindad aprendida entre objetos conocidos.
    </p>
    <p>
      Con eso genera hipótesis relacionales de conjunto, no decisiones
      individuales por detección.
    </p>
  `;
  explanationSection.body.appendChild(inputsCard);

  const outputsCard = document.createElement('div');
  outputsCard.className = 'detail-card';
  outputsCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué deja preparado</strong></div>
    <p>
      Deja un conjunto retenido de hipótesis, una shortlist global de objetos
      plausibles y un prior por objeto. Ese material todavía no significa que
      el contexto vaya a usarse: solo significa que ya existe una propuesta
      relacional que el bloque <strong>Activación de sets</strong> podrá
      aceptar, degradar o descartar.
    </p>
    <p>
      Ojo con la lectura: aquí no se enumeran todas las combinaciones posibles
      del frame. La búsqueda ya viene acotada y esta salida conserva solo las
      hipótesis retenidas porque hacen falta una shortlist y un prior
      discriminativos, no una nube plana de alternativas. La recuperación por
      <code>kernel</code> que se ve en la tabla
      <em>Objetos de la clase respaldados por el contexto</em> no sustituye a
      ese top-k: lo usa como semilla para reapoyar objetos parecidos sin
      perder foco ni llenar el contexto de ruido.
    </p>
  `;
  explanationSection.body.appendChild(outputsCard);

  const whyTopKCard = document.createElement('div');
  whyTopKCard.className = 'detail-card';
  whyTopKCard.innerHTML = `
    <div class="detail-card-head"><strong>Por qué no usa todas las hipótesis</strong></div>
    <p>
      Si este bloque arrastrara demasiadas hipótesis, el contexto se volvería
      demasiado plano: muchos objetos recibirían apoyo parecido, el
      <code>prior</code> perdería capacidad para discriminar y
      <strong>Activación de sets</strong> vería una escena menos selectiva.
    </p>
    <p>
      Por eso este bloque primero conserva una parte fuerte del contexto y
      solo después permite que la afinidad con el <code>kernel</code> vuelva a
      abrir un poco el foco sobre otros objetos de la clase. No hacen la misma
      función: el top-k concentra la hipótesis base y el kernel recupera
      variantes cercanas sin rehacer la búsqueda completa.
    </p>
  `;
  explanationSection.body.appendChild(whyTopKCard);
  infoContainer.appendChild(explanationSection.section);

  const summarySection = createDetailSection('Resumen de hipótesis', {
    badge: setsRun?.decision?.branch || '—',
    open: false,
  });

  renderFactRows(summarySection.body, [
    { label: 'computed', value: pretty(values.computed) },
    { label: 'hipótesis retenidas', value: pretty(values.retained_hypotheses ?? values.n_hypotheses) },
    { label: 'beam width', value: pretty(values.beam_width) },
    { label: 'topk sets', value: pretty(values.topk_sets_limit) },
    { label: 'context k', value: pretty(values.context_k) },
    { label: 'best_score', value: pretty(values.best_score) },
    { label: 'second_score', value: pretty(values.second_score) },
    { label: 'gap_best', value: pretty(values.gap_best) },
    { label: 'coverage_eff_best', value: pretty(values.coverage_eff_best) },
    { label: 'density_best', value: pretty(values.density_best) },
    { label: 'mean_maturity_best', value: pretty(values.mean_maturity_best) },
    { label: 'objetos plausibles de la clase', value: objectLabelsText(values.shortlist_object_ids) },
    { label: 'anchors de la clase', value: objectLabelsText(values.anchor_object_ids) },
    { label: 'objetos con prior en la clase', value: objectLabelsText(values.prior_object_ids) },
    { label: 'class_prior', value: pretty(values.class_prior) },
  ]);
  valueContainer.appendChild(summarySection.section);

  const hypothesesSection = createDetailSection('Hipótesis relevantes para la clase', {
    badge: `${hypothesisRows.length} filas`,
    open: false,
  });

  if (!hypothesisRows.length) {
    hypothesesSection.body.innerHTML = '<div class="empty-state">No hay hipótesis serializadas para esta clase en la traza actual.</div>';
    valueContainer.appendChild(hypothesesSection.section);
  } else {
    const glossaryCard = document.createElement('div');
    glossaryCard.className = 'detail-card';
    glossaryCard.innerHTML = `
      <div class="detail-card-head"><strong>Cómo leer esta tabla</strong></div>
      <p>
        <strong>score</strong>: fuerza global de la hipótesis.
        <strong> k</strong>: tamaño del grupo de objetos que propone.
        <strong> coverage</strong>: cuánto de la escena consigue explicar.
      </p>
      <p>
        <strong>density</strong>: lo compacta o coherente que resulta la relación entre sus objetos.
        <strong> maturity</strong>: cuán madura parece la memoria relacional que sostiene esa hipótesis.
      </p>
    `;
    hypothesesSection.body.appendChild(glossaryCard);

    const card = document.createElement('div');
    card.className = 'detail-card';
    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>rank</th>
          <th>score</th>
          <th>objetos</th>
          <th>dets explicadas</th>
          <th>k</th>
          <th>coverage</th>
          <th>density</th>
          <th>maturity</th>
        </tr>
      </thead>
      <tbody>
        ${hypothesisRows.map((row) => `
          <tr>
            <td>${pretty(row.rank)}</td>
            <td>${pretty(row.score_sets)}</td>
            <td>${objectLabelsText(row.object_ids)}</td>
            <td>${detLabelsText(row.det_ids_explained)}</td>
            <td>${pretty(row.k)}</td>
            <td>${pretty(row.coverage_eff)}</td>
            <td>${pretty(row.density)}</td>
            <td>${pretty(row.mean_maturity)}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    card.appendChild(tableWrap);
    hypothesesSection.body.appendChild(card);
    valueContainer.appendChild(hypothesesSection.section);
  }

  const objectRowsSorted = sortContextObjectRows(objectRows);

  const objectsSection = createDetailSection('Objetos de la clase respaldados por el contexto', {
    badge: `${objectRowsSorted.length} objetos`,
    open: false,
  });

  if (!objectRowsSorted.length) {
    objectsSection.body.innerHTML = '<div class="empty-state">No hay lectura por objeto serializada para este contexto.</div>';
    valueContainer.appendChild(objectsSection.section);
    return;
  }

  const objectsCard = document.createElement('div');
  objectsCard.className = 'detail-card';
  objectsCard.innerHTML = `
    <div class="detail-card-head"><strong>Lectura por objeto de la clase</strong></div>
    <p>
      Esta tabla no representa la activación del contexto, sino cómo queda
      distribuido el apoyo relacional entre todos los objetos conocidos de la
      clase una vez construidas las hipótesis globales.
    </p>
    <p>
      Por eso aquí sí pueden aparecer objetos que no quedaron arriba en las
      hipótesis retenidas: el pipeline vuelve a mirar todos los objetos de la
      clase y les da apoyo adicional según su afinidad con el kernel
      contextual que sale de anchors e hipótesis fuertes.
    </p>
  `;
  const glossaryCard = document.createElement('div');
  glossaryCard.className = 'detail-card';
  glossaryCard.innerHTML = `
    <div class="detail-card-head"><strong>Cómo leer esta tabla</strong></div>
    <p>
      <strong>prior</strong>: apoyo previo del objeto en este frame.
      <strong> support sum</strong>: apoyo acumulado que recibe desde el contexto.
      <strong> madurez</strong>, <strong>hits</strong> y <strong>episodes</strong>: cuánto historial y estabilidad arrastra en memoria.
    </p>
    <p>
      <strong>shortlist</strong>, <strong>supported</strong> y <strong>soft</strong>: nivel de respaldo contextual.
      <strong> coverage ok</strong>: si tiene base mínima para ser tenido en cuenta.
      <strong>compat/kernel/hyp</strong>: distintas lecturas de afinidad relacional desde vecindad e hipótesis.
    </p>
  `;
  const objectsWrap = document.createElement('div');
  objectsWrap.className = 'table-wrap';
  const objectsTable = document.createElement('table');
  objectsTable.className = 'detail-table';
  objectsTable.innerHTML = `
    <thead>
      <tr>
        <th>objeto</th>
        <th>prior</th>
        <th>support sum</th>
        <th>madurez</th>
        <th>hits</th>
        <th>episodes</th>
        <th>shortlist</th>
        <th>supported</th>
        <th>soft</th>
        <th>coverage ok</th>
        <th>compat rel</th>
        <th>kernel raw</th>
        <th>kernel hits</th>
        <th>kernel hit ratio</th>
        <th>kernel rel</th>
        <th>hyp rel</th>
      </tr>
    </thead>
    <tbody>
      ${objectRowsSorted.map((row) => `
        <tr>
          <td>${objectLabelForId(row.object_id)}</td>
          <td>${pretty(row.prior)}</td>
          <td>${pretty(row.support_sum)}</td>
          <td>${pretty(row.maturity_score)}</td>
          <td>${pretty(row.hits)}</td>
          <td>${pretty(row.neighbor_episode_count)}</td>
          <td>${row.shortlist_hit ? 'si' : 'no'}</td>
          <td>${row.supported_hit ? 'si' : 'no'}</td>
          <td>${row.soft_supported_hit ? 'si' : 'no'}</td>
          <td>${row.coverage_ok ? 'si' : 'no'}</td>
          <td>${pretty(row.compat_rel)}</td>
          <td>${pretty(row.kernel_raw)}</td>
          <td>${pretty(row.kernel_hit_count)}</td>
          <td>${pretty(row.kernel_hit_ratio)}</td>
          <td>${pretty(row.kernel_rel)}</td>
          <td>${pretty(row.hyp_rel)}</td>
        </tr>
      `).join('')}
    </tbody>
  `;
  objectsWrap.appendChild(objectsTable);
  objectsCard.appendChild(objectsWrap);
  objectsSection.body.appendChild(glossaryCard);
  objectsSection.body.appendChild(objectsCard);
  valueContainer.appendChild(objectsSection.section);
}

function renderSetsActivationSection(infoContainer, valueContainer) {
  const setsRun = getNodeRun('context.sets_activation');
  const values = setsRun?.values || {};

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });

  const whatCard = document.createElement('div');
  whatCard.className = 'detail-card';
  whatCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
    <p>
      Este nodo no construye ya hipótesis nuevas. Parte de las hipótesis
      relacionales generadas justo antes y decide si ese contexto es lo bastante
      consistente como para ayudar de verdad a la asociación.
    </p>
    <p>
      La pregunta aquí no es “qué hipótesis hay”, sino “¿merece la pena confiar
      en ellas para influir en <strong>Uso de contexto por reporte</strong>,
      <strong>Shaping por candidato</strong> y
      <strong>Tablas finales de score</strong>?”.
    </p>
  `;
  explanationSection.body.appendChild(whatCard);

  const inputsCard = document.createElement('div');
  inputsCard.className = 'detail-card';
  inputsCard.innerHTML = `
    <div class="detail-card-head"><strong>Entradas reales</strong></div>
    <p>
      Recibe un resumen de las hipótesis de sets ya construidas: cuántas hay,
      qué tamaño tiene la mejor, qué cobertura logran, cuánta consistencia
      relacional muestran y qué shortlist de objetos plausibles dejan abierta.
    </p>
    <p>
      En esta traza la clase recibe <code>${pretty(values.n_hypotheses ?? 0)}</code>
      hipótesis globales, con <code>${pretty(values.shortlist_size ?? 0)}</code>
      objetos plausibles y <code>${pretty(values.anchor_count ?? 0)}</code>
      anchors relevantes dentro de su clase.
    </p>
  `;
  explanationSection.body.appendChild(inputsCard);

  const decisionCard = document.createElement('div');
  decisionCard.className = 'detail-card';
  decisionCard.innerHTML = `
    <div class="detail-card-head"><strong>Cómo decide</strong></div>
    <p>
      Resume la calidad global del contexto mirando si las hipótesis son pocas
      o abundantes, si explican bien el frame, si el mejor grupo es demasiado
      pequeño, si la escena resulta coherente relacionalmente y si la memoria
      que lo sostiene parece madura o frágil.
    </p>
    <p>
      Si esa mezcla sale suficientemente consistente, el bloque marca el
      contexto como utilizable. Si no, lo deja degradado o inactivo. El
      resultado no decide aún un candidato concreto: solo regula cuánto peso
      contextual se permitirá después.
    </p>
  `;
  explanationSection.body.appendChild(decisionCard);

  const outputsCard = document.createElement('div');
  outputsCard.className = 'detail-card';
  outputsCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué deja preparado</strong></div>
    <p>
      Deja el contexto de la clase en uno de tres estados prácticos:
      activo, degradado o inactivo. Junto con eso conserva una shortlist
      plausible, los anchors relevantes y el prior conocido de la clase.
    </p>
    <p>
      Este nodo todavía no aplica bonus ni veto sobre candidatos concretos.
      Eso ocurre en <strong>Uso de contexto por reporte</strong>,
      <strong>Shaping por candidato</strong> y
      <strong>Tablas finales de score</strong>.
    </p>
  `;
  explanationSection.body.appendChild(outputsCard);
  infoContainer.appendChild(explanationSection.section);

  const summarySection = createDetailSection('Resumen del contexto', {
    badge: setsRun?.decision?.branch || '—',
    open: false,
  });

  renderFactRows(summarySection.body, [
    { label: 'enabled', value: pretty(values.enabled) },
    { label: 'global_ok', value: pretty(values.global_ok) },
    { label: 'reason', value: pretty(values.reason) },
    { label: 'quality', value: pretty(values.quality) },
    { label: 'best', value: pretty(values.best) },
    { label: 'coverage_eff', value: pretty(values.coverage_eff) },
    { label: 'maturity', value: pretty(values.maturity) },
    { label: 'density', value: pretty(values.density) },
    { label: 'k_best', value: pretty(values.k_best) },
    { label: 'n_hypotheses', value: pretty(values.n_hypotheses) },
    { label: 'quality_threshold', value: pretty(values.quality_threshold) },
    { label: 'best_score_threshold', value: pretty(values.best_score_threshold) },
    { label: 'coverage_eff_threshold', value: pretty(values.coverage_eff_threshold) },
    { label: 'min_size_threshold', value: pretty(values.min_size_threshold) },
    { label: 'objetos plausibles de la clase', value: objectLabelsText(values.shortlist_object_ids) },
    { label: 'anchors de la clase', value: objectLabelsText(values.anchor_object_ids) },
    { label: 'objetos con prior en la clase', value: objectLabelsText(values.prior_object_ids) },
  ]);
  valueContainer.appendChild(summarySection.section);

}

function renderAllowForReportSection(infoContainer, valueContainer) {
  const allowRun = getNodeRun('shape.allow_for_report');
  const values = allowRun?.values || {};
  const rows = allowRun?.detection_rows || [];
  const activeRows = state.selectedDetId == null
    ? rows
    : rows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });

  const whatCard = document.createElement('div');
  whatCard.className = 'detail-card';
  whatCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
    <p>
      Este nodo no rankea candidatos ni decide matches. Solo marca si, para
      esa detección, el contexto de <code>sets</code> está autorizado a entrar
      después como señal contextual adicional.
    </p>
    <p>
      El resultado es por detección, no por objeto. Una misma clase puede
      tener ramas que sí usan contexto y otras que siguen casi solo con
      evidencia visual.
    </p>
  `;
  explanationSection.body.appendChild(whatCard);

  const outputsCard = document.createElement('div');
  outputsCard.className = 'detail-card';
  outputsCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué deja preparado</strong></div>
    <p>
      Marca qué detecciones podrán usar bonus o lectura contextual en las
      etapas posteriores. Si aquí una detección queda bloqueada, el contexto
      global no desaparece del frame, pero esa rama concreta no lo aprovecha.
    </p>
    <p>
      En esta traza hay <code>${pretty(values.allowed_count ?? 0)}</code>
      detecciones permitidas y <code>${pretty(values.blocked_count ?? 0)}</code>
      bloqueadas antes de entrar en <strong>Shaping por candidato</strong> y
      <strong>Tablas finales de score</strong>.
    </p>
  `;
  explanationSection.body.appendChild(outputsCard);
  infoContainer.appendChild(explanationSection.section);

  const summarySection = createDetailSection('Resumen por detección', {
    badge: `${rows.length} filas`,
    open: false,
  });

  if (!activeRows.length) {
    summarySection.body.innerHTML = '<div class="empty-state">No hay filas por detección visibles para este nodo.</div>';
    valueContainer.appendChild(summarySection.section);
    return;
  }

  const card = document.createElement('div');
  card.className = 'detail-card';
  const tableWrap = document.createElement('div');
  tableWrap.className = 'table-wrap';
  const table = document.createElement('table');
  table.className = 'detail-table';
  table.innerHTML = `
    <thead>
      <tr>
        <th>det</th>
        <th>estado visual</th>
        <th>contexto permitido</th>
        <th>razón</th>
      </tr>
    </thead>
    <tbody>
      ${activeRows.map((row) => `
        <tr>
          <td>det ${pretty(row.det_id)}</td>
          <td>${pretty(row.report_status)}</td>
          <td>${row.allowed ? 'si' : 'no'}</td>
          <td>${pretty(row.reason)}</td>
        </tr>
      `).join('')}
    </tbody>
  `;
  tableWrap.appendChild(table);
  card.appendChild(tableWrap);
  summarySection.body.appendChild(card);
  valueContainer.appendChild(summarySection.section);
}

function renderContextVetoSection(infoContainer, valueContainer) {
  const vetoRun = getNodeRun('shape.context_veto');
  const rows = vetoRun?.candidate_rows || [];
  const activeRows = state.selectedDetId == null
    ? rows
    : rows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });

  const whatCard = document.createElement('div');
  whatCard.className = 'detail-card';
  whatCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
    <p>
      Aquí ya se ve el shaping tardío real del matching conocido. No es solo
      “veto contextual”: en este bloque conviven plausibilidad conocida,
      gates por umbral, rescate contextual, veto fuerte y filtros duros antes
      de construir las tablas operativas.
    </p>
    <p>
      La salida sigue siendo por candidato, porque unas parejas
      detección-objeto pueden sobrevivir y otras caer aunque pertenezcan a la
      misma clase.
    </p>
  `;
  explanationSection.body.appendChild(whatCard);

  const decisionCard = document.createElement('div');
  decisionCard.className = 'detail-card';
  decisionCard.innerHTML = `
    <div class="detail-card-head"><strong>Cómo leer la tabla</strong></div>
    <p>
      <strong>plausible conocido</strong> indica si el candidato sigue vivo
      para razonamiento temporal. <strong>keep</strong> refleja si además sigue
      operativo para el matching actual. <strong>veto</strong> y
      <strong>gate</strong> cuentan qué lo bloqueó: veto contextual, umbral
      visual, rescate, objeto ya usado o filtros finales.
    </p>
    <p>
      Los términos <strong>sets</strong>, <strong>support</strong>,
      <strong>compat</strong>, <strong>kernel</strong> e <strong>hyp</strong>
      ayudan a leer qué parte del contexto lo empujaba o lo contradecía.
    </p>
    <p>
      Si aparece <code>LOCAL_CTX_CONTRADICTION</code>, las columnas locales
      ayudan a entenderla: cuántos episodios arrastra el objeto, qué kernel se
      usó como referencia, cuántos vecinos esperados tenía y cuánto solape real
      hubo con el kernel visible del frame.
    </p>
  `;
  explanationSection.body.appendChild(decisionCard);
  infoContainer.appendChild(explanationSection.section);

  const vetoSection = createDetailSection('Veto por candidato', {
    badge: `${rows.length} filas`,
    open: false,
  });

  if (!activeRows.length) {
    vetoSection.body.innerHTML = '<div class="empty-state">No hay filas por candidato visibles para este nodo.</div>';
    valueContainer.appendChild(vetoSection.section);
    return;
  }

  for (const [detId, detRows] of groupRowsByDetection(activeRows)) {
    const kept = detRows.filter((row) => Number(row.decision_keep) === 1).length;
    const card = document.createElement('div');
    card.className = 'detail-card';
    const header = document.createElement('div');
    header.className = 'detail-card-head';
    header.innerHTML = `
      <strong>det ${detId}</strong>
      <span class="badge">${kept} sobreviven · ${detRows.length - kept} caen</span>
    `;
    card.appendChild(header);

    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>objeto</th>
          <th>plausible conocido</th>
          <th>keep</th>
          <th>razón conocida</th>
          <th>veto</th>
          <th>gate</th>
          <th>sim</th>
          <th>sets</th>
          <th>bonus</th>
          <th>support</th>
          <th>quality</th>
          <th>compat</th>
          <th>kernel</th>
          <th>hyp</th>
          <th>episodes</th>
          <th>kernel src</th>
          <th>kernel size</th>
          <th>expected</th>
          <th>hits</th>
          <th>hit ratio</th>
          <th>maturity</th>
        </tr>
      </thead>
      <tbody>
        ${detRows.map((row) => `
          <tr>
            <td>${objectLabelForId(row.object_id)}</td>
            <td>${Number(row.known_plausible_keep) === 1 ? 'si' : 'no'}</td>
            <td>${Number(row.decision_keep) === 1 ? 'si' : 'no'}</td>
            <td>${pretty(row.known_plausible_reason)}</td>
            <td>${pretty(row.veto_reason)}</td>
            <td>${pretty(row.gate_reason)}</td>
            <td>${pretty(row.score_sim)}</td>
            <td>${pretty(row.score_sets)}</td>
            <td>${pretty(row.bonus_sets)}</td>
            <td>${pretty(row.support_sets)}</td>
            <td>${pretty(row.quality_sets)}</td>
            <td>${pretty(row.compat_rel)}</td>
            <td>${pretty(row.kernel_rel)}</td>
            <td>${pretty(row.hyp_rel)}</td>
            <td>${pretty(row.local_ctx_episode_count)}</td>
            <td>${pretty(row.local_ctx_kernel_source)}</td>
            <td>${pretty(row.local_ctx_kernel_size)}</td>
            <td>${pretty(row.local_ctx_expected_count)}</td>
            <td>${pretty(row.local_ctx_hit_count)}</td>
            <td>${pretty(row.local_ctx_hit_ratio)}</td>
            <td>${pretty(row.local_ctx_maturity)}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    card.appendChild(tableWrap);
    vetoSection.body.appendChild(card);
  }

  valueContainer.appendChild(vetoSection.section);
}

function renderFinalScoreTablesSection(infoContainer, valueContainer) {
  const scoreRun = getNodeRun('shape.final_score_tables');
  const detectionRows = scoreRun?.detection_rows || [];
  const candidateRows = scoreRun?.candidate_rows || [];
  const activeDetectionRows = state.selectedDetId == null
    ? detectionRows
    : detectionRows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));
  const activeCandidateRows = state.selectedDetId == null
    ? candidateRows
    : candidateRows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });

  const whatCard = document.createElement('div');
  whatCard.className = 'detail-card';
  whatCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
    <p>
      Este nodo materializa las tablas que de verdad consume la resolución
      global. Aquí ya solo aparece lo que sobrevivió al shaping anterior y
      cada score cumple una función distinta dentro del matching.
    </p>
    <p>
      La diferencia entre <code>score_sim</code>, <code>score_assign</code> y
      <code>score_final</code> no es cosmética: Hungarian optimiza
      <code>score_assign</code>, mientras que la aceptación dura del match y
      parte del post-assignment leen <code>score_final</code>.
    </p>
  `;
  explanationSection.body.appendChild(whatCard);

  const glossaryCard = document.createElement('div');
  glossaryCard.className = 'detail-card';
  glossaryCard.innerHTML = `
    <div class="detail-card-head"><strong>Cómo leer la tabla</strong></div>
    <p>
      <strong>sim</strong> es la base visual. <strong>assign</strong> es la
      señal estable usada por Hungarian para optimizar la asignación.
      <strong>final</strong> es la puntuación rica que luego se usa para
      aceptar o rechazar la asignación elegida y para el reporting posterior.
    </p>
    <p>
      <strong>sets</strong>, <strong>bonus sets</strong>,
      <strong>ctx local</strong> y <strong>ctx global</strong> muestran qué
      parte del empuje contextual viene de apoyo cercano al kernel y cuál viene
      de la lectura global de la clase.
    </p>
  `;
  explanationSection.body.appendChild(glossaryCard);
  infoContainer.appendChild(explanationSection.section);

  const summarySection = createDetailSection('Ranking final por detección', {
    badge: `${candidateRows.length} filas`,
    open: false,
  });

  if (!activeCandidateRows.length) {
    summarySection.body.innerHTML = '<div class="empty-state">No hay filas visibles en la tabla final de scores.</div>';
    valueContainer.appendChild(summarySection.section);
    return;
  }

  const bestByDet = new Map();
  activeDetectionRows.forEach((row) => {
    bestByDet.set(Number(row.det_id), row);
  });

  for (const [detId, detRows] of groupRowsByDetection(activeCandidateRows)) {
    detRows.sort((left, right) => Number(left.rank ?? 9999) - Number(right.rank ?? 9999));
    const detSummary = bestByDet.get(Number(detId));
    const card = document.createElement('div');
    card.className = 'detail-card';
    const header = document.createElement('div');
    header.className = 'detail-card-head';
    header.innerHTML = `
      <strong>det ${detId}</strong>
      <span class="badge">mejor ${detSummary?.best_object_id != null ? objectLabelForId(detSummary.best_object_id) : '—'} · final=${pretty(detSummary?.best_score_final)}</span>
    `;
    card.appendChild(header);

    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>objeto</th>
          <th>rank</th>
          <th>sim</th>
          <th>assign</th>
          <th>final</th>
          <th>sets</th>
          <th>bonus sets</th>
          <th>ctx local</th>
          <th>ctx global</th>
          <th>gate</th>
          <th>plausible conocido</th>
        </tr>
      </thead>
      <tbody>
        ${detRows.map((row) => `
          <tr>
            <td>${objectLabelForId(row.object_id)}</td>
            <td>${pretty(row.rank)}</td>
            <td>${pretty(row.score_sim)}</td>
            <td>${pretty(row.score_assign)}</td>
            <td>${pretty(row.score_final)}</td>
            <td>${pretty(row.score_sets)}</td>
            <td>${pretty(row.bonus_sets)}</td>
            <td>${pretty(row.score_ctx_local)}</td>
            <td>${pretty(row.score_ctx_global)}</td>
            <td>${pretty(row.gate_reason)}</td>
            <td>${Number(row.known_plausible_keep) === 1 ? 'si' : 'no'}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    card.appendChild(tableWrap);
    summarySection.body.appendChild(card);
  }

  valueContainer.appendChild(summarySection.section);
}

function renderHungarianSection(infoContainer, valueContainer) {
  const nodeRun = getNodeRun('resolve.hungarian');
  const detectionRows = nodeRun?.detection_rows || [];
  const candidateRows = nodeRun?.candidate_rows || [];
  const globalRows = nodeRun?.global_rows || [];
  const activeDetectionRows = state.selectedDetId == null
    ? detectionRows
    : detectionRows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));
  const activeCandidateRows = state.selectedDetId == null
    ? candidateRows
    : candidateRows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));
  const activeGlobalRows = state.selectedDetId == null
    ? globalRows
    : globalRows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });

  const whatCard = document.createElement('div');
  whatCard.className = 'detail-card';
  whatCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
    <p>
      Hungarian resuelve conjuntamente las detecciones que siguen vivas tras
      locks. La optimización compite contra columnas de objeto y, cuando están
      habilitadas, contra columnas dummy para poder dejar una detección fuera
      del matching conocido.
    </p>
    <p>
      La asignación elegida no se acepta sin más: si cae en un objeto real, el
      bloque todavía comprueba <code>match_thr</code> sobre
      <code>score_final</code> y <code>min_match_score</code> sobre
      <code>score_sim</code> antes de dejarla como match.
    </p>
  `;
  explanationSection.body.appendChild(whatCard);

  const readingCard = document.createElement('div');
  readingCard.className = 'detail-card';
  readingCard.innerHTML = `
    <div class="detail-card-head"><strong>Cómo leer la tabla</strong></div>
    <p>
      <strong>assigned kind</strong> enseña si la optimización cayó en un
      objeto real o en dummy. <strong>selected assign</strong> es el score con
      el que Hungarian optimizó; <strong>selected sim</strong> y
      <strong>selected final</strong> son los scores que luego se usan para
      validar la asignación elegida.
    </p>
    <p>
      <strong>final action</strong> resume el efecto real del nodo:
      <code>MATCH</code> si la asignación al objeto sobrevive a los umbrales,
      <code>CREATE</code> si cae en dummy o si un objeto real es rechazado por
      thresholds.
    </p>
  `;
  explanationSection.body.appendChild(readingCard);
  infoContainer.appendChild(explanationSection.section);

  const globalSection = createDetailSection('Salida global', {
    badge: `${activeGlobalRows.length} filas`,
    open: false,
  });
  if (!activeGlobalRows.length) {
    globalSection.body.innerHTML = '<div class="empty-state">No hay salidas globales visibles para Hungarian en este foco.</div>';
  } else {
    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>det</th>
          <th>salida</th>
          <th>objeto</th>
          <th>score final</th>
        </tr>
      </thead>
      <tbody>
        ${activeGlobalRows.map((row) => `
          <tr>
            <td>det ${pretty(row.det_id)}</td>
            <td>${String(row.kind || '').toUpperCase() || '—'}</td>
            <td>${row.object_id == null ? '—' : objectLabelForId(row.object_id)}</td>
            <td>${pretty(row.score_final)}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    globalSection.body.appendChild(tableWrap);
  }
  valueContainer.appendChild(globalSection.section);

  const perDetectionSection = createDetailSection('Resolución por detección', {
    badge: `${activeDetectionRows.length} filas`,
    open: false,
  });
  if (!activeDetectionRows.length) {
    perDetectionSection.body.innerHTML = '<div class="empty-state">No hay detecciones visibles para este nodo.</div>';
    valueContainer.appendChild(perDetectionSection.section);
    return;
  }

  activeDetectionRows
    .slice()
    .sort((left, right) => Number(left.det_id) - Number(right.det_id))
    .forEach((row) => {
      const detId = Number(row.det_id);
      const detCandidateRows = activeCandidateRows
        .filter((candidate) => Number(candidate.det_id) === detId)
        .sort((left, right) => Number(left.rank ?? 9999) - Number(right.rank ?? 9999));

      const card = document.createElement('div');
      card.className = 'detail-card';
      const header = document.createElement('div');
      header.className = 'detail-card-head';
      header.innerHTML = `
        <strong>det ${detId}</strong>
        <span class="badge">${pretty(row.final_action)} · ${pretty(row.reason)}</span>
      `;
      card.appendChild(header);

      const summary = document.createElement('p');
      summary.innerHTML = row.assigned_kind === 'OBJECT'
        ? `La optimización cae en ${objectLabelForId(row.assigned_object_id)}, pero el match solo sobrevive si pasa <code>match_thr</code> y <code>min_match_score</code>.`
        : 'La optimización no reserva un objeto real para esta detección y la salida se orienta a create.';
      card.appendChild(summary);

      const decisionWrap = document.createElement('div');
      decisionWrap.className = 'table-wrap';
      const decisionTable = document.createElement('table');
      decisionTable.className = 'detail-table';
      decisionTable.innerHTML = `
        <thead>
          <tr>
            <th>assigned kind</th>
            <th>columna</th>
            <th>objeto</th>
            <th>selected assign</th>
            <th>selected sim</th>
            <th>selected final</th>
            <th>dummy score</th>
            <th>match thr</th>
            <th>min sim</th>
            <th>final action</th>
            <th>reason</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>${pretty(row.assigned_kind)}</td>
            <td>${pretty(row.assigned_column)}</td>
            <td>${row.assigned_object_id == null ? '—' : objectLabelForId(row.assigned_object_id)}</td>
            <td>${pretty(row.selected_score_assign)}</td>
            <td>${pretty(row.selected_score_sim)}</td>
            <td>${pretty(row.selected_score_final)}</td>
            <td>${pretty(row.dummy_score)}</td>
            <td>${row.passes_match_thr ? 'si' : 'no'}</td>
            <td>${row.passes_min_match_score ? 'si' : 'no'}</td>
            <td>${pretty(row.final_action)}</td>
            <td>${pretty(row.reason)}</td>
          </tr>
        </tbody>
      `;
      decisionWrap.appendChild(decisionTable);
      card.appendChild(decisionWrap);

      if (detCandidateRows.length) {
        const rankWrap = document.createElement('div');
        rankWrap.className = 'table-wrap';
        const rankTable = document.createElement('table');
        rankTable.className = 'detail-table';
        rankTable.innerHTML = `
          <thead>
            <tr>
              <th>objeto</th>
              <th>rank</th>
              <th>selected</th>
              <th>assign</th>
              <th>sim</th>
              <th>final</th>
            </tr>
          </thead>
          <tbody>
            ${detCandidateRows.map((candidate) => `
              <tr>
                <td>${objectLabelForId(candidate.object_id)}</td>
                <td>${pretty(candidate.rank)}</td>
                <td>${candidate.selected ? 'si' : 'no'}</td>
                <td>${pretty(candidate.score_assign)}</td>
                <td>${pretty(candidate.score_sim)}</td>
                <td>${pretty(candidate.score_final)}</td>
              </tr>
            `).join('')}
          </tbody>
        `;
        rankWrap.appendChild(rankTable);
        card.appendChild(rankWrap);
      }

      perDetectionSection.body.appendChild(card);
    });

  valueContainer.appendChild(perDetectionSection.section);
}

function renderFinalDecisionPackSection(infoContainer, valueContainer) {
  const nodeRun = getNodeRun('post.final_decision_pack');
  const detectionRows = nodeRun?.detection_rows || [];
  const activeRows = state.selectedDetId == null
    ? detectionRows
    : detectionRows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));
  const values = nodeRun?.values || {};

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });

  const whatCard = document.createElement('div');
  whatCard.className = 'detail-card';
  whatCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
    <p>
      Este nodo no recalcula similitud ni vuelve a resolver la clase. Lo que
      hace es arbitrar la precedencia final entre las salidas ya producidas por
      el postproceso: match, create, ambiguous y provisional.
    </p>
    <p>
      Su trabajo consiste en garantizar que cada detección llegue a
      <strong>Outcome final</strong> con un único bucket semántico y sin
      conflictos entre ramas paralelas del post-assignment.
    </p>
  `;
  explanationSection.body.appendChild(whatCard);

  const readingCard = document.createElement('div');
  readingCard.className = 'detail-card';
  readingCard.innerHTML = `
    <div class="detail-card-head"><strong>Cómo leer la tabla</strong></div>
    <p>
      Las columnas <strong>input *</strong> enseñan de qué ramas venía viva la
      detección antes del arbitraje final. <strong>blocked *</strong> muestra
      si un match o un create iniciales fueron desplazados por una salida con
      más prioridad. <strong>final bucket</strong> es la única categoría que
      sobrevive al empaquetado.
    </p>
  `;
  explanationSection.body.appendChild(readingCard);

  const inputsCard = document.createElement('div');
  inputsCard.className = 'detail-card';
  inputsCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué entra de verdad</strong></div>
    <p>
      Este nodo recibe cuatro bolsas ya producidas antes:
      <strong>matches</strong>, <strong>creates</strong>,
      <strong>ambiguous</strong> y <strong>provisional</strong>. No crea una
      rama nueva ni recalcula scores; solo decide qué bucket conserva cada
      detección cuando varias ramas siguen vivas a la vez.
    </p>
  `;
  explanationSection.body.appendChild(inputsCard);
  infoContainer.appendChild(explanationSection.section);

  const summarySection = createDetailSection('Resumen de clase', {
    open: false,
  });
  summarySection.body.innerHTML = `
    <div class="detail-card">
      <div class="detail-card-head"><strong>Conteos de entrada y salida</strong></div>
      <p>
        entrada: match=${pretty(values.input_match_count ?? 0)} · create=${pretty(values.input_create_count ?? 0)} · ambiguous=${pretty(values.input_ambiguous_count ?? 0)} · provisional=${pretty(values.input_provisional_count ?? 0)}
      </p>
      <p>
        salida: match=${pretty(values.final_match_count ?? 0)} · create=${pretty(values.final_create_count ?? 0)} · ambiguous=${pretty(values.final_ambiguous_count ?? 0)} · provisional=${pretty(values.final_provisional_count ?? 0)}
      </p>
    </div>
  `;
  valueContainer.appendChild(summarySection.section);

  const perDetectionSection = createDetailSection('Empaquetado por detección', {
    badge: `${activeRows.length} filas`,
    open: false,
  });
  if (!activeRows.length) {
    perDetectionSection.body.innerHTML = '<div class="empty-state">No hay detecciones visibles para este nodo.</div>';
    valueContainer.appendChild(perDetectionSection.section);
    return;
  }

  activeRows
    .slice()
    .sort((left, right) => Number(left.det_id) - Number(right.det_id))
    .forEach((row) => {
      const card = document.createElement('div');
      card.className = 'detail-card';

      const header = document.createElement('div');
      header.className = 'detail-card-head';
      header.innerHTML = `
        <strong>det ${pretty(row.det_id)}</strong>
        <span class="badge">${pretty(row.final_bucket)} · ${pretty(row.reason)}</span>
      `;
      card.appendChild(header);

      const summary = document.createElement('p');
      summary.innerHTML = `
        objeto final: <strong>${row.final_object_id == null ? '—' : objectLabelForId(row.final_object_id)}</strong>
        · score: <strong>${pretty(row.final_score)}</strong>
        · arbitraje: <strong>${row.blocked_match || row.blocked_create ? 'con conflicto previo' : 'sin conflicto'}</strong>
      `;
      card.appendChild(summary);

      const inputsWrap = document.createElement('div');
      inputsWrap.className = 'table-wrap';
      const inputsTable = document.createElement('table');
      inputsTable.className = 'detail-table';
      inputsTable.innerHTML = `
        <thead>
          <tr>
            <th>in match</th>
            <th>in create</th>
            <th>in ambiguous</th>
            <th>in provisional</th>
            <th>blocked match</th>
            <th>blocked create</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>${row.input_match ? 'si' : 'no'}</td>
            <td>${row.input_create ? 'si' : 'no'}</td>
            <td>${row.input_ambiguous ? 'si' : 'no'}</td>
            <td>${row.input_provisional ? 'si' : 'no'}</td>
            <td>${row.blocked_match ? 'si' : 'no'}</td>
            <td>${row.blocked_create ? 'si' : 'no'}</td>
          </tr>
        </tbody>
      `;
      inputsWrap.appendChild(inputsTable);
      card.appendChild(inputsWrap);

      const checksWrap = document.createElement('div');
      checksWrap.className = 'table-wrap';
      const checksTable = document.createElement('table');
      checksTable.className = 'detail-table';
      const checks = Array.isArray(row.checks) ? row.checks : [];
      checksTable.innerHTML = `
        <thead>
          <tr>
            <th>check</th>
            <th>lhs</th>
            <th>op</th>
            <th>rhs</th>
            <th>passed</th>
            <th>effect</th>
          </tr>
        </thead>
        <tbody>
          ${checks.map((check) => `
            <tr>
              <td>${pretty(check.label || check.id)}</td>
              <td>${pretty(check.lhs)}</td>
              <td>${pretty(check.op)}</td>
              <td>${pretty(check.rhs)}</td>
              <td>${check.passed ? 'si' : 'no'}</td>
              <td>${pretty(check.effect)}</td>
            </tr>
          `).join('')}
        </tbody>
      `;
      checksWrap.appendChild(checksTable);
      card.appendChild(checksWrap);

      perDetectionSection.body.appendChild(card);
    });
  valueContainer.appendChild(perDetectionSection.section);
}

function renderOutcomeFinalizeSection(infoContainer, valueContainer) {
  const nodeRun = getNodeRun('outcome.finalize');
  const detectionRows = nodeRun?.detection_rows || [];
  const activeRows = state.selectedDetId == null
    ? detectionRows
    : detectionRows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));
  const values = nodeRun?.values || {};

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });

  const whatCard = document.createElement('div');
  whatCard.className = 'detail-card';
  whatCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
    <p>
      Outcome final no decide el matching por sí solo. Su función es anotar en
      cada <code>SimilarityReport</code> la salida legible que ya quedó fijada
      por el pipeline: decisión, razón, objeto final y soportes adicionales si
      el caso terminó como ambiguo o provisional.
    </p>
  `;
  explanationSection.body.appendChild(whatCard);

  const notDecisionCard = document.createElement('div');
  notDecisionCard.className = 'detail-card';
  notDecisionCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué no hace</strong></div>
    <p>
      No reabre Hungarian, no arbitra precedencias y no decide si algo es
      ambiguous o provisional. Todo eso ya viene fijado desde
      <strong>Empaquetado final</strong>. Aquí solo se copia esa semántica al
      <code>SimilarityReport</code> que consumirá update.
    </p>
  `;
  explanationSection.body.appendChild(notDecisionCard);

  const readingCard = document.createElement('div');
  readingCard.className = 'detail-card';
  readingCard.innerHTML = `
    <div class="detail-card-head"><strong>Cómo leer la tabla</strong></div>
    <p>
      <strong>match source</strong> ayuda a distinguir si un match final llega
      de la asociación principal o de una resolución posterior. Las columnas de
      <strong>ambiguous</strong> y <strong>provisional</strong> no abren ramas
      nuevas: solo muestran qué soporte quedó anotado para que update y el
      visor puedan entender por qué la detección no terminó como match limpio.
    </p>
  `;
  explanationSection.body.appendChild(readingCard);
  infoContainer.appendChild(explanationSection.section);

  const summarySection = createDetailSection('Resumen de clase', {
    open: false,
  });
  summarySection.body.innerHTML = `
    <div class="detail-card">
      <div class="detail-card-head"><strong>Conteo final anotado</strong></div>
      <p>${Object.entries(values.decision_counts || {})
        .map(([label, count]) => `${label}=${pretty(count)}`)
        .join(' · ') || 'sin conteos disponibles'}</p>
    </div>
  `;
  valueContainer.appendChild(summarySection.section);

  const perDetectionSection = createDetailSection('Outcome por detección', {
    badge: `${activeRows.length} filas`,
    open: false,
  });
  if (!activeRows.length) {
    perDetectionSection.body.innerHTML = '<div class="empty-state">No hay detecciones visibles para este nodo.</div>';
    valueContainer.appendChild(perDetectionSection.section);
    return;
  }

  activeRows
    .slice()
    .sort((left, right) => Number(left.det_id) - Number(right.det_id))
    .forEach((row) => {
      const card = document.createElement('div');
      card.className = 'detail-card';
      const header = document.createElement('div');
      header.className = 'detail-card-head';
      header.innerHTML = `
        <strong>det ${pretty(row.det_id)}</strong>
        <span class="badge">${pretty(row.final_decision)} · ${pretty(row.final_reason)}</span>
      `;
      card.appendChild(header);

      const summary = document.createElement('p');
      summary.innerHTML = `
        objeto final: <strong>${row.final_object_id == null ? '—' : objectLabelForId(row.final_object_id)}</strong>
        · score final: <strong>${pretty(row.final_score)}</strong>
        · source: <strong>${pretty(row.match_source || '—')}</strong>
        · anotación: <strong>${row.final_decision === 'MATCH' ? 'match limpio o reasignado ya resuelto' : 'salida no-match ya fijada antes'}</strong>
      `;
      card.appendChild(summary);

      const tableWrap = document.createElement('div');
      tableWrap.className = 'table-wrap';
      const table = document.createElement('table');
      table.className = 'detail-table';
      table.innerHTML = `
        <thead>
          <tr>
            <th>ambiguous ids</th>
            <th>ambiguous scores</th>
            <th>prov support ids</th>
            <th>prov support scores</th>
            <th>prov blocked ids</th>
            <th>prov blocked scores</th>
            <th>prov related ids</th>
            <th>prov related scores</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>${pretty(row.ambiguous_candidate_ids)}</td>
            <td>${objectScoreMapText(row.ambiguous_candidate_scores)}</td>
            <td>${pretty(row.provisional_support_ids)}</td>
            <td>${objectScoreMapText(row.provisional_support_scores)}</td>
            <td>${pretty(row.provisional_blocked_known_ids)}</td>
            <td>${objectScoreMapText(row.provisional_blocked_known_scores)}</td>
            <td>${pretty(row.provisional_related_known_ids)}</td>
            <td>${objectScoreMapText(row.provisional_related_known_scores)}</td>
          </tr>
        </tbody>
      `;
      tableWrap.appendChild(table);
      card.appendChild(tableWrap);
      perDetectionSection.body.appendChild(card);
    });

  valueContainer.appendChild(perDetectionSection.section);
}

function renderAssignmentAmbiguitySection(infoContainer, valueContainer) {
  const nodeRun = getNodeRun('post.assignment_ambiguity');
  const globalRows = nodeRun?.global_rows || [];
  const activeRows = state.selectedDetId == null
    ? globalRows
    : globalRows.filter((row) => (row.component_det_ids || []).map((value) => Number(value)).includes(Number(state.selectedDetId)));
  const values = nodeRun?.values || {};

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });
  explanationSection.body.innerHTML = `
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
      <p>
        Este nodo compara asignaciones completas dentro de cada componente
        conectado de detecciones y objetos. No revisa un candidato aislado:
        mira si el mejor reparto global y la segunda mejor alternativa quedan
        demasiado cerca y si realmente intercambian detecciones.
      </p>
    </div>
    <div class="detail-card">
      <div class="detail-card-head"><strong>Cómo leer la tabla</strong></div>
      <p>
      <strong>gap</strong> mide la distancia entre la mejor y la segunda
      asignación completa. <strong>ambiguous dets</strong> enseña qué
      detecciones cambian de objeto entre ambas y por eso quedan realmente
      discutidas.
    </p>
  </div>
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué entra y qué sale</strong></div>
      <p>
        Entra solo el subconjunto de detecciones que ya tienen match dentro de
        la clase. Sale una lectura por componente conectado: si hay o no una
        alternativa casi empatada y qué detecciones quedan realmente afectadas
        por ese empate.
      </p>
    </div>
  `;
  infoContainer.appendChild(explanationSection.section);

  const summarySection = createDetailSection('Resumen de clase', {
    open: false,
  });
  summarySection.body.innerHTML = `
    <div class="detail-card">
      <div class="detail-card-head"><strong>Componentes comparados</strong></div>
      <p>
        matched dets=${pretty(values.matched_det_count ?? 0)} · matched objects=${pretty(values.matched_object_count ?? 0)} · components=${pretty(values.component_count ?? 0)} · ambiguous components=${pretty(values.ambiguous_component_count ?? 0)}
      </p>
    </div>
  `;
  valueContainer.appendChild(summarySection.section);

  const section = createDetailSection('Componentes comparados', {
    badge: `${activeRows.length} filas`,
    open: false,
  });
  if (!activeRows.length) {
    section.body.innerHTML = '<div class="empty-state">No hay componentes ambiguos visibles en este foco.</div>';
    valueContainer.appendChild(section.section);
    return;
  }
  activeRows.forEach((row, index) => {
    const card = document.createElement('div');
    card.className = 'detail-card';

    const header = document.createElement('div');
    header.className = 'detail-card-head';
    header.innerHTML = `
      <strong>componente ${index + 1}</strong>
      <span class="badge">${row.is_ambiguous ? 'AMBIGUO' : 'NO AMBIGUO'} · ${pretty(row.reason)}</span>
    `;
    card.appendChild(header);

    const summary = document.createElement('p');
    summary.innerHTML = `
      dets del componente: <strong>${detLabelsText(row.component_det_ids)}</strong>
      · objetos: <strong>${objectLabelsText(row.component_object_ids)}</strong>
      · dets realmente afectadas: <strong>${detLabelsText(row.ambiguous_det_ids)}</strong>
    `;
    card.appendChild(summary);

    const checksWrap = document.createElement('div');
    checksWrap.className = 'table-wrap';
    const checksTable = document.createElement('table');
    checksTable.className = 'detail-table';
    const checks = Array.isArray(row.checks) ? row.checks : [];
    checksTable.innerHTML = `
      <thead>
        <tr>
          <th>check</th>
          <th>lhs</th>
          <th>op</th>
          <th>rhs</th>
          <th>passed</th>
          <th>effect</th>
        </tr>
      </thead>
      <tbody>
        ${checks.map((check) => `
          <tr>
            <td>${pretty(check.label || check.id)}</td>
            <td>${pretty(check.lhs)}</td>
            <td>${pretty(check.op)}</td>
            <td>${pretty(check.rhs)}</td>
            <td>${check.passed ? 'si' : 'no'}</td>
            <td>${pretty(check.effect)}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    checksWrap.appendChild(checksTable);
    card.appendChild(checksWrap);

    const assignmentsWrap = document.createElement('div');
    assignmentsWrap.className = 'table-wrap';
    const assignmentsTable = document.createElement('table');
    assignmentsTable.className = 'detail-table';
    assignmentsTable.innerHTML = `
      <thead>
        <tr>
          <th>gap</th>
          <th>best</th>
          <th>second</th>
          <th>asignación actual</th>
          <th>best assignment</th>
          <th>second assignment</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>${pretty(row.gap)}</td>
          <td>${pretty(row.best_score)}</td>
          <td>${pretty(row.second_score)}</td>
          <td>${assignmentText(row.current_assignment)}</td>
          <td>${assignmentText(row.best_assignment)}</td>
          <td>${assignmentText(row.second_assignment)}</td>
        </tr>
      </tbody>
    `;
    assignmentsWrap.appendChild(assignmentsTable);
    card.appendChild(assignmentsWrap);

    section.body.appendChild(card);
  });
  valueContainer.appendChild(section.section);
}

function renderIdentityStabilitySection(infoContainer, valueContainer) {
  const nodeRun = getNodeRun('post.identity_stability');
  const rows = nodeRun?.detection_rows || [];
  const activeRows = state.selectedDetId == null
    ? rows
    : rows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });
  explanationSection.body.innerHTML = `
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
      <p>
        Revisa los matches que salen de la resolución global y comprueba si se
        pueden conservar, si deben remapearse o si conviene desviarlos a create
        para no consolidar un cambio de identidad frágil.
      </p>
    </div>
  `;
  infoContainer.appendChild(explanationSection.section);

  const section = createDetailSection('Estado por detección', {
    badge: `${activeRows.length} filas`,
    open: false,
  });
  if (!activeRows.length) {
    section.body.innerHTML = '<div class="empty-state">No hay filas visibles de identity stability.</div>';
    valueContainer.appendChild(section.section);
    return;
  }
  const tableWrap = document.createElement('div');
  tableWrap.className = 'table-wrap';
  const table = document.createElement('table');
  table.className = 'detail-table';
  table.innerHTML = `
    <thead>
      <tr>
        <th>det</th>
        <th>objeto inicial</th>
        <th>objeto final</th>
        <th>score inicial</th>
        <th>score final</th>
        <th>state</th>
        <th>origin mode</th>
      </tr>
    </thead>
    <tbody>
      ${activeRows.map((row) => `
        <tr>
          <td>det ${pretty(row.det_id)}</td>
          <td>${objectLabelForId(row.initial_object_id)}</td>
          <td>${row.final_object_id == null ? '—' : objectLabelForId(row.final_object_id)}</td>
          <td>${pretty(row.initial_score_final)}</td>
          <td>${pretty(row.final_score_final)}</td>
          <td>${pretty(row.state)}</td>
          <td>${pretty(row.origin_mode)}</td>
        </tr>
      `).join('')}
    </tbody>
  `;
  tableWrap.appendChild(table);
  section.body.appendChild(tableWrap);
  valueContainer.appendChild(section.section);
}

function renderCreateCompetitionSection(infoContainer, valueContainer) {
  const nodeRun = getNodeRun('post.create_competition');
  const globalRows = nodeRun?.global_rows || [];
  const activeRows = state.selectedDetId == null
    ? globalRows
    : globalRows.filter((row) => Number(row.create_det_id) === Number(state.selectedDetId) || Number(row.parent_det_id) === Number(state.selectedDetId));

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });
  explanationSection.body.innerHTML = `
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
      <p>
        Este nodo decide cuándo un create compite de verdad contra una opción
        conocida ya asignada. No genera candidatos ambiguos por sí solo: deja
        preparadas las competiciones committed-new que luego pasarán a la bolsa
        ambigua.
      </p>
    </div>
  `;
  infoContainer.appendChild(explanationSection.section);

  const section = createDetailSection('Competiciones serializadas', {
    badge: `${activeRows.length} filas`,
    open: false,
  });
  if (!activeRows.length) {
    section.body.innerHTML = '<div class="empty-state">No hay competiciones create-vs-known visibles en este foco.</div>';
    valueContainer.appendChild(section.section);
    return;
  }
  const tableWrap = document.createElement('div');
  tableWrap.className = 'table-wrap';
  const table = document.createElement('table');
  table.className = 'detail-table';
  table.innerHTML = `
    <thead>
      <tr>
        <th>parent det</th>
        <th>parent oid</th>
        <th>create det</th>
        <th>selected</th>
        <th>create score</th>
        <th>parent score</th>
        <th>min score</th>
        <th>gap max</th>
        <th>reason</th>
      </tr>
    </thead>
    <tbody>
      ${activeRows.map((row) => `
        <tr>
          <td>det ${pretty(row.parent_det_id)}</td>
          <td>${objectLabelForId(row.parent_oid)}</td>
          <td>det ${pretty(row.create_det_id)}</td>
          <td>${row.selected ? 'si' : 'no'}</td>
          <td>${pretty(row.create_score)}</td>
          <td>${pretty(row.parent_score)}</td>
          <td>${pretty(row.min_score)}</td>
          <td>${pretty(row.pair_gap_max)}</td>
          <td>${pretty(row.reason)}</td>
        </tr>
      `).join('')}
    </tbody>
  `;
  tableWrap.appendChild(table);
  section.body.appendChild(tableWrap);
  valueContainer.appendChild(section.section);
}

function renderAmbiguousTrackCandidatesSection(infoContainer, valueContainer) {
  const nodeRun = getNodeRun('post.ambiguous_track_candidates');
  const rows = nodeRun?.detection_rows || [];
  const activeRows = state.selectedDetId == null
    ? rows
    : rows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });
  explanationSection.body.innerHTML = `
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
      <p>
        Este nodo materializa la bolsa ambigua real que entra en la resolución
        temporal. Aquí confluyen tres fuentes distintas: la ambigüedad
        contextual de la policy, la identidad inestable y las competiciones
        committed-new.
      </p>
    </div>
    <div class="detail-card">
      <div class="detail-card-head"><strong>Cómo leer la tabla</strong></div>
      <p>
        <strong>selected source</strong> enseña qué fuente prevaleció para esa
        detección cuando varias generaban una entrada ambigua a la vez.
      </p>
    </div>
  `;
  infoContainer.appendChild(explanationSection.section);

  const section = createDetailSection('Bolsa ambigua por detección', {
    badge: `${activeRows.length} filas`,
    open: false,
  });
  if (!activeRows.length) {
    section.body.innerHTML = '<div class="empty-state">No se han materializado candidatos ambiguos en este foco.</div>';
    valueContainer.appendChild(section.section);
    return;
  }
  const tableWrap = document.createElement('div');
  tableWrap.className = 'table-wrap';
  const table = document.createElement('table');
  table.className = 'detail-table';
  table.innerHTML = `
    <thead>
      <tr>
        <th>det</th>
        <th>policy</th>
        <th>identity</th>
        <th>committed new</th>
        <th>selected source</th>
        <th>candidatos</th>
        <th>scores</th>
        <th>best</th>
        <th>gap</th>
        <th>reason</th>
      </tr>
    </thead>
    <tbody>
      ${activeRows.map((row) => `
        <tr>
          <td>det ${pretty(row.det_id)}</td>
          <td>${row.from_policy ? 'si' : 'no'}</td>
          <td>${row.from_identity_stability ? 'si' : 'no'}</td>
          <td>${row.from_committed_new ? 'si' : 'no'}</td>
          <td>${pretty(row.selected_source)}</td>
          <td>${objectLabelsText(row.candidate_ids)}</td>
          <td>${objectScoreMapText(row.candidate_scores)}</td>
          <td>${pretty(row.best_score)}</td>
          <td>${pretty(row.score_gap)}</td>
          <td>${pretty(row.reason)}</td>
        </tr>
      `).join('')}
    </tbody>
  `;
  tableWrap.appendChild(table);
  section.body.appendChild(tableWrap);
  valueContainer.appendChild(section.section);
}

function renderKnownSetDistanceSection(infoContainer, valueContainer) {
  const nodeRun = getNodeRun('post.known_set_distance_disambiguation');
  const globalRows = nodeRun?.global_rows || [];
  const activeRows = state.selectedDetId == null
    ? globalRows
    : globalRows.filter((row) => (
      (row.det_ids || []).map((value) => Number(value)).includes(Number(state.selectedDetId))
      || (row.det_pair || []).map((value) => Number(value)).includes(Number(state.selectedDetId))
      || (row.input_det_ids || []).map((value) => Number(value)).includes(Number(state.selectedDetId))
      || (row.resolved_det_ids || []).map((value) => Number(value)).includes(Number(state.selectedDetId))
      || (row.remaining_det_ids || []).map((value) => Number(value)).includes(Number(state.selectedDetId))
      || (row.pass_input_det_ids || []).map((value) => Number(value)).includes(Number(state.selectedDetId))
      || (row.pass_resolved_det_ids || []).map((value) => Number(value)).includes(Number(state.selectedDetId))
      || (row.pass_remaining_det_ids || []).map((value) => Number(value)).includes(Number(state.selectedDetId))
    ));
  const passRows = activeRows.filter((row) => String(row.row_type || '') === 'pass_summary');
  const componentRows = activeRows.filter((row) => String(row.row_type || 'component') === 'component');
  const pairRows = activeRows.filter((row) => String(row.row_type || '') === 'pair_anchor');

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });
  explanationSection.body.innerHTML = `
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
      <p>
        Known-set-distance puede iterar varias pasadas sobre componentes
        ambiguos. Cada componente evalúa evidencia mínima, score de asignación,
        core score, core gap y gap final antes de resolver o dejar el caso
        ambiguo.
      </p>
    </div>
    <div class="detail-card">
      <div class="detail-card-head"><strong>Cómo leer la separación interna</strong></div>
      <p>
        <strong>Pasadas</strong> enseña qué bolsa ambigua entra en cada
        iteración y qué detecciones salen resueltas o siguen vivas.
        <strong>Componentes</strong> muestra la evaluación fuerte por
        subproblema ambiguo. <strong>Pair anchors</strong> deja visibles los
        anclajes discriminativos que el desambiguador usó como apoyo adicional.
      </p>
    </div>
  `;
  infoContainer.appendChild(explanationSection.section);

  const passSection = createDetailSection('Pasadas del desambiguador', {
    badge: `${passRows.length} filas`,
    open: false,
  });
  if (!passRows.length) {
    passSection.body.innerHTML = '<div class="empty-state">No hay resumen por pasada visible en este foco.</div>';
  } else {
    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>pass</th>
          <th>input dets</th>
          <th>componentes</th>
          <th>pair anchors</th>
          <th>resolved dets</th>
          <th>remaining dets</th>
        </tr>
      </thead>
      <tbody>
        ${passRows.map((row) => `
          <tr>
            <td>${pretty(row.pass_index)}</td>
            <td>${detLabelsText(row.input_det_ids)}</td>
            <td>${pretty(row.component_count)}</td>
            <td>${pretty(row.pair_anchor_count)}</td>
            <td>${detLabelsText(row.resolved_det_ids)}</td>
            <td>${detLabelsText(row.remaining_det_ids)}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    passSection.body.appendChild(tableWrap);
  }
  valueContainer.appendChild(passSection.section);

  const componentsSection = createDetailSection('Componentes evaluados', {
    badge: `${componentRows.length} filas`,
    open: false,
  });
  if (!componentRows.length) {
    componentsSection.body.innerHTML = '<div class="empty-state">No hay componentes serializados para known-set-distance en este foco.</div>';
  } else {
    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>pass</th>
          <th>input dets</th>
          <th>dets</th>
          <th>candidate union</th>
          <th>status</th>
          <th>reason</th>
          <th>best</th>
          <th>core score</th>
          <th>core gap</th>
          <th>gap</th>
          <th>resolved</th>
          <th>remaining</th>
        </tr>
      </thead>
      <tbody>
        ${componentRows.map((row) => `
          <tr>
            <td>${pretty(row.pass_index)}</td>
            <td>${detLabelsText(row.pass_input_det_ids)}</td>
            <td>${detLabelsText(row.det_ids)}</td>
            <td>${objectLabelsText(row.candidate_union)}</td>
            <td>${pretty(row.status)}</td>
            <td>${pretty(row.reason)}</td>
            <td>${pretty(row.best_score)}</td>
            <td>${pretty(row.core_score)}</td>
            <td>${pretty(row.core_gap)}</td>
            <td>${pretty(row.gap)}</td>
            <td>${detLabelsText(row.pass_resolved_det_ids)}</td>
            <td>${detLabelsText(row.pass_remaining_det_ids)}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    componentsSection.body.appendChild(tableWrap);
  }
  valueContainer.appendChild(componentsSection.section);

  const pairSection = createDetailSection('Pair anchors', {
    badge: `${pairRows.length} filas`,
    open: false,
  });
  if (!pairRows.length) {
    pairSection.body.innerHTML = '<div class="empty-state">No hay pair anchors serializados en este foco.</div>';
  } else {
    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>pass</th>
          <th>det pair</th>
          <th>anchor pair</th>
          <th>score</th>
          <th>reason</th>
        </tr>
      </thead>
      <tbody>
        ${pairRows.map((row) => `
          <tr>
            <td>${pretty(row.pass_index)}</td>
            <td>${detLabelsText(row.det_pair)}</td>
            <td>${objectLabelsText(row.anchor_pair)}</td>
            <td>${pretty(row.score)}</td>
            <td>${pretty(row.reason)}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    pairSection.body.appendChild(tableWrap);
  }
  valueContainer.appendChild(pairSection.section);
}

function renderProvisionalReconciliationSection(infoContainer, valueContainer) {
  const nodeRun = getNodeRun('post.provisional_reconciliation');
  const rows = nodeRun?.detection_rows || [];
  const activeRows = state.selectedDetId == null
    ? rows
    : rows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });
  explanationSection.body.innerHTML = `
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
      <p>
        Relee creates y ambigüedades que siguen vivas tras known-set-distance y
        decide si deben quedarse como create, promoverse a ambiguous o salir
        como provisional_new/provisional_parent.
      </p>
    </div>
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué entra realmente aquí</strong></div>
      <p>
        La policy no trabaja a ciegas: parte del estado temporal del report,
        del foco conocido soportado, de los modos de contexto calculados y de
        una tabla interna de candidatos temporales. Esta vista enseña esas
        piezas tal cual quedaron serializadas.
      </p>
    </div>
  `;
  infoContainer.appendChild(explanationSection.section);

  const section = createDetailSection('Resolución temporal por detección', {
    badge: `${activeRows.length} filas`,
    open: false,
  });
  if (!activeRows.length) {
    section.body.innerHTML = '<div class="empty-state">No hay decisiones temporales visibles en este foco.</div>';
    valueContainer.appendChild(section.section);
    return;
  }
  activeRows
    .slice()
    .sort((left, right) => Number(left.det_id) - Number(right.det_id))
    .forEach((row) => {
      const card = document.createElement('div');
      card.className = 'detail-card';

      const header = document.createElement('div');
      header.className = 'detail-card-head';
      header.innerHTML = `
        <strong>det ${pretty(row.det_id)}</strong>
        <span class="badge">${pretty(row.decision_kind)} · ${pretty(row.reason)}</span>
      `;
      card.appendChild(header);

      const summary = document.createElement('p');
      summary.innerHTML = `
        temporal status: <strong>${pretty(row.temporal_status || '—')}</strong>
        · final kind: <strong>${pretty(row.final_kind || '—')}</strong>
        · focus source: <strong>${pretty(row.focus_source || '—')}</strong>
        · context: <strong>${pretty(row.context_mode || '—')}</strong>
        / <strong>${pretty(row.support_mode || '—')}</strong>
        · relation: <strong>${pretty(row.relation || '—')}</strong>
      `;
      card.appendChild(summary);

      const summaryWrap = document.createElement('div');
      summaryWrap.className = 'table-wrap';
      const summaryTable = document.createElement('table');
      summaryTable.className = 'detail-table';
      summaryTable.innerHTML = `
        <thead>
          <tr>
            <th>best obj</th>
            <th>best score</th>
            <th>top supported</th>
            <th>top score</th>
            <th>known ctx</th>
            <th>visual fallback</th>
            <th>known blocked</th>
            <th>status blocked</th>
            <th>parent status ok</th>
            <th>parent ok</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>${row.best_object_id == null ? '—' : objectLabelForId(row.best_object_id)}</td>
            <td>${pretty(row.best_score)}</td>
            <td>${row.top_supported_object_id == null ? '—' : objectLabelForId(row.top_supported_object_id)}</td>
            <td>${pretty(row.top_supported_score)}</td>
            <td>${row.has_known_context ? 'si' : 'no'}</td>
            <td>${row.visual_fallback_ok ? 'si' : 'no'}</td>
            <td>${row.known_blocked_ok ? 'si' : 'no'}</td>
            <td>${row.status_not_allowed ? 'si' : 'no'}</td>
            <td>${row.provisional_parent_status_ok ? 'si' : 'no'}</td>
            <td>${row.provisional_parent_ok ? 'si' : 'no'}</td>
          </tr>
        </tbody>
      `;
      summaryWrap.appendChild(summaryTable);
      card.appendChild(summaryWrap);

      const supportWrap = document.createElement('div');
      supportWrap.className = 'table-wrap';
      const supportTable = document.createElement('table');
      supportTable.className = 'detail-table';
      supportTable.innerHTML = `
        <thead>
          <tr>
            <th>support ids</th>
            <th>support scores</th>
            <th>blocked ids</th>
            <th>blocked scores</th>
            <th>related ids</th>
            <th>related scores</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>${objectLabelsText(row.support_known_ids)}</td>
            <td>${objectScoreMapText(row.support_known_scores)}</td>
            <td>${objectLabelsText(row.blocked_known_ids)}</td>
            <td>${objectScoreMapText(row.blocked_known_scores)}</td>
            <td>${objectLabelsText(row.related_known_ids)}</td>
            <td>${objectScoreMapText(row.related_known_scores)}</td>
          </tr>
        </tbody>
      `;
      supportWrap.appendChild(supportTable);
      card.appendChild(supportWrap);

      const candidateRows = Array.isArray(row.candidate_rows) ? row.candidate_rows : [];
      if (candidateRows.length) {
        const candidateLabel = document.createElement('p');
        candidateLabel.innerHTML = '<strong>Filas candidatas internas</strong>';
        card.appendChild(candidateLabel);

        const candidatesWrap = document.createElement('div');
        candidatesWrap.className = 'table-wrap';
        const candidatesTable = document.createElement('table');
        candidatesTable.className = 'detail-table';
        candidatesTable.innerHTML = `
          <thead>
            <tr>
              <th>object</th>
              <th>temp</th>
              <th>sim</th>
              <th>final</th>
              <th>ctx</th>
              <th>blocked</th>
              <th>keep final</th>
              <th>kp</th>
              <th>dk</th>
              <th>min</th>
              <th>gap</th>
              <th>why</th>
            </tr>
          </thead>
          <tbody>
            ${candidateRows.map((candidate) => `
              <tr>
                <td>${candidate.object_id == null ? '—' : objectLabelForId(candidate.object_id)}</td>
                <td>${pretty(candidate.temp_score)}</td>
                <td>${pretty(candidate.score_sim)}</td>
                <td>${pretty(candidate.score_final)}</td>
                <td>${candidate.support_ctx ? 'si' : 'no'}</td>
                <td>${candidate.blocked ? 'si' : 'no'}</td>
                <td>${candidate.support_final ? 'si' : 'no'}</td>
                <td>${candidate.known_plausible_keep ? 'si' : 'no'}</td>
                <td>${candidate.decision_keep ? 'si' : 'no'}</td>
                <td>${candidate.min_ok ? 'si' : 'no'}</td>
                <td>${candidate.gap_ok ? 'si' : 'no'}</td>
                <td>${pretty(candidate.why)}</td>
              </tr>
            `).join('')}
          </tbody>
        `;
        candidatesWrap.appendChild(candidatesTable);
        card.appendChild(candidatesWrap);
      } else {
        const empty = document.createElement('div');
        empty.className = 'empty-state';
        empty.textContent = 'No hay filas candidatas serializadas para esta detección.';
        card.appendChild(empty);
      }

      section.body.appendChild(card);
    });

  valueContainer.appendChild(section.section);
}

function renderFinalAmbiguitySection(infoContainer, valueContainer) {
  const diagnosisRun = getNodeRun('outcome.final_ambiguity');
  const rows = diagnosisRun?.detection_rows || [];
  const activeRows = state.selectedDetId == null
    ? rows
    : rows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));

  const explanationSection = createDetailSection('Lectura del diagnóstico final', {
    open: false,
  });
  explanationSection.body.innerHTML = `
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué decide este bloque</strong></div>
      <p>
        Recalcula la claridad final del caso usando <code>score_final</code>.
        Ya no estamos leyendo el ranking visual bruto, sino el resultado
        operativo final con sus filtros y reinterpretaciones post-assignment.
      </p>
    </div>
  `;
  infoContainer.appendChild(explanationSection.section);

  const section = createDetailSection('Diagnóstico final por detección', {
    badge: `${rows.length} filas`,
    open: false,
  });
  if (!rows.length) {
    section.body.innerHTML = '<div class="empty-state">No hay diagnóstico final serializado en la traza actual.</div>';
    valueContainer.appendChild(section.section);
    return;
  }

  for (const [detId, detRows] of groupRowsByDetection(activeRows)) {
    const detRow = detRows[0];
    const card = document.createElement('div');
    card.className = 'detail-card';
    const header = document.createElement('div');
    header.className = 'detail-card-head';
    header.innerHTML = `
      <strong>det ${detId}</strong>
      <span class="badge">${detRow?.status || '—'} · ${detRow?.reason || 'sin razón'}</span>
    `;
    card.appendChild(header);
    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>estado</th>
          <th>razón</th>
          <th>s1</th>
          <th>s2</th>
          <th>gap</th>
          <th>n close</th>
          <th>confidence</th>
          <th>final decision</th>
          <th>final reason</th>
        </tr>
      </thead>
      <tbody>
        ${detRows.map((row) => `
          <tr>
            <td>${pretty(row.status)}</td>
            <td>${pretty(row.reason)}</td>
            <td>${pretty(row.s1)}</td>
            <td>${pretty(row.s2)}</td>
            <td>${pretty(row.gap)}</td>
            <td>${pretty(row.n_close)}</td>
            <td>${pretty(row.confidence)}</td>
            <td>${pretty(row.final_decision)}</td>
            <td>${pretty(row.final_reason)}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    card.appendChild(tableWrap);
    section.body.appendChild(card);
  }
  valueContainer.appendChild(section.section);
}

function renderVisualBuildCandidateSection(infoContainer, valueContainer) {
  const buildRun = getNodeRun('visual.build_candidates');
  const rows = buildRun?.candidate_rows || [];
  const activeRows = state.selectedDetId == null
    ? rows
    : rows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));
  const hasWeightData = rows.some((row) => (
    row.weight_eff_obj != null
    || row.weight_eff_bg != null
    || row.weight_eff_bg_partial != null
    || row.weight_eff_parts != null
  ));
  const nodeValues = buildRun?.values || {};

  const explanationSection = createDetailSection('Lectura del bloque', {
    open: false,
  });
  explanationSection.body.innerHTML = `
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
      <p>
        Este nodo construye la evidencia visual base por detección frente a los
        objetos conocidos de la misma clase. A partir de los descriptores
        observados y de la memoria visual guardada, calcula los canales
        <code>obj</code>, <code>bg</code>, <code>bg partial</code> y
        <code>parts</code>, y con ellos forma <code>score_sim</code>.
      </p>
    </div>
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué entra realmente</strong></div>
      <p>
        Entra la detección ya observada en percepción, con sus features
        visuales, y entra el snapshot de objetos de memoria comparables de esa
        clase. Todavía no intervienen Hungarian, la precedencia final ni la
        reinterpretación temporal.
      </p>
    </div>
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué no decide aún</strong></div>
      <p>
        Aquí no se decide <code>MATCH</code>, <code>NEW</code> ni
        <code>AMBIGUOUS</code>. El resultado es un ranking visual bruto por
        candidato que después leerán <strong>Diagnóstico visual</strong> y las
        fases de shaping y resolución.
      </p>
    </div>
    <div class="detail-card">
      <div class="detail-card-head"><strong>Qué deja preparado</strong></div>
      <p>
        Deja <strong>${pretty(nodeValues.candidate_count ?? rows.length)}</strong>
        filas candidatas visibles en la traza y, por detección, una base común
        de comparación visual para el resto del pipeline.
      </p>
    </div>
  `;
  infoContainer.appendChild(explanationSection.section);

  const descriptorsSection = createDetailSection('Descriptores', {
    open: false,
  });

  const objectDescriptorCard = document.createElement('div');
  objectDescriptorCard.className = 'detail-card';
  objectDescriptorCard.innerHTML = `
    <div class="detail-card-head"><strong>Descriptor de objeto</strong></div>
    <p>
      El objeto observado se convierte en varias representaciones del área del
      objeto: un descriptor global, un descriptor global recortado
      (<code>trimmed</code>) y, si está activado, un conjunto de descriptores
      por patch del objeto. Todos salen del mapa de features ya alineado a
      patches.
    </p>
    <p>
      Para pasar de muchos patches a un descriptor único, el pipeline agrega
      los patches del objeto con media. Si el modo ponderado está activo, cada
      patch pesa según su cobertura dentro del objeto. En la variante
      <code>trimmed</code>, primero se calcula un descriptor provisional y
      luego se repite la media solo con los patches más consistentes con él.
    </p>
    <p>
      Después, cada canal del objeto se compara contra la memoria guardada del
      candidato con una política común que mantenga comparables todos los
      candidatos en ese término. En el canal de objeto no cuenta solo la
      similitud observada: también influye la calidad del prototipo guardado
      con el que se está comparando, porque no todas las memorias del objeto
      tienen la misma fiabilidad.
    </p>
  `;
  descriptorsSection.body.appendChild(objectDescriptorCard);

  const bgDescriptorCard = document.createElement('div');
  bgDescriptorCard.className = 'detail-card';
  bgDescriptorCard.innerHTML = `
    <div class="detail-card-head"><strong>Descriptores de fondo</strong></div>
    <p>
      El fondo se construye alrededor del objeto a partir de anillos en
      patch-space. Primero se sanea la máscara del objeto y luego se definen
      regiones de fondo interior y exterior alrededor de él.
    </p>
    <p>
      A partir de esos anillos se agregan patches para formar un descriptor
      global de fondo interior y otro de fondo exterior. Ambos pueden usar
      media ponderada por patch, y luego se combinan con pesos fijos para dar
      un único descriptor global de fondo, que es <code>bg</code>.
    </p>
    <p>
      <code>bg partial</code> no colapsa todo el fondo en un único vector.
      Parte de los patches de cada anillo, los agrupa con <code>kmeans</code>,
      puede fusionar clusters muy parecidos y después selecciona varios
      prototipos representativos. Esos prototipos parciales son los que se
      comparan más tarde con la memoria de fondo del candidato. Así,
      <code>bg</code> resume el contexto global y <code>bg partial</code>
      conserva variantes locales del entorno.
    </p>
  `;
  descriptorsSection.body.appendChild(bgDescriptorCard);

  const partsDescriptorCard = document.createElement('div');
  partsDescriptorCard.className = 'detail-card';
  partsDescriptorCard.innerHTML = `
    <div class="detail-card-head"><strong>Descriptor de partes</strong></div>
    <p>
      Las partes se obtienen dentro del objeto observado, no fuera. El pipeline
      puede proponerlas de dos formas: agrupando patches del objeto
      (<code>kmeans</code>) o usando propuestas basadas en atención
      (<code>attention</code>).
    </p>
    <p>
      Cada parte propuesta se convierte en un descriptor agregando los patches
      de su región con media o media ponderada, y opcionalmente con versión
      <code>trimmed</code> igual que en objeto. Así se obtiene un conjunto de
      descriptores de parte, no un único descriptor global de partes.
    </p>
    <p>
      Luego se comparan contra las partes guardadas del objeto candidato. Si la
      detección no tiene suficiente soporte, pocas partes válidas o el objeto
      es demasiado pequeño, este canal pierde peso o deja de entrar.
    </p>
  `;
  descriptorsSection.body.appendChild(partsDescriptorCard);
  infoContainer.appendChild(descriptorsSection.section);

  const weightExplanationSection = createDetailSection('Explicación de pesos', {
    open: false,
  });

  const weightsIntro = document.createElement('div');
  weightsIntro.className = 'detail-card';
  weightsIntro.innerHTML = `
    <div class="detail-card-head"><strong>Pesos efectivos por término</strong></div>
    <p>
      Estos pesos indican cuánta confianza da el pipeline a cada fuente de
      evidencia antes de formar el <code>score_sim</code>. No miden parecido,
      sino cuánto debe influir cada término en la combinación final.
    </p>
    <p>
      El peso depende de dos cosas: que ese término tenga una comparación
      utilizable con el objeto guardado y que la evidencia observada en la
      detección actual sea suficientemente fiable. Si un término no tiene score
      válido, no entra en la combinación.
    </p>
    <p>
      En <code>obj</code> influyen sobre todo el soporte y la cobertura útil del
      objeto observado. En <code>bg</code> influyen cuántos parches útiles de
      fondo interior y exterior hay, y la calidad de la máscara de fondo. En
      <code>parts</code> influyen el soporte del objeto, cuántas partes válidas
      hay y la fracción de soporte de partes; si el objeto observado queda por
      debajo del mínimo exigido, ese término puede quedar anulado.
    </p>
    <p>
      El cálculo práctico es: primero se estima la fiabilidad de cada término,
      luego se aplica esa fiabilidad a sus pesos base y, al final, se
      normalizan solo entre los términos que sí participan en ese candidato.
    </p>
  `;
  weightExplanationSection.body.appendChild(weightsIntro);
  infoContainer.appendChild(weightExplanationSection.section);

  const { section, body } = createDetailSection('Scores', {
    badge: `${rows.length} filas`,
    open: false,
  });

  const intro = document.createElement('div');
  intro.className = 'detail-card';
  intro.innerHTML = `
    <div class="detail-card-head"><strong>Similitud visual</strong></div>
    <p>
      Aquí se ordenan los candidatos únicamente por <code>score_sim</code>.
      Cada canal compara descriptores mediante similitud coseno y luego resume
      esa comparación según su propio esquema: en objeto se conserva la mejor
      coincidencia del canal activo; en fondo global se usa la vista combinada
      de fondo; en fondo parcial y partes se agregan las mejores coincidencias
      parciales con una regla tipo top-k. Con esos términos ya colapsados, el
      bloque forma el score visual compuesto y ordena los candidatos.
    </p>
    <p>
      La salida no es solo una tabla de scores: por cada detección deja un
      ranking completo de candidatos, incluyendo mejor y segundo mejor caso.
      Esa salida alimenta directamente el bloque
      <strong>Diagnóstico visual</strong>, que decide si la evidencia es
      fuerte, ambigua o débil.
    </p>
  `;
  body.appendChild(intro);

  if (!rows.length) {
    body.innerHTML += '<div class="empty-state">No hay filas de similitud en la traza actual.</div>';
    valueContainer.appendChild(section);
    return;
  }

  for (const [detId, detRows] of groupRowsByDetection(activeRows)) {
    detRows.sort((a, b) => {
      const av = Number(a.score_sim ?? 0);
      const bv = Number(b.score_sim ?? 0);
      return bv - av;
    });

    const card = document.createElement('div');
    card.className = 'detail-card';
    const header = document.createElement('div');
    header.className = 'detail-card-head';
    header.innerHTML = `
      <strong>det ${detId}</strong>
      <span class="badge">${detRows.length} candidatos</span>
    `;
    card.appendChild(header);

    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    const headColumns = ['objeto', 'rank', 'sim', 'obj', 'bg', 'bg partial', 'parts'];
    table.innerHTML = `
      <thead>
        <tr>${headColumns.map((key) => `<th>${key}</th>`).join('')}</tr>
      </thead>
      <tbody>
        ${detRows.map((row, index) => `
          <tr>
            <td>${objectLabelForId(row.object_id)}</td>
            <td>${pretty(row.rank ?? (index + 1))}</td>
            <td>${pretty(row.score_sim)}</td>
            <td>${pretty(row.score_obj)}</td>
            <td>${pretty(row.score_bg)}</td>
            <td>${pretty(row.score_bg_partial)}</td>
            <td>${pretty(row.score_parts)}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    card.appendChild(tableWrap);
    body.appendChild(card);
  }

  valueContainer.appendChild(section);

  const weightsSection = createDetailSection('Pesos', {
    badge: `${rows.length} filas`,
    open: false,
  });

  if (!rows.length) {
    weightsSection.body.innerHTML += '<div class="empty-state">No hay pesos visibles en la traza actual.</div>';
    valueContainer.appendChild(weightsSection.section);
    return;
  }

  if (!hasWeightData) {
    weightsSection.body.innerHTML += '<div class="empty-state">La traza no serializa pesos efectivos para este nodo.</div>';
    valueContainer.appendChild(weightsSection.section);
    return;
  }

  for (const [detId, detRows] of groupRowsByDetection(activeRows)) {
    detRows.sort((a, b) => {
      const av = Number(a.score_sim ?? 0);
      const bv = Number(b.score_sim ?? 0);
      return bv - av;
    });

    const card = document.createElement('div');
    card.className = 'detail-card';
    const header = document.createElement('div');
    header.className = 'detail-card-head';
    header.innerHTML = `
      <strong>det ${detId}</strong>
      <span class="badge">${detRows.length} candidatos</span>
    `;
    card.appendChild(header);

    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    const headColumns = ['objeto', 'rank', 'peso obj', 'peso bg', 'peso bg partial', 'peso parts'];
    table.innerHTML = `
      <thead>
        <tr>${headColumns.map((key) => `<th>${key}</th>`).join('')}</tr>
      </thead>
      <tbody>
        ${detRows.map((row, index) => `
          <tr>
            <td>${objectLabelForId(row.object_id)}</td>
            <td>${pretty(row.rank ?? (index + 1))}</td>
            <td>${pretty(candidateNormalizedWeight(row, 'obj'))}</td>
            <td>${pretty(candidateNormalizedWeight(row, 'bg'))}</td>
            <td>${pretty(candidateNormalizedWeight(row, 'bg_partial'))}</td>
            <td>${pretty(candidateNormalizedWeight(row, 'parts'))}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    card.appendChild(tableWrap);
    weightsSection.body.appendChild(card);
  }

  valueContainer.appendChild(weightsSection.section);
}

function renderVisualReportDiagnosisSection(infoContainer, valueContainer) {
  const diagnosisRun = getNodeRun('visual.report_diagnosis');
  const rows = diagnosisRun?.detection_rows || [];
  const activeRows = state.selectedDetId == null
    ? rows
    : rows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));

  const explanationSection = createDetailSection('Lectura del diagnóstico', {
    open: false,
  });

  const summaryCard = document.createElement('div');
  summaryCard.className = 'detail-card';
  summaryCard.innerHTML = `
    <div class="detail-card-head"><strong>Qué decide este bloque</strong></div>
    <p>
      Este nodo no crea candidatos nuevos ni aplica contexto. Toma el ranking
      visual ya construido y resume qué tan clara es la evidencia para cada
      detección antes de entrar en los filtros de matching.
    </p>
    <p>
      La lectura se apoya sobre el mejor candidato visual
      (<code>s1</code>), el segundo mejor (<code>s2</code>), la separación
      entre ambos (<code>gap</code>) y cuántas alternativas quedan cerca del
      primer puesto (<code>n_close</code>). Con eso clasifica la detección
      como <code>STRONG</code>, <code>AMBIGUOUS</code> o <code>WEAK</code>.
    </p>
    <p>
      La consecuencia práctica es simple: cuanto más fuerte salga aquí, más
      estable llega esa detección a <strong>Filtro de detecciones válidas</strong>,
      <strong>Uso de contexto por reporte</strong> y al matching global. Si sale
      ambigua o débil, el resto del pipeline necesitará shortlist, contexto o
      resolución global para terminar de resolverla.
    </p>
  `;
  explanationSection.body.appendChild(summaryCard);

  const signalsCard = document.createElement('div');
  signalsCard.className = 'detail-card';
  signalsCard.innerHTML = `
    <div class="detail-card-head"><strong>Cómo leer las métricas</strong></div>
    <p>
      <code>s1</code> es la mejor similitud visual disponible para la
      detección. <code>s2</code> es la segunda alternativa. <code>gap</code>
      es la diferencia entre ambas, así que un valor alto indica una cabeza
      clara del ranking.
    </p>
    <p>
      <code>n_close</code> cuenta cuántos candidatos siguen demasiado cerca de
      <code>s1</code>. Aunque el mejor score sea alto, si hay varias
      alternativas cerca, la detección tiende a etiquetarse como ambigua.
    </p>
    <p>
      <code>confidence</code> es una lectura compacta de esa claridad visual.
      No sustituye a la decisión final, pero ayuda a entender si el ranking
      visual viene limpio o ya nace disputado.
    </p>
  `;
  explanationSection.body.appendChild(signalsCard);
  infoContainer.appendChild(explanationSection.section);

  const perDetectionSection = createDetailSection('Diagnóstico por detección', {
    badge: `${rows.length} filas`,
    open: false,
  });

  if (!rows.length) {
    perDetectionSection.body.innerHTML = '<div class="empty-state">No hay diagnóstico visual serializado en la traza actual.</div>';
    valueContainer.appendChild(perDetectionSection.section);
    return;
  }

  for (const [detId, detRows] of groupRowsByDetection(activeRows)) {
    const detRow = [...detRows].sort((a, b) => {
      const confidenceA = Number(a.confidence ?? -1);
      const confidenceB = Number(b.confidence ?? -1);
      return confidenceB - confidenceA;
    })[0];

    const card = document.createElement('div');
    card.className = 'detail-card';
    const header = document.createElement('div');
    header.className = 'detail-card-head';
    header.innerHTML = `
      <strong>det ${detId}</strong>
      <span class="badge">${detRow?.status || '—'} · ${detRow?.reason || 'sin razón'}</span>
    `;
    card.appendChild(header);

    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    table.innerHTML = `
      <thead>
        <tr>
          <th>estado</th>
          <th>razón</th>
          <th>s1</th>
          <th>s2</th>
          <th>gap</th>
          <th>n close</th>
          <th>confidence</th>
        </tr>
      </thead>
      <tbody>
        ${detRows.map((row) => `
          <tr>
            <td>${pretty(row.status)}</td>
            <td>${pretty(row.reason)}</td>
            <td>${pretty(row.s1)}</td>
            <td>${pretty(row.s2)}</td>
            <td>${pretty(row.gap)}</td>
            <td>${pretty(row.n_close)}</td>
            <td>${pretty(row.confidence)}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    card.appendChild(tableWrap);
    perDetectionSection.body.appendChild(card);
  }

  valueContainer.appendChild(perDetectionSection.section);
}

function renderNodeSpecificSections(nodeId, infoContainer, valueContainer) {
  if (String(nodeId) === 'prepare.reliable_visual_anchors') {
    renderReliableVisualAnchorsSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'context.neighbor_sets_hypotheses') {
    renderNeighborSetsHypothesesSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'context.sets_activation') {
    renderSetsActivationSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'shape.allow_for_report') {
    renderAllowForReportSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'shape.context_veto') {
    renderContextVetoSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'shape.final_score_tables') {
    renderFinalScoreTablesSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'resolve.hungarian') {
    renderHungarianSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'post.assignment_ambiguity') {
    renderAssignmentAmbiguitySection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'post.identity_stability') {
    renderIdentityStabilitySection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'post.create_competition') {
    renderCreateCompetitionSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'post.ambiguous_track_candidates') {
    renderAmbiguousTrackCandidatesSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'post.known_set_distance_disambiguation') {
    renderKnownSetDistanceSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'post.provisional_reconciliation') {
    renderProvisionalReconciliationSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'visual.build_candidates') {
    renderVisualBuildCandidateSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'visual.report_diagnosis') {
    renderVisualReportDiagnosisSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'post.final_decision_pack') {
    renderFinalDecisionPackSection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'outcome.final_ambiguity') {
    renderFinalAmbiguitySection(infoContainer, valueContainer);
    return;
  }
  if (String(nodeId) === 'outcome.finalize') {
    renderOutcomeFinalizeSection(infoContainer, valueContainer);
  }
}

function renderGenericNodeTab(nodeId) {
  const node = orderedNodes().find((item) => item.id === nodeId);
  const nodeRun = getNodeRun(nodeId);
  clearChildren(detailTabBadges);
  clearChildren(detailTabContent);

  detailTabKind.textContent = 'Detalle de nodo';
  detailTabTitle.textContent = nodeLabel(nodeId);
  ['fase: ' + moduleLabelForNode(nodeId), 'puerta: ' + nodeDoorState(nodeRun)].forEach((label) => {
    const badge = document.createElement('div');
    badge.className = 'badge';
    badge.textContent = label;
    detailTabBadges.appendChild(badge);
  });

  if (!nodeRun) {
    detailTabContent.innerHTML = '<div class="empty-state">Nodo no presente en esta traza.</div>';
    return;
  }

  const infoGroup = document.createElement('div');
  infoGroup.className = 'detail-group';
  infoGroup.innerHTML = '<div class="detail-group-title">Información y Explicaciones</div>';

  const valuesGroup = document.createElement('div');
  valuesGroup.className = 'detail-group';
  valuesGroup.innerHTML = '<div class="detail-group-title">Valores del Frame</div>';

  const participants = nodeRun.participants || {};
  renderIncomingDependenciesSection(nodeId, infoGroup);

  if (!nodeHasCustomNarrative(nodeId)) {
    const explanation = createDetailSection('Lectura del bloque', { open: false });

    const whatCard = document.createElement('div');
    whatCard.className = 'detail-card';
    whatCard.innerHTML = `
      <div class="detail-card-head"><strong>Qué hace este bloque</strong></div>
      <p>${generalDescription(nodeId)}</p>
    `;
    explanation.body.appendChild(whatCard);

    const whyCard = document.createElement('div');
    whyCard.className = 'detail-card';
    whyCard.innerHTML = `
      <div class="detail-card-head"><strong>Por qué existe</strong></div>
      <p>${nodeWhyText(nodeId)}</p>
    `;
    explanation.body.appendChild(whyCard);

    const preparesCard = document.createElement('div');
    preparesCard.className = 'detail-card';
    preparesCard.innerHTML = `
      <div class="detail-card-head"><strong>Qué deja preparado</strong></div>
      <p>${nodePreparesText(nodeId)}</p>
    `;
    explanation.body.appendChild(preparesCard);

    const readingCard = document.createElement('div');
    readingCard.className = 'detail-card';
    readingCard.innerHTML = `
      <div class="detail-card-head"><strong>Lectura rápida de esta run</strong></div>
      <p>${summarizeNodeRun(node, nodeRun)}</p>
    `;
    explanation.body.appendChild(readingCard);
    infoGroup.appendChild(explanation.section);
  }

  renderPathDecisionSection(nodeId, nodeRun, infoGroup);

  renderNodeSpecificSections(nodeId, infoGroup, valuesGroup);

  const inputs = createDetailSection('Inputs', { open: false });
  renderFactRows(inputs.body, nodeSpecificInputRows(nodeId, nodeRun, participants));
  valuesGroup.appendChild(inputs.section);

  const outputs = createDetailSection('Outputs', { open: false });
  renderFactRows(outputs.body, nodeSpecificOutputRows(nodeId, nodeRun));
  valuesGroup.appendChild(outputs.section);

  const checksSection = createDetailSection('Checks', {
    badge: gatherRelevantChecks(nodeRun, state.selectedDetId).length,
    open: false,
  });
  const checksWrap = document.createElement('div');
  checksWrap.className = 'detail-grid';
  renderCheckCards(gatherRelevantChecks(nodeRun, state.selectedDetId), checksWrap);
  checksSection.body.appendChild(checksWrap);
  valuesGroup.appendChild(checksSection.section);

  detailTabContent.appendChild(infoGroup);
  detailTabContent.appendChild(valuesGroup);
}

function candidateScoreKeys(rows) {
  return VISUAL_SCORE_COLUMNS.filter((column) => rows.some((row) => row[column.key] != null));
}

function groupRowsByDetection(rows) {
  const byDet = new Map();
  for (const row of rows) {
    const detId = Number(row.det_id);
    const bucket = byDet.get(detId) || [];
    bucket.push(row);
    byDet.set(detId, bucket);
  }
  return [...byDet.entries()].sort((a, b) => a[0] - b[0]);
}

function renderVisualSimilarityTab() {
  clearChildren(detailTabBadges);
  clearChildren(detailTabContent);
  detailTabKind.textContent = 'Detalle de similitud';
  detailTabTitle.textContent = 'Similitud visual';

  const buildRun = getNodeRun('visual.build_candidates');
  const diagRun = getNodeRun('visual.report_diagnosis');
  const rows = buildRun?.candidate_rows || [];
  const scoreColumns = candidateScoreKeys(rows);

  [
    `${rows.length} filas`,
    `${scoreColumns.length} scores visibles`,
    `foco: ${state.selectedDetId == null ? 'todas' : `det ${state.selectedDetId}`}`,
  ].forEach((label) => {
    const badge = document.createElement('div');
    badge.className = 'badge';
    badge.textContent = label;
    detailTabBadges.appendChild(badge);
  });

  const intro = document.createElement('div');
  intro.className = 'detail-card';
  intro.innerHTML = `
    <div class="detail-card-head"><strong>Matriz de similitud</strong></div>
    <p>
      Esta pestaña resume la evidencia visual serializada de verdad por la
      traza del bloque <strong>Construcción de candidatos</strong>. Hoy las columnas
      canónicas son <code>score_sim</code>, <code>score_sim_base</code> y
      <code>score_known</code>; no se inventan canales que la run no haya
      publicado.
    </p>
  `;
  detailTabContent.appendChild(intro);

  if (!rows.length) {
    detailTabContent.innerHTML += '<div class="empty-state">No hay filas de similitud en la traza actual.</div>';
    return;
  }

  const activeRows = state.selectedDetId == null
    ? rows
    : rows.filter((row) => Number(row.det_id) === Number(state.selectedDetId));

  for (const [detId, detRows] of groupRowsByDetection(activeRows)) {
    detRows.sort((a, b) => {
      const av = Number(a.score_sim ?? 0);
      const bv = Number(b.score_sim ?? 0);
      return bv - av;
    });
    const diag = (diagRun?.detection_rows || []).find((row) => Number(row.det_id) === Number(detId));

    const section = document.createElement('div');
    section.className = 'detail-card';
    const header = document.createElement('div');
    header.className = 'detail-card-head';
    header.innerHTML = `
      <strong>det ${detId}</strong>
      <span class="badge">${diag ? `${diag.status || '—'} · ${diag.reason || 'sin razón'}` : 'sin diagnóstico'}</span>
    `;
    section.appendChild(header);

    const tableWrap = document.createElement('div');
    tableWrap.className = 'table-wrap';
    const table = document.createElement('table');
    table.className = 'detail-table';
    const headColumns = ['objeto', 'rank', ...scoreColumns.map((column) => column.label), 'estado', 'razón'];
    table.innerHTML = `
      <thead>
        <tr>${headColumns.map((key) => `<th>${key}</th>`).join('')}</tr>
      </thead>
      <tbody>
        ${detRows.map((row, index) => `
          <tr>
            <td>${objectLabelForId(row.object_id)}</td>
            <td>${pretty(row.rank ?? (index + 1))}</td>
            ${scoreColumns.map((column) => `<td>${pretty(row[column.key])}</td>`).join('')}
            <td>${isDroppedCandidate(row) ? 'descartado' : 'activo'}</td>
            <td>${row.reason || row.veto_reason || row.gate_reason || '—'}</td>
          </tr>
        `).join('')}
      </tbody>
    `;
    tableWrap.appendChild(table);
    section.appendChild(tableWrap);
    detailTabContent.appendChild(section);
  }
}

function renderObjectsTab() {
  clearChildren(detailTabBadges);
  clearChildren(detailTabContent);
  detailTabKind.textContent = 'Snapshot de memoria';
  detailTabTitle.textContent = 'Objetos en memoria';

  const rows = objectSnapshotRows();
  [
    `${rows.length} objetos`,
    `frame ${String(state.trace?.frame_id ?? '').padStart(3, '0')}`,
    'snapshot de memoria',
  ].forEach((label) => {
    const badge = document.createElement('div');
    badge.className = 'badge';
    badge.textContent = label;
    detailTabBadges.appendChild(badge);
  });

  const intro = document.createElement('div');
  intro.className = 'detail-card';
  intro.innerHTML = `
    <div class="detail-card-head"><strong>Resumen disponible</strong></div>
    <p>
      Este snapshot muestra todos los objetos persistentes en memoria para el
      frame seleccionado, no solo los de la clase activa. Incluye label
      completo, ID global, hits, last seen y el número de descriptores de
      objeto, fondo y partes.
    </p>
  `;
  detailTabContent.appendChild(intro);

  if (!manifestHasMemorySnapshot(state.selectedFrameId)) {
    detailTabContent.innerHTML += '<div class="empty-state">La traza seleccionada no incluye snapshots de memoria para este frame.</div>';
    return;
  }

  if (!rows.length) {
    detailTabContent.innerHTML += '<div class="empty-state">No se encontró snapshot de memoria para este frame.</div>';
    return;
  }

  const tableWrap = document.createElement('div');
  tableWrap.className = 'table-wrap';
  const table = document.createElement('table');
  table.className = 'detail-table';
    table.innerHTML = `
    <thead>
      <tr>
        <th>label</th>
        <th>ID global</th>
        <th>hits</th>
        <th>last seen</th>
        <th>desc obj</th>
        <th>desc bg</th>
        <th>desc parts</th>
      </tr>
    </thead>
    <tbody>
      ${rows.map((row) => `
        <tr>
          <td>${row.label}</td>
          <td>${row.object_id}</td>
          <td>${row.hits}</td>
          <td>${pretty(row.last_seen)}</td>
          <td>${row.obj_desc_count}</td>
          <td>${row.bg_desc_count}</td>
          <td>${row.parts_desc_count}</td>
        </tr>
      `).join('')}
    </tbody>
  `;
  tableWrap.appendChild(table);
  detailTabContent.appendChild(tableWrap);
}

function renderVisiblePane() {
  const tab = currentTab();
  const isOverview = tab.id === 'overview';
  overviewPane.classList.toggle('hidden', !isOverview);
  detailPane.classList.toggle('hidden', isOverview);
  if (isOverview) return;

  if (tab.type === 'visual_similarity') {
    renderVisualSimilarityTab();
    return;
  }
  if (tab.type === 'objects') {
    renderObjectsTab();
    return;
  }
  if (tab.type === 'node') {
    renderGenericNodeTab(tab.nodeId);
  }
}

function canKeepTabOpen(tab) {
  if (!tab || typeof tab !== 'object') return false;
  if (tab.type === 'objects' || tab.type === 'visual_similarity') return true;
  if (tab.type === 'node') {
    return orderedNodes().some((node) => String(node.id) === String(tab.nodeId));
  }
  return false;
}

async function loadRuns() {
  const payload = await fetchJson('/api/runs');
  state.runs = payload.runs || [];
  if (!state.runs.length) throw new Error('No se encontraron runs de association_trace.');
  state.selectedRunId = state.runs[0].run_id;
  renderRuns();
}

async function loadManifestAndSchema() {
  state.manifest = await fetchJson(`/api/run/${encodeURIComponent(state.selectedRunId)}/manifest`);
  state.schema = await fetchJson(`/api/run/${encodeURIComponent(state.selectedRunId)}/schema`);
  const firstFrame = (state.manifest.frame_ids || [])[0];
  state.selectedFrameId = firstFrame;
  renderFrames();
  state.selectedClassId = resolveSelectedClassIdForFrame(firstFrame, state.selectedClassId);
  renderClasses();
}

async function loadTrace() {
  if (state.selectedFrameId == null || state.selectedClassId == null) return;
  const previousActiveTabId = state.activeTabId;
  const previousOpenTabs = [...(state.openTabs || [])];
  const previousSelectedNodeId = state.selectedNodeId;
  state.trace = await fetchJson(`/api/run/${encodeURIComponent(state.selectedRunId)}/trace?frame_id=${state.selectedFrameId}&class_id=${state.selectedClassId}`);
  state.memorySnapshot = await fetchJson(`/api/run/${encodeURIComponent(state.selectedRunId)}/memory?frame_id=${state.selectedFrameId}`).catch(() => null);
  const defaultNodeId = state.trace.node_runs?.[0]?.node_id || state.schema?.nodes?.[0]?.id || null;
  const availableNodeIds = new Set((state.schema?.nodes || []).map((node) => String(node.id)));
  state.selectedNodeId = availableNodeIds.has(String(previousSelectedNodeId)) ? previousSelectedNodeId : defaultNodeId;
  state.selectedDetId = null;
  state.openTabs = previousOpenTabs.filter((tab) => canKeepTabOpen(tab));
  const activeTabStillAvailable = previousActiveTabId === 'overview'
    || state.openTabs.some((tab) => tab.id === previousActiveTabId);
  state.activeTabId = activeTabStillAvailable ? previousActiveTabId : (state.openTabs[0]?.id || 'overview');
  renderHeader();
  renderFramePreview();
  renderMemoryOverview();
  renderTabStrip();
  renderGraph();
  renderVisiblePane();
  fitGraphToViewport();
}

async function onRunChange() {
  state.selectedRunId = runSelect.value;
  await loadManifestAndSchema();
  await loadTrace();
}

async function onFrameChange() {
  const previousClassId = state.selectedClassId;
  state.selectedFrameId = Number(frameSelect.value);
  state.selectedClassId = resolveSelectedClassIdForFrame(state.selectedFrameId, previousClassId);
  renderClasses();
  await loadTrace();
}

async function onClassChange() {
  state.selectedClassId = Number(classSelect.value);
  await loadTrace();
}

function bindViewport() {
  window.addEventListener('resize', () => fitGraphToViewport());
}

function canHandleGlobalFrameShortcut(event) {
  if (!event || event.defaultPrevented) return false;
  if (event.altKey || event.ctrlKey || event.metaKey) return false;
  const target = event.target;
  if (!(target instanceof HTMLElement)) return true;
  if (target.isContentEditable) return false;
  const tag = String(target.tagName || '').toUpperCase();
  return tag !== 'INPUT' && tag !== 'TEXTAREA';
}

async function moveFrameSelection(delta) {
  if (state.frameNavigationBusy) return;
  const frameIds = (state.manifest?.frame_ids || []).map((value) => Number(value)).filter((value) => Number.isFinite(value));
  if (!frameIds.length || !Number.isFinite(Number(state.selectedFrameId))) return;
  const currentIndex = frameIds.findIndex((frameId) => frameId === Number(state.selectedFrameId));
  if (currentIndex < 0) return;
  const nextIndex = currentIndex + Number(delta);
  if (nextIndex < 0 || nextIndex >= frameIds.length) return;
  state.frameNavigationBusy = true;
  try {
    const previousClassId = state.selectedClassId;
    state.selectedFrameId = frameIds[nextIndex];
    renderFrames();
    state.selectedClassId = resolveSelectedClassIdForFrame(state.selectedFrameId, previousClassId);
    renderClasses();
    await loadTrace();
  } finally {
    state.frameNavigationBusy = false;
  }
}

function bindKeyboardShortcuts() {
  window.addEventListener('keydown', (event) => {
    if (!canHandleGlobalFrameShortcut(event)) return;
    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      void moveFrameSelection(-1);
      return;
    }
    if (event.key === 'ArrowRight') {
      event.preventDefault();
      void moveFrameSelection(1);
    }
  });
}

runSelect.addEventListener('change', () => void onRunChange());
frameSelect.addEventListener('change', () => void onFrameChange());
classSelect.addEventListener('change', () => void onClassChange());
openObjectsTabButton.addEventListener('click', () => openObjectsTab());
memoryOverview.addEventListener('dblclick', () => openObjectsTab());

async function boot() {
  bindViewport();
  bindKeyboardShortcuts();
  try {
    await loadRuns();
    await loadManifestAndSchema();
    await loadTrace();
  } catch (error) {
    traceTitle.textContent = 'No se pudo cargar la traza';
    if (memoryOverview) {
      memoryOverview.innerHTML = `<div class="empty-state">${error.message}</div>`;
    }
  }
}

boot();
