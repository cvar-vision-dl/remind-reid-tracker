export const DET_COLORS = [
  '#c04b2d', '#1d7f6f', '#2e5faa', '#a15c14', '#8d3ea3', '#27724f', '#7f4c1c', '#006f9e',
];

export const OUTCOME_COLORS = {
  MATCH: '#2f7c55',
  NEW: '#1d7f6f',
  AMBIGUOUS_TRACK: '#b87712',
  PROVISIONAL_PARENT: '#6f5ca8',
  PROVISIONAL_NEW: '#a24831',
  UNASSIGNED: '#6a7779',
};

export const STATUS_COLORS = {
  pass: '#2f7c55',
  fail: '#a24831',
  soft: '#b87712',
  na: '#6a7779',
};

export const DOOR_COLORS = {
  open: '#2f7c55',
  filtered: '#b87712',
  inactive: '#6a7779',
  blocked: '#a24831',
  closed: '#6a7779',
  soft: '#b87712',
  missing: '#6a7779',
};

export const NODE_LABELS = {
  'prepare.class_partition': 'Preparar clase',
  'prepare.reliable_visual_anchors': 'Anchors visuales fiables',
  'prepare.valid_detections': 'Filtro de detecciones válidas',
  'visual.build_candidates': 'Construcción de candidatos',
  'visual.report_diagnosis': 'Diagnóstico visual',
  'context.neighbor_sets_hypotheses': 'Hipótesis de sets',
  'context.sets_activation': 'Activación de sets',
  'shape.allow_for_report': 'Uso de contexto por reporte',
  'shape.context_veto': 'Shaping por candidato',
  'shape.final_score_tables': 'Tablas finales de score',
  'resolve.locks': 'Locks',
  'resolve.hungarian': 'Hungarian',
  'post.identity_stability': 'Estabilidad de identidad',
  'post.assignment_ambiguity': 'Ambigüedad de asignación',
  'post.known_set_distance_disambiguation': 'Desambiguación known-set-distance',
  'post.create_competition': 'Competición de create',
  'post.ambiguous_track_candidates': 'Candidatos ambiguos',
  'post.provisional_reconciliation': 'Reconciliación provisional',
  'post.final_decision_pack': 'Empaquetado final',
  'outcome.final_ambiguity': 'Diagnóstico final',
  'outcome.finalize': 'Outcome final',
};

export const MODULE_SPECS = [
  {
    id: 'inputs',
    label: 'Entradas y Particion',
    caption: 'detecciones de la clase y partición inicial',
    color: '#7f6d44',
    nodes: ['prepare.class_partition'],
  },
  {
    id: 'visual',
    label: 'Evidencia Visual',
    caption: 'candidatos visuales, anchors fiables y lectura visual base',
    color: '#2b6e66',
    nodes: ['visual.build_candidates', 'prepare.reliable_visual_anchors', 'visual.report_diagnosis'],
  },
  {
    id: 'context',
    label: 'Contexto',
    caption: 'hipótesis relacionales y activación contextual',
    color: '#2f5f95',
    nodes: ['context.neighbor_sets_hypotheses', 'context.sets_activation'],
  },
  {
    id: 'decision',
    label: 'Matching',
    caption: 'dependencias de preparación, gating y resolución por clase antes del resultado final',
    color: '#8a5f1d',
    nodes: [
      'prepare.valid_detections',
      'shape.allow_for_report',
      'shape.context_veto',
      'shape.final_score_tables',
      'resolve.locks',
      'resolve.hungarian',
    ],
  },
  {
    id: 'disambiguation',
    label: 'Post Assignment',
    caption: 'estabilidad, ambigüedad y reconciliación temporal',
    color: '#8a3f60',
    nodes: [
      'post.assignment_ambiguity',
      'post.identity_stability',
      'post.create_competition',
      'post.ambiguous_track_candidates',
      'post.known_set_distance_disambiguation',
      'post.provisional_reconciliation',
      'post.final_decision_pack',
    ],
  },
  {
    id: 'outputs',
    label: 'Decision Final',
    caption: 'diagnóstico final y outcome legible por detección',
    color: '#52743a',
    nodes: ['outcome.final_ambiguity', 'outcome.finalize'],
  },
];

export const MODULE_BY_NODE_ID = Object.fromEntries(
  MODULE_SPECS.flatMap((spec) => spec.nodes.map((nodeId) => [nodeId, spec]))
);

export const DEFAULT_NODE_W = 460;
export const DEFAULT_NODE_H = 190;
export const MAX_NODE_W = 920;
export const GRAPH_W = 2860;
export const NODE_ROAD_CLEARANCE = 38;
export const ROAD_GRID_MARGIN = 30;
export const ROAD_TURN_PENALTY = 28;
export const ROAD_DET_SPREAD = 10;

export const VISUAL_SCORE_COLUMNS = [
  { key: 'score_sim', label: 'sim' },
  { key: 'score_sim_base', label: 'sim base' },
  { key: 'score_known', label: 'known' },
];

export const NODE_GENERAL_DESCRIPTIONS = {
  'prepare.class_partition': 'Aísla las detecciones y objetos de la clase que se van a comparar.',
  'prepare.reliable_visual_anchors': 'Selecciona referencias visuales especialmente fiables para apoyar el contexto posterior.',
  'prepare.valid_detections': 'Separa las detecciones con features comparables de las que no pueden entrar al matching conocido por falta de features.',
  'visual.build_candidates': 'Calcula y ordena los objetos candidatos por score de similitud visual. Aquí todavía no se materializa un NEW final aunque una detección salga sin candidatos.',
  'visual.report_diagnosis': 'Resume si la evidencia visual sale fuerte, ambigua o débil.',
  'context.neighbor_sets_hypotheses': 'Construye hipótesis relacionales globales a partir de las detecciones visibles, anchors fiables y memoria de vecindad. Todavía no decide si ese contexto será suficientemente fiable para influir.',
  'context.sets_activation': 'Decide si las hipótesis relacionales ya construidas son lo bastante consistentes como para entrar de verdad en el proceso de asociación.',
  'shape.allow_for_report': 'Decide si una detección válida puede usar contexto adicional según su report.',
  'shape.context_veto': 'Aplica el shaping tardío por candidato: plausibilidad conocida, gates, rescates y vetos antes de construir las tablas operativas.',
  'shape.final_score_tables': 'Materializa score_sim, score_assign y score_final con los candidatos vivos. Si queda vacía, aquí se cierra la rama de matching conocido.',
  'resolve.locks': 'Cierra coincidencias claras antes de Hungarian.',
  'resolve.hungarian': 'Resuelve conjuntamente las detecciones que siguen compitiendo contra objetos y dummies, y luego acepta o rechaza la asignación.',
  'post.identity_stability': 'Comprueba si el match propuesto es estable.',
  'post.assignment_ambiguity': 'Detecta si la asignación todavía sigue siendo ambigua.',
  'post.known_set_distance_disambiguation': 'Intenta romper ambigüedades con evidencia conocida.',
  'post.create_competition': 'Decide si compiten altas nuevas frente a alternativas conocidas.',
  'post.ambiguous_track_candidates': 'Materializa la bolsa real de candidatos ambiguos que pasa a la resolución temporal, mezclando ambigüedad contextual, identidad inestable y committed new.',
  'post.provisional_reconciliation': 'Reinterpreta creates y ambigüedades residuales como salidas temporales coherentes cuando hace falta.',
  'post.final_decision_pack': 'Materializa la precedencia final entre matches, creates, ambiguos y provisionales antes del outcome legible.',
  'outcome.final_ambiguity': 'Calcula el diagnóstico final por score_final para dejar visible si el resultado ya llega fuerte, ambiguo o débil antes del outcome legible.',
  'outcome.finalize': 'Anota la salida final legible por detección dentro del report consumido por update.',
};

export const TREE_PHASES = [
  {
    id: 'inputs',
    label: 'Entradas posibles',
    caption: 'detecciones de la clase y objetos disponibles en memoria',
    color: '#8a6a36',
    rows: [
      ['synthetic.input_detections', 'synthetic.input_memory'],
      ['prepare.class_partition'],
    ],
  },
  {
    id: 'visual',
    label: 'Evidencia visual',
    caption: 'candidatos de similitud, anchors fiables y diagnóstico visual bruto',
    color: '#2b6e66',
    rows: [
      ['visual.build_candidates'],
      ['prepare.reliable_visual_anchors', 'visual.report_diagnosis'],
    ],
  },
  {
    id: 'context',
    label: 'Contexto',
    caption: 'hipótesis globales y activación contextual',
    color: '#2f5f95',
    rows: [
      ['context.neighbor_sets_hypotheses'],
      ['context.sets_activation'],
    ],
  },
  {
    id: 'decision',
    label: 'Matching',
    caption: 'dependencias reales entre preparación, gating y resolución antes del matching final',
    color: '#8a5f1d',
    rows: [
      ['prepare.valid_detections'],
      ['shape.allow_for_report'],
      ['shape.context_veto'],
      ['shape.final_score_tables'],
      ['resolve.locks'],
      ['resolve.hungarian'],
    ],
  },
  {
    id: 'disambiguation',
    label: 'Desambiguación',
    caption: 'guards post-assignment, competición y reconciliación temporal',
    color: '#8a3f60',
    rows: [
      [null, 'post.assignment_ambiguity', null],
      ['post.identity_stability', null, 'post.create_competition'],
      [null, 'post.ambiguous_track_candidates', null],
      [null, 'post.known_set_distance_disambiguation', null],
      [null, 'post.provisional_reconciliation', null],
      [null, 'post.final_decision_pack', null],
    ],
  },
  {
    id: 'final',
    label: 'Decisión final',
    caption: 'salidas finales hermanas materializadas para la clase',
    color: '#52743a',
    rows: [['outcome.final_ambiguity', 'outcome.finalize']],
  },
];

export const NODE_LAYOUT_OFFSETS = {};

export const VISUAL_GRAPH_EDGES = [
  { from: 'synthetic.input_detections', to: 'prepare.class_partition' },
  { from: 'synthetic.input_memory', to: 'prepare.class_partition' },
  { from: 'prepare.class_partition', to: 'visual.build_candidates' },
  { from: 'prepare.class_partition', to: 'prepare.valid_detections' },
  { from: 'prepare.class_partition', to: 'context.neighbor_sets_hypotheses' },
  { from: 'visual.build_candidates', to: 'prepare.reliable_visual_anchors' },
  { from: 'visual.build_candidates', to: 'visual.report_diagnosis' },
  { from: 'prepare.reliable_visual_anchors', to: 'context.neighbor_sets_hypotheses' },
  { from: 'context.neighbor_sets_hypotheses', to: 'context.sets_activation' },
  { from: 'visual.report_diagnosis', to: 'shape.allow_for_report' },
  { from: 'context.sets_activation', to: 'shape.allow_for_report' },
  { from: 'prepare.valid_detections', to: 'shape.allow_for_report' },
  { from: 'shape.allow_for_report', to: 'shape.context_veto' },
  { from: 'shape.context_veto', to: 'shape.final_score_tables' },
  { from: 'shape.final_score_tables', to: 'resolve.locks' },
  { from: 'resolve.locks', to: 'resolve.hungarian' },
  { from: 'resolve.hungarian', to: 'post.assignment_ambiguity' },
  { from: 'post.assignment_ambiguity', to: 'post.identity_stability' },
  { from: 'post.identity_stability', to: 'post.create_competition' },
  { from: 'post.create_competition', to: 'post.ambiguous_track_candidates' },
  { from: 'post.ambiguous_track_candidates', to: 'post.known_set_distance_disambiguation' },
  { from: 'post.known_set_distance_disambiguation', to: 'post.provisional_reconciliation' },
  { from: 'post.provisional_reconciliation', to: 'post.final_decision_pack' },
  { from: 'post.final_decision_pack', to: 'outcome.final_ambiguity' },
  { from: 'post.final_decision_pack', to: 'outcome.finalize' },
  {
    from: 'resolve.locks',
    to: 'post.assignment_ambiguity',
    skipNodes: ['resolve.hungarian'],
  },
  {
    from: 'prepare.valid_detections',
    to: 'post.create_competition',
    skipNodes: [
      'shape.allow_for_report',
      'shape.context_veto',
      'shape.final_score_tables',
      'resolve.locks',
      'resolve.hungarian',
      'post.assignment_ambiguity',
      'post.identity_stability',
    ],
  },
  {
    from: 'shape.final_score_tables',
    to: 'post.create_competition',
    skipNodes: ['resolve.locks', 'resolve.hungarian', 'post.assignment_ambiguity', 'post.identity_stability'],
  },
  {
    from: 'resolve.hungarian',
    to: 'post.create_competition',
    skipNodes: ['post.assignment_ambiguity', 'post.identity_stability'],
  },
  {
    from: 'post.create_competition',
    to: 'post.provisional_reconciliation',
    skipNodes: ['post.ambiguous_track_candidates', 'post.known_set_distance_disambiguation'],
  },
  {
    from: 'post.create_competition',
    to: 'post.final_decision_pack',
    skipNodes: ['post.ambiguous_track_candidates', 'post.known_set_distance_disambiguation', 'post.provisional_reconciliation'],
  },
  {
    from: 'post.ambiguous_track_candidates',
    to: 'post.provisional_reconciliation',
    skipNodes: ['post.known_set_distance_disambiguation'],
  },
  {
    from: 'post.ambiguous_track_candidates',
    to: 'post.final_decision_pack',
    skipNodes: ['post.known_set_distance_disambiguation', 'post.provisional_reconciliation'],
  },
  {
    from: 'post.known_set_distance_disambiguation',
    to: 'post.final_decision_pack',
    skipNodes: ['post.provisional_reconciliation'],
  },
];
