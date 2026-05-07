# REMIND: RE-Identification with Memory for INDoor Navigation

## Overview

REMIND is an online multi-object re-identification tracker that operates frame-by-frame without future-frame lookahead. Given a sequence of frames with per-object instance segmentation masks (from a detector or ground-truth annotations), the system assigns persistent identity labels to each detected object, maintaining consistency across the entire sequence even under occlusions, re-appearances, and visually similar instances.

The pipeline follows a three-stage architecture executed sequentially for every incoming frame:

1. **Perception** -- Extract visual features from each detected object.
2. **Association** -- Match current detections to previously tracked identities.
3. **Update** -- Incorporate new observations into the persistent memory.

---

## 1. Perception Stage

### 1.1 Frame Preprocessing

The input frame $I_t \in \mathbb{R}^{H \times W \times 3}$ is first resized to a target width while preserving aspect ratio, then cropped (or padded) so that both spatial dimensions are exact multiples of the ViT patch size $p$ (typically $p = 14$):

$$H_a = p \cdot \lfloor H' / p \rfloor, \quad W_a = p \cdot \lfloor W' / p \rfloor$$

This yields an aligned frame $I_a \in \mathbb{R}^{H_a \times W_a \times 3}$.

### 1.2 Detection

An instance segmentation backend (YOLO or ground-truth masks in DAVIS/ScanNet++ format) produces a set of detections $\mathcal{D}_t = \{d_1, \ldots, d_N\}$, each carrying:

- A binary pixel mask $M_d \in \{0,1\}^{H_a \times W_a}$
- A class label $c_d \in \mathcal{C}$
- Geometric attributes: bounding box, centroid, area

An optional class-level filter removes detections belonging to ignored semantic categories.

### 1.3 DINOv3 Feature Extraction

A single forward pass of a frozen DINOv3 vision transformer (ViT-S/B/L) on the aligned frame produces a dense patch-level feature map:

$$F \in \mathbb{R}^{H_p \times W_p \times D}$$

where $H_p = H_a / p$, $W_p = W_a / p$, and $D$ is the embedding dimension. When attention-based part descriptors are enabled, the model additionally returns multi-head self-attention maps $A^{(l,h)} \in \mathbb{R}^{N_{patch} \times N_{patch}}$ from each layer $l$ and head $h$.

### 1.4 Patch Coverage

For each detection $d$ with pixel mask $M_d$, a patch-level coverage map is computed by measuring the fraction of foreground pixels within each $p \times p$ patch cell:

$$\text{cov}(i,j) = \frac{1}{p^2} \sum_{(u,v) \in \text{cell}(i,j)} M_d(u,v)$$

A patch $(i,j)$ is considered valid for the object if $\text{cov}(i,j) > 0$ (or above a configurable threshold). The coverage values also serve as spatial weights for descriptor pooling.

### 1.5 Object Descriptors

For each detection, object-level descriptors are computed by pooling DINOv3 patch features over the valid object patches:

**Global descriptor.** A coverage-weighted mean over valid patches, followed by $\ell_2$-normalization:

$$\mathbf{g}_d = \frac{\sum_{(i,j) \in \mathcal{P}_d} \text{cov}(i,j) \cdot F(i,j)}{\sum_{(i,j) \in \mathcal{P}_d} \text{cov}(i,j)}, \quad \hat{\mathbf{g}}_d = \frac{\mathbf{g}_d}{\|\mathbf{g}_d\|_2}$$

**Trimmed-mean descriptor.** A two-pass pooling that first computes a preliminary mean $\mathbf{g}_0$, then retains only the top fraction $\rho$ of patches by cosine similarity to $\mathbf{g}_0$ before re-pooling. This discards outlier patches near mask boundaries.

**Per-patch descriptors.** The individual $\ell_2$-normalized feature vectors $\{\hat{F}(i,j)\}_{(i,j) \in \mathcal{P}_d}$ are retained for part-based and prototype-based matching.

### 1.6 Part Descriptors

Part descriptors decompose each object into a small set of semantic sub-regions, producing a multi-prototype representation. Two methods are available:

**K-means parts.** The $\ell_2$-normalized patch features of the object are clustered via K-means into $K$ groups ($K$ typically 3--5). Each cluster yields a part descriptor via weighted pooling of its member patches. Clusters below a minimum patch count are discarded. A greedy merge step fuses clusters whose centroids have cosine similarity above a threshold $\tau_{\text{merge}}$.

**Attention parts.** Seed patches are selected by ranking object patches according to their in-degree in the self-attention graph (summed across layers). For each seed, a region is formed by taking the top-$k$ most-attended object patches. The part descriptor is the weighted pool of that region.

Each part descriptor $\mathbf{p}_k$ is $\ell_2$-normalized, and accompanied by statistics: patch count, coverage support, and intra-cluster coherence.

### 1.7 Background Descriptors

Local background context is captured through concentric rings in patch space around each object:

1. The object patch mask is morphologically sanitized (hole-filling, closing).
2. An **inner ring** is constructed by dilating the object mask by $r_{\text{in}}$ patches and subtracting the object region (plus a border-exclusion zone).
3. An **outer ring** is constructed analogously with radius $r_{\text{out}} > r_{\text{in}}$.
4. Both radii adapt upward if the ring contains fewer patches than a minimum threshold.

For each ring, two types of descriptors are produced:

**Global ring descriptor.** Weighted mean of patch features in the ring (weights = $1 - \text{cov}_{\text{obj}}$), $\ell_2$-normalized.

**Ring prototypes.** K-means clustering on $\ell_2$-normalized ring patches, followed by inter-cluster merging (union-find on cosine similarity), and selection of the top-$N$ most stable prototypes (ranked by mass $\times$ cohesion$^{\alpha}$). Prototypes are represented either as medoids or centroids.

A combined background descriptor linearly blends the inner and outer global descriptors with configurable weights.

---

## 2. Association Stage

The association stage determines, for each current detection, whether it corresponds to an existing tracked identity or should create a new one. The process follows an explicit multi-step decision flow.

### 2.1 Visual Evidence (Similarity Reports)

For each detection $d$, a **similarity report** is computed against every tracked object $o$ in memory of the same class. The report aggregates evidence from multiple channels:

**Object similarity.** The observed global descriptor $\hat{\mathbf{g}}_d$ is compared against the tracked object's stored prototypes (both "work" and "stable" banks) via cosine similarity:

$$s_{\text{obj}}(d, o) = \max_{k} \; \cos(\hat{\mathbf{g}}_d, \; \hat{\mathbf{e}}_k^{(o)})$$

where $\hat{\mathbf{e}}_k^{(o)}$ are the appearance prototypes of object $o$. Both work and stable banks are queried, and the result is collapsed (max, or stable-preferred) depending on configuration.

**Part similarity.** For each active part channel (kmeans, attention), each observed part descriptor is matched against the stored part prototypes of the tracked object via best-match cosine similarity. The channel score is the mean of the top-$k$ best matches:

$$s_{\text{parts}}(d, o) = \frac{1}{\min(k, |\mathcal{P}_d|)} \sum_{i=1}^{\min(k, |\mathcal{P}_d|)} \max_{j} \; \cos(\mathbf{p}_i^{(d)}, \; \mathbf{p}_j^{(o)})$$

**Background similarity.** The observed ring descriptors (global and prototype-level) are compared against the tracked object's background model prototypes using the same best-match cosine scheme. Inner and outer terms are combined with configurable weights.

### 2.2 Quality-Weighted Score Combination

The individual channel similarities are combined into a single score $s_{\text{sim}}(d,o)$ via a quality-aware weighted sum. Each channel $c \in \{\text{object}, \text{background}, \text{parts}\}$ carries:

- A nominal weight $w_c$ (from configuration).
- A quality factor $q_c \in [0,1]$ derived from the detection's feature richness (number of valid patches, part count, background ring sizes, mask quality). A floor mechanism ensures a minimum effective weight.

The combined score is:

$$s_{\text{sim}}(d, o) = \frac{\sum_{c} w_c \cdot q_c^{\text{eff}} \cdot s_c(d,o)}{\sum_{c} w_c \cdot q_c^{\text{eff}}}$$

where the denominator re-normalizes when channels are unavailable (e.g., parts disabled, background too small).

### 2.3 Reliable Visual Anchors

Before assignment, the system identifies a set of **reliable anchor pairs** -- detections whose best match is unambiguously strong (score above a confirmation threshold $\tau_{\text{confirm}}$ and gap to second-best above a clear margin $\delta_{\text{confirm}}$). These anchors serve as stable reference points for contextual reasoning in subsequent steps.

### 2.4 Neighbor Sets (Contextual Layer)

When enabled, a spatial co-occurrence context layer enriches the association evidence. The neighbor-sets module operates on the principle that objects frequently seen together form stable spatial neighborhoods:

1. For each tracked object, historical neighbor co-occurrence frequencies are maintained as a probability kernel over object IDs.
2. Given the current frame's reliable anchors, a **neighbor-sets hypothesis** is generated: which tracked objects are expected to be present based on the observed spatial context.
3. Each detection-object candidate receives a contextual bonus or penalty:
   - **Support**: candidates compatible with the neighborhood hypothesis receive a positive adjustment (capped at $\delta_+$).
   - **Contradiction**: candidates clearly incompatible receive a negative adjustment (capped at $\delta_-$).

The influence magnitude scales with a quality metric that combines coverage, maturity, density, and pruning effectiveness of the neighbor model.

### 2.5 Ambiguity Diagnosis

Each similarity report is classified as:

- **STRONG**: the best candidate has high score and clear margin over the second-best.
- **AMBIGUOUS**: two or more candidates have similar scores (small gap).
- **WEAK**: no candidate exceeds the matching threshold.

This classification informs downstream gating in the assignment and update stages.

### 2.6 Global Assignment (Hungarian Algorithm)

The assignment problem is solved globally per semantic class using the Hungarian algorithm on a bipartite cost matrix between detections and tracked objects:

1. **Score table construction.** For each (detection, object) pair, the assignment score is derived from $s_{\text{sim}}$ plus optional neighbor-sets adjustments. A context veto can zero-out candidates that the neighbor model strongly rejects.

2. **Lock resolution.** Before Hungarian, high-confidence pairs (score $\geq \tau_{\text{lock}}$ with sufficient gap) are "locked" -- pre-assigned and removed from the cost matrix. This prevents the global optimization from breaking obvious matches.

3. **Dummy columns.** Virtual "new object" columns are added to the cost matrix so that detections can be assigned to "create new" if no existing track is sufficiently similar. The dummy score adapts per detection based on a confidence-aware formula.

4. **Cost matrix and optimization.** The score matrix is negated (Hungarian minimizes cost) and solved via `scipy.optimize.linear_sum_assignment`. Assignments to dummy columns produce "create new" decisions; assignments below minimum thresholds are also redirected to creation.

The output is a list of decided matches $\{(d_i, o_j, s_{ij})\}$ and detections to create as new tracks.

### 2.7 Post-Assignment Guards

After Hungarian, several guard mechanisms refine the raw assignments:

**Identity stability check.** Verifies that the assignment does not produce implausible identity swaps by examining temporal consistency.

**Ambiguous track candidates.** Detections assigned to a match but with ambiguous similarity (multiple plausible identities with close scores) are flagged for deferred resolution. These become **AmbiguousTracks** -- temporary entities that accumulate observations until the ambiguity is resolved.

**Provisional new tracks.** Detections assigned as "new" but with moderate similarity to existing objects are created as **ProvisionalNewTracks** rather than immediately confirmed. This prevents premature identity proliferation when a re-appearing object is temporarily hard to match.

**Known-set distance disambiguation.** When a group of detections maps ambiguously to a known set of object IDs (all candidates within a closed set), a relational disambiguator uses spatial distance memory to break ties:

- Observed spatial relations (center distances, containment, contact) between current detections are compared against the historical distance graph stored for each pair of tracked objects.
- Anchor-based scoring leverages reliably matched objects as reference points to triangulate the correct assignment.
- A combined score of visual similarity and relational evidence is used to resolve the ambiguity.

---

## 3. Update Stage

The update stage incorporates the association decisions into persistent memory.

### 3.1 Lifecycle Management

Each tracked object follows a state machine: **NEW** $\to$ **TENTATIVE** $\to$ **CONFIRMED** $\to$ **INACTIVE** (and optionally removed).

- **Hits and misses**: Each frame, a matched object increments its hit counter and resets misses. Unmatched objects increment misses.
- **Confirmation**: After accumulating $h_{\text{confirm}}$ hits, an object transitions from TENTATIVE to CONFIRMED.
- **Inactivation**: A CONFIRMED object exceeding $m_{\text{max}}$ consecutive misses becomes INACTIVE.
- **Removal**: TENTATIVE objects exceeding their miss budget, or INACTIVE objects exceeding a TTL, are removed from memory.

### 3.2 Descriptor Update (Appearance Prototypes)

Each tracked object maintains a multi-prototype appearance model organized in channels (e.g., `global`, `global_trimmed`). Each channel has two banks:

- **Work prototypes**: Recently observed embeddings, actively updated.
- **Stable prototypes**: Consolidated embeddings promoted from the work bank after sufficient observations.

When a detection is matched to an object, the observed descriptor is integrated into the work bank:

1. **Duplicate check**: The new observation $\hat{\mathbf{x}}$ is compared against all existing work prototypes. If $s_{\max} = \max_k \cos(\hat{\mathbf{x}}, \hat{\mathbf{e}}_k) > \tau_{\text{dup}}$, it is treated as a duplicate observation of the closest prototype.

2. **EMA update** (duplicate case): The matched prototype's embedding is updated via exponential moving average:

$$\hat{\mathbf{e}}_k \leftarrow \text{normalize}\big((1 - \alpha) \hat{\mathbf{e}}_k + \alpha \hat{\mathbf{x}}\big)$$

where $\alpha$ is gated by $s_{\max}$ (higher similarity $\Rightarrow$ more aggressive update) and scaled by the update decision's confidence.

3. **New prototype insertion** (non-duplicate case): If $s_{\max} < \tau_{\text{dup}}$, a new prototype is added to the work bank. If the bank is full, either the most redundant or the least-recently-used prototype is evicted.

4. **Merge maintenance**: After insertion, if any two work prototypes become too similar (cosine $> \tau_{\text{merge,internal}}$), they are merged by weighted average.

5. **Promotion to stable**: Work prototypes that accumulate sufficient observation count are promoted to the stable bank (with a copy), which provides a more conservative representation for matching.

### 3.3 Robust Update Gating

When robust updates are enabled, the update intensity adapts to the match quality:

| Match quality | Update behavior |
|---|---|
| **STRONG** | Full update: insert, merge, promote, EMA |
| **AMBIGUOUS** | Safe mode: EMA only with reduced $\alpha$ scale |
| **WEAK** | No update: descriptors are not modified |

This prevents unreliable matches from corrupting the stored appearance model.

### 3.4 Part Model Update

Part descriptors (kmeans and/or attention channels) are updated analogously to object appearance prototypes: each part channel maintains work and stable banks, with duplicate detection, EMA update, insertion, merge, and promotion following the same policies.

### 3.5 Background Model Update

The local background model for each object maintains separate prototype banks for inner and outer rings (both global and prototype-level), with the same work/stable dual-bank architecture. Updates follow the same insert-or-EMA policy.

### 3.6 Neighbor Graph Update

Two relational models are maintained per tracked object:

**Co-occurrence neighbor graph.** Records which other objects co-occur with a given object across frames. Each edge accumulates co-visibility episodes and an exponentially-weighted frequency kernel. This graph feeds the neighbor-sets context layer in the association stage.

**Distance neighbor graph.** Records pairwise spatial relations (normalized center distance, contact, containment, relative scale) between co-visible objects. Each directed edge stores a statistical summary of observed distances. This graph feeds the known-set distance disambiguator.

### 3.7 Cross-View Identity

For each object, a cross-view identity model captures how the object appears relative to spatial landmarks (larger "support" objects that contain it). This includes:

- Which support object (the smallest enclosing large-area neighbor) the object is associated with.
- A zone key (spatial grid cell within the support object's bounding box).
- Ordered lists of nearby anchor objects, enabling viewpoint-aware identity reasoning.

### 3.8 Ambiguous and Provisional Track Management

**Ambiguous tracks** accumulate observations frame-by-frame. The memory manager re-evaluates existing ambiguous tracks each frame, attempting to match new detections against them. When enough evidence accumulates (or the TTL expires), the track is either resolved into a confirmed identity or discarded.

**Provisional new tracks** similarly accumulate evidence before being materialized as new tracked objects. This avoids fragmenting identities when a re-appearing object initially has weak match scores.

---

## 4. System-Level Design

### 4.1 Configuration

All thresholds, weights, and architectural choices are specified in a YAML configuration file with a hierarchical structure. A base config can be merged with an override config for ablation studies. Key parameter groups include:

- `dino`: Model variant (S/B/L), patch size, normalization.
- `detector`: Backend (YOLO/DAVIS), ignored classes.
- `association.matching`: Weights ($w_{\text{obj}}, w_{\text{bg}}, w_{\text{parts}}$), match threshold, margin, Hungarian options.
- `association.matching.neighbor_sets_influence`: Context layer activation and capping.
- `memory.appearance`: Prototype counts, duplicate/merge thresholds, promotion criteria.
- `update.robust_updates`: Match-quality gating.

### 4.2 Memory Architecture

The `MemoryStore` is the central data structure holding:

- A dictionary of `TrackedObject` instances indexed by `object_id`.
- A class-based index for efficient same-class retrieval during association.
- Temporary track pools (`AmbiguousTrack`, `ProvisionalNewTrack`) with TTL-based lifecycle.
- An `AnchorViewStore` for managing canonical views.

Each `TrackedObject` aggregates:

- `ObjectAppearanceModel`: multi-channel, dual-bank (work/stable) prototype memory.
- `PartModel`: per-channel part prototype memory.
- `LocalBackgroundModel`: inner/outer ring prototype memory.
- `NeighborGraph`: co-occurrence frequency model.
- `NeighborDistanceGraph`: pairwise spatial relation statistics.
- `CrossViewIdentity`: viewpoint-relative spatial identity.

### 4.3 Processing Flow Summary

```
For each frame t:

  PERCEPTION:
    1. Preprocess: resize + align to patch grid
    2. Detect: YOLO / GT masks -> detections {d_1, ..., d_N}
    3. Extract: DINOv3 forward pass -> feature map F
    4. Per detection:
       a. Object descriptors (global, trimmed, patches)
       b. Part descriptors (kmeans, attention)
       c. Background descriptors (inner/outer rings, prototypes)

  ASSOCIATION:
    1. Similarity reports: score each (detection, tracked_object) pair
    2. Reliable anchor selection
    3. Neighbor-sets context activation (optional)
    4. Ambiguity diagnosis (STRONG / AMBIGUOUS / WEAK)
    5. Hungarian assignment with locks and dummies
    6. Post-assignment guards (identity stability, ambiguous/provisional)
    7. Known-set distance disambiguation (if needed)

  UPDATE:
    1. Apply matches: update descriptors (robust-gated)
    2. Create new objects from unmatched detections
    3. Manage ambiguous and provisional tracks
    4. Update neighbor graphs (co-occurrence + distance)
    5. Update cross-view identity
    6. Apply misses and lifecycle transitions
```
