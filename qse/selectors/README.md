# Selector YAML schema (Mode C)

Each selector file should include:

- `name`: selector type (currently `gap_volume`)
- `description`: human-readable purpose
- `parameters`: selector-specific fields
  - For `gap_volume`: `gap_min` (float), `volume_z_min` (float), `horizon` (int)

Examples: see `gap_down_volume_spike.yaml`, `breakout_momentum.yaml`.
