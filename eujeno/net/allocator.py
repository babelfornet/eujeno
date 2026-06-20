# SPDX-FileCopyrightText: 2026 Alberto Ferrazzoli <alberto.ferrazzoli@gmail.com>
# SPDX-License-Identifier: Apache-2.0

"""Auto-assignment decision: given the picture of coverage gaps (coverage_gaps) and the
node's capacity, picks the stage spec to claim. Pure function."""


def choose_stages(gaps: dict, max_decoder_layers: int, num_layers: int,
                  take_embed_head: bool) -> str:
    """Returns a stage spec for parse_stages (e.g. 'embed,decoder:12-17,head'),
    or '' if there is nothing useful/possible to claim."""
    target = gaps.get("target", 1)
    parts = []
    if take_embed_head and gaps.get("embed_replicas", 0) < target:
        parts.append("embed")
    decoder_gaps = sorted(gaps.get("decoder_gaps", []),
                          key=lambda g: (g["replicas"], -(g["hi"] - g["lo"])))
    if decoder_gaps and max_decoder_layers > 0:
        g = decoder_gaps[0]
        hi = min(g["hi"], g["lo"] + max_decoder_layers)
        parts.append(f"decoder:{g['lo']}-{hi}")
    if take_embed_head and gaps.get("head_replicas", 0) < target:
        parts.append("head")
    return ",".join(parts)
