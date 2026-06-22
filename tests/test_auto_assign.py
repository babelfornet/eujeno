from eujeno.cli import stages_from_registry
from eujeno.net.discovery import coverage_gaps


def test_stages_from_registry_coordinator_list():
    # coordinator /registry shape: a list of {conn, stages}
    nodes = [{"conn": "c1", "stages": {"embed": True, "head": False, "decoders": ["0-12"]}}]
    out = stages_from_registry(nodes)
    assert out == {"c1": {"embed": True, "head": False, "decoders": ["0-12"]}}


def test_stages_from_registry_gossip_dict():
    nodes = {"http://n1": {"embed": True, "head": False, "decoders": ["0-12"]}}
    assert stages_from_registry(nodes) == nodes


def test_stages_from_registry_empty():
    assert stages_from_registry({}) == {}
    assert stages_from_registry([]) == {}


def test_coverage_gap_from_coordinator_list():
    # the regression: a node already covers embed+0-12 (coordinator list shape);
    # the gap must come out as decoder 12-24 + head, NOT the whole model.
    nodes = [{"conn": "c1", "stages": {"embed": True, "head": False, "decoders": ["0-12"]}}]
    gaps = coverage_gaps(stages_from_registry(nodes), num_layers=24, target=1)
    assert gaps["decoder_gaps"] == [{"lo": 12, "hi": 24, "replicas": 0}]
    assert gaps["head_replicas"] == 0      # head still uncovered
    assert gaps["embed_replicas"] == 1     # embed already covered


def test_coverage_gap_ignored_when_list_not_parsed():
    # guards the bug: if the list weren't normalized to a dict, coverage would be
    # read as empty and the whole 0-24 range would look uncovered.
    nodes = [{"conn": "c1", "stages": {"embed": True, "head": False, "decoders": ["0-12"]}}]
    empty = coverage_gaps({}, num_layers=24, target=1)        # the buggy path
    assert empty["decoder_gaps"] == [{"lo": 0, "hi": 24, "replicas": 0}]
    fixed = coverage_gaps(stages_from_registry(nodes), num_layers=24, target=1)
    assert fixed["decoder_gaps"] != empty["decoder_gaps"]
