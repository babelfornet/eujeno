from eujeno.net.metrics import NodeMetrics

def test_requests_and_throughput():
    m = NodeMetrics()
    m.inc_request(); m.inc_request(3)
    assert m.requests_served == 4
    m.record_job(10, 2.0); m.record_job(10, 2.0)
    assert m.throughput_tok_s() == 5.0   # 20 tokens / 4.0s total
    assert m.uptime_sec() >= 0

def test_latency_and_speed_prefers_faster():
    m = NodeMetrics(ewma_alpha=1.0)  # alpha=1 -> last value wins (deterministic)
    m.observe_hop_time("fast", 0.10)
    m.observe_hop_time("slow", 1.00)
    sp = m.speed_map(["fast", "slow", "new"])
    assert sp["fast"] > sp["slow"]            # faster -> higher score
    assert sp["new"] > 0                       # unmeasured gets a neutral default
    m.observe_latency("fast", 20); m.observe_latency("slow", 80)
    assert m.avg_latency_ms() == 50
