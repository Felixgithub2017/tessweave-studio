"""Traffic-to-replica scenario calculator, not a hardware performance oracle.

Single logical model and identical, warmed replicas. mu is jointly measured
prefill+decode capacity under the SAME workload, SLO, precision and topology.
User-entered calibration is always labelled unverified; no rental/launch occurs.
"""
import math


def plan_serving(body):
    def number(key, default, lo, hi, integer=False):
        value = body.get(key, default)
        if isinstance(value, bool):
            raise ValueError(f"{key}: expected a number")
        try:
            value = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{key}: expected a number") from None
        if not math.isfinite(value) or not lo <= value <= hi or integer and value != int(value):
            raise ValueError(f"{key}: outside [{lo}, {hi}]")
        return int(value) if integer else value

    daily = number('daily_requests', 100000, 1, 1e12)
    peak_factor = number('peak_factor', 10, 1, 10000)
    calls = number('calls_per_request', 1, .001, 1000)
    peak_override = number('peak_rps', 0, 0, 1e9)
    pin = number('mean_input_tokens', 2000, 1, 1e8)
    pout = number('mean_output_tokens', 500, 1, 1e8)
    maxout = number('max_output_tokens', 2048, pout, 1e8)
    mu = number('replica_rps', 0, 0, 1e9)
    headroom = number('headroom_percent', 20, 0, 90) / 100
    gpu = number('gpus_per_replica', 1, 1, 65536, True)
    per_domain = number('replicas_per_domain', 1, 1, 1024, True)
    failures = number('failed_domains', 1, 0, 100, True)
    price = number('gpu_hour_price', 0, 0, 1e6)
    first_slo = number('first_text_slo_s', 2, .001, 86400)
    total_slo = number('total_slo_s', 30, first_slo, 86400)
    measured_first = number('observed_first_text_p95_s', 0, 0, 86400)
    measured_total = number('observed_total_p95_s', 0, 0, 86400)
    error_rate = number('observed_error_percent', 0, 0, 100)
    error_budget = number('error_budget_percent', 1, 0, 100)
    average = daily / 86400 * calls
    peak = peak_override * calls if peak_override else average * peak_factor
    trace = [dict(step='traffic', formula='model average RPS = daily business requests / 86400 × calls per request', value=average),
             dict(step='peak', formula='model peak RPS = explicit business peak RPS × calls OR average model RPS × peak factor', value=peak)]
    blockers = []
    if not mu or not measured_first or not measured_total:
        blockers.append('Provide joint per-replica RPS and observed p95 first-text/total latency from an open-loop load test.')
    if measured_first > first_slo or measured_total > total_slo or error_rate > error_budget:
        blockers.append('Calibration violates the requested SLO/error budget. Use a lower tested arrival rate; adding replicas does not fix an oversized request.')
    report = dict(schema='tessweave.serving-plan.v1', evidence='user_entered_scenario_not_verified',
                  average_model_rps=average, peak_model_rps=peak,
                  peak_input_tokens_s=peak * pin, peak_output_tokens_s=peak * pout,
                  max_output_tokens=maxout, blockers=blockers, decision_trace=trace,
                  normal=None, resilient=None,
                  assumptions=dict(headroom_percent=headroom * 100, replica_rps=mu, gpus_per_replica=gpu,
                                   replicas_per_domain=per_domain, failed_domains=failures,
                                   first_text_slo_s=first_slo, total_slo_s=total_slo),
                  caveats=[
                      'First-text latency is not token-level TTFT; p95 checks are not an availability guarantee.',
                      'Replica capacity must cover the joint input/output length distribution, bursts and cache regime, not independent peak token rates.',
                      'GPU count per replica must first pass model/runtime, context, KV and per-rank memory qualification.',
                      'A failure domain contains at most the configured replicas. Multi-node replicas require disjoint node groups; availability zones may need different capacity curves.',
                      'Replica scaling assumes identical warm replicas and balanced routing; validate routing, shared storage/network bottlenecks and node-loss drills.',
                      'Max output is an admission limit, not the average. Stress long-tail requests separately. No automatic cloud spend or deployment.',
                  ])
    if blockers:
        return report
    usable = mu * (1 - headroom)
    normal = max(1, math.ceil(peak / usable))
    domains = math.ceil(normal / per_domain) + failures
    resilient = domains * per_domain
    if resilient > 1000000:
        raise ValueError('Replica scenario exceeds one million; check units and calibration')
    trace.extend([
        dict(step='capacity', formula=f'usable RPS / replica = {mu} × (1 − {headroom})', value=usable),
        dict(step='normal', formula=f'ceil({peak} / {usable})', value=normal),
        dict(step='fault_domains', formula=f'(ceil({normal} / {per_domain}) + {failures}) × {per_domain}', value=resilient),
        dict(step='survivors', formula=f'({domains} − {failures}) × {per_domain} × {usable} ≥ {peak}', value=(domains-failures)*per_domain*usable),
    ])
    def scenario(replicas):
        return dict(replicas=replicas, gpus=replicas*gpu, usable_rps=replicas*usable,
                    hourly_gpu_cost=replicas*gpu*price if price else None)
    report.update(normal=scenario(normal), resilient={**scenario(resilient), 'domains': domains,
                  'surviving_replicas': (domains-failures)*per_domain})
    return report
