"""Decision explanations derived from actual solver outcomes, not LLM narration."""
def explain(design):
    a=design['assumptions'];partial=design.get('structural_capacity')
    def node(key,en,zh,status,facts,rules):
        return dict(id=key,title_en=en,title_zh=zh,status=status,facts=facts,rule_ids=rules)
    evidence=design.get('knowledge_evidence',{})
    steps=[
        node('inputs','1 · Identify workload','1 · 确认模型与任务','checked',
             {'parameters':a['parameters'],'sequence_tokens':a['seq'],'microbatch':a['microbatch'],
              'optimizer':a.get('optimizer'),'total_tokens':a['total_tokens']},['state-layout']),
        node('support','2 · Check estimator applicability','2 · 判断估算适用范围',
             'blocked' if design['blockers'] else 'partial' if partial else 'checked',
             {'blockers':design['blockers'],'state_only':bool(partial)},['state-layout','multimodal-freezing','role-liveness']),
        node('search','3 · Constrain and enumerate layouts','3 · 约束与枚举并行方案',
             'partial' if partial else 'checked' if design.get('examined') else 'not_run',
             {'examined_after_shape_filters':design.get('examined'),
              'state_fitting_candidates':partial.get('candidate_count') if partial else None,
              'feasible_candidates':design.get('feasible_count'),'rejected':design.get('rejected'),
              'gpu_search_ceiling':a['max_search_gpus']},['three-walls']),
        node('rank','4 · Rank by explicit objectives','4 · 按明确目标推荐','partial' if partial else design['status'],
             {'objective':'fewest GPUs; then capacity headroom' if partial else 'fewest GPUs → communication proxy → pipeline bubble → sharding → recomputation → memory',
              'time_cost_available':bool(design.get('recommendations',{}).get('lowest_estimated_cost')),
              'effective_tflops':a['effective_tflops'],'inter_node_gib_s':a['inter_node_gib_s_per_node'],
              'hbm_gib_s':a['hbm_gib_s'],'gpu_hour_price':a['gpu_hour_price']},['three-walls']),
        node('validate','5 · Calibrate before execution','5 · 实测校准后再执行','required',
             {'launch_ready':False,'packed_sequences_upper_bound':a.get('packed_sequences_per_update_upper_bound'),
              'checks':design['validation']},['scaling-pilot','length-gates','routing-gates'])]
    known={r['id']:r for r in evidence.get('rules',[])}
    for step in steps:step['rules']=[known[x] for x in step['rule_ids'] if x in known]
    design['decision_trace']=steps
    return design
