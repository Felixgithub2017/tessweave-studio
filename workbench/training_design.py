"""Bounded, explainable training search. Analytical candidates, never launch approval.

Rule snapshot 2026-10-07. Independent of the legacy capacity calculator.
No peak-FLOPS table masquerades as measured throughput. Cost requires user inputs.
"""
import math
from collections import Counter

GIB = 2**30
SOURCES = [
    'https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html',
    'https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html',
    'https://docs.nvidia.com/nemo/megatron-bridge/latest/training/memory-estimator.html',
    'https://docs.nvidia.com/nemo/megatron-bridge/latest/training/moe-optimization.html',
    'https://deepspeed.readthedocs.io/en/latest/zero3.html',
    'https://github.com/pytorch/torchtitan',
]


def design_training(body, legacy, model=None):
    from .resources import number
    def num(k, default, low=0, high=1e18, integer=False):
        return number(body,k,default,low,high,integer)
    cfg=(model or {}).get('config') or {}
    dims=legacy['dims'];p=legacy['parameters'];L=dims['layers'];d=dims['hidden']
    s=num('seq',4096,128,1000000,True);b=num('batch',1,1,65536,True)
    total=num('train_tokens_b',0)*1e9
    target=num('global_batch_tokens',1048576,128,1e10,True)
    packing=num('packing_efficiency',1,.01,1)
    deadline=num('deadline_hours',0);price=num('gpu_hour_price',0)
    compute=num('effective_tflops',0);intra=num('intra_gib_s',0);inter=num('inter_gib_s',0)
    hbm=num('hbm_gib_s',0);overhead=num('overhead_percent',15,0,200)/100
    node=num('gpus_per_node',8,1,16,True);fixed=num('gpu_count',0,0,4096,True)
    maximum=num('max_search_gpus',256,1,4096,True)
    fabric=body.get('node_fabric','unknown')
    if fabric not in ('unknown','pcie','nvlink'):raise ValueError('Invalid node_fabric')
    phase=body.get('training_phase','cpt')
    from .knowledge_base import planning_evidence
    evidence=planning_evidence(body,model)
    result={'version':'2026-10-07.v1','status':'blocked','sources':SOURCES,
            'blockers':[],'recommendations':{},'rejected':{},'candidates':[],
            'measured':False,'launch_ready':False,
            'assumptions':{'parameters':p,'seq':s,'microbatch':b,'total_tokens':total,
                'global_batch_tokens_target':target,'packing_efficiency':packing,
                'effective_tflops':compute,'intra_gib_s':intra,'inter_node_gib_s_per_node':inter,
                'hbm_gib_s':hbm,'gpu_hour_price':price,'deadline_hours':deadline,
                'overhead_percent':overhead*100,'node_fabric':fabric,'max_search_gpus':maximum},
            'limits':['Analytical uniform-layer proxy, not a measured liveness trace or backend validation.',
                'Selected GPU only; BF16 weights, Megatron FP32 gradients (DeepSpeed BF16 assumption), FP32 Adam; no FP8/offload search.',
                'TP kept within an explicitly NVLink node; EP within node; PP may cross nodes.',
                'EP subset of DP; EP is not an extra world-size multiplier. MoE search restricts TP=CP=1.',
                'Timing assumes causal dense attention, fused attention and 1F1B; packing uses fixed-length sequences.',
                'Batch target is an upper bound for enumeration, not a recommended optimization hyperparameter.',
                'No model quality, convergence, checkpoint recovery, or engine support guarantee.'],
            'validation':['Verify model conversion and backend version first.',
                'Measure NCCL collectives on actual node/NIC topology and representative message sizes.',
                'Warm up then record at least 20 steady-state steps: max allocated/reserved memory, step time, loss and effective tokens.',
                'Test checkpoint save/resume and compare loss/gradients against a small reference run.',
                'Replace assumed throughput/bandwidth with measurements; re-run search before a long rental.']}
    if phase not in ('pretrain','cpt','sft'):
        result['blockers'].append('DPO/PPO/GRPO require role-specific training/rollout pools; single-model search cannot represent them.')
    result['knowledge_evidence']=evidence
    result['validation'].extend(r['en'] for r in evidence['rules'])
    result['assumptions']['optimizer']=body.get('optimizer','adamw')
    if body.get('optimizer','adamw') not in ('adam','adamw'):
        result['blockers'].append('Selected optimizer needs a state-layout adapter; the Adam ledger cannot estimate Muon or a mixed optimizer.')
    result['assumptions']['packed_sequences_per_update_upper_bound']=target//s
    result['assumptions']['sequence_count_note']='Packed sequences, not independent documents; document boundaries and routing must be measured.'
    if body.get('extra_gpus',0):
        result['blockers'].append('Use fixed GPU count or search ceiling for joint planning; legacy extra-GPU input has no joint-search interpretation.')
    if legacy['mode']!='full':
        result['blockers'].append('Joint search currently covers full-parameter Adam; LoRA/QLoRA remain in the separate capacity audit.')
    if not all(dims[k]>0 for k in ('layers','hidden','heads','ffn','vocab')):
        result['blockers'].append('Supply layer count, hidden width, attention heads, FFN width and vocabulary from config; no size-based guessing.')
    inventory=(model or {}).get('architecture_inventory') or {}
    result['architecture_inventory']=inventory
    special=bool(cfg.get('layer_types') or cfg.get('hybrid_layer_pattern') or cfg.get('kv_lora_rank') or (model or {}).get('modality') not in (None,'text','text-generation'))
    if special and not inventory.get('complete'):
        result['blockers'].append('Hybrid attention, MLA or non-text architecture needs an architecture-specific activation/compute adapter; ordinary dense-attention estimate disabled.')
    experts=int(cfg.get('n_routed_experts') or cfg.get('num_local_experts') or cfg.get('num_experts') or num('expert_count',0,0,4096,True))
    frac=num('expert_parameter_fraction',0,0,.9999)
    if inventory.get('complete'):
        frac=inventory['expert_parameter_fraction']
        result['assumptions']['expert_fraction_source']='complete tensor-header inventory'
    topk=int(cfg.get('num_experts_per_tok') or cfg.get('moe_router_topk') or num('expert_topk',0,0,4096,True))
    if experts and (not frac or not topk or topk>experts):
        result['blockers'].append('MoE requires routed-expert parameter fraction and top-k (shared experts belong in the dense fraction).')
    if result['blockers']:return result
    result['assumptions'].update(experts=experts,expert_parameter_fraction=frac,expert_topk=topk)
    if special:
        from .structural_capacity import structural_capacity
        result['status']='partial'
        result['structural_capacity']=structural_capacity(body,legacy,inventory)
        result['limits']=['Exact stored-element state accounting; activation and communication peaks remain unmeasured. All stored elements conservatively assumed trainable. Backend model adaptation is not verified.']
        return result
    gpu=legacy['selected'];cap=gpu['vram_gib'];usable=cap*(1-legacy['reserve_percent']/100)
    active=p*(1-frac+frac*topk/experts) if experts else p
    extra=sum(num(k,0,0,100000) for k in ('runtime_gib','communication_gib','cuda_graph_gib','other_models_gib','rollout_gib'))
    work=num('workspace_gib',4,.1,1024)+extra
    floor=num('activation_gib',8,0,100000)
    def powers(limit):return [2**i for i in range(13) if 2**i<=limit]
    sizes=[fixed] if fixed else sorted(set(powers(min(node,maximum))+[node*x for x in powers(maximum//node)]))
    rejected=Counter();candidates=[];examined=0
    for n in sizes:
      for tp in powers(min(node,n)):
       if n%tp or node%tp or (tp>1 and fabric!='nvlink'):continue
       if any(dims[k]%tp for k in ('hidden','heads','ffn','vocab')):continue
       for cp in powers(min(8,n//tp)):
        if n%(tp*cp) or s%(2*cp) or (cp>1 and s<8192):continue
        for pp in powers(min(L,n//tp//cp)):
         if n%(tp*cp*pp) or L%pp:continue
         dp=n//(tp*cp*pp)
         # Global sample batch is invariant across TP/PP/CP. Never silently increase target.
         acc=target//(b*s*dp)
         if not acc or acc>4096:
            rejected['global_batch_or_accumulation_limit']+=1;continue
         ep_options=[x for x in powers(min(node,dp,experts)) if dp%x==0 and experts%x==0 and node%x==0] if experts else [1]
         if experts and (tp!=1 or cp!=1):continue
         for ep in ep_options:
          # Framework families, not arbitrary combinations of every flag.
          strategies=[('megatron',0),('megatron',1)]
          if tp==pp==cp==ep==1 and not experts:strategies += [('deepspeed',2),('deepspeed',3)]
          for backend,z in strategies:
           for checkpoint in ('full','none'):
            examined+=1
            shard=tp*pp
            # Worst-stage allowance: uniform body + extra embedding/output footprint.
            edge=2*d*dims['vocab']/tp if pp>1 else 0
            dense=p*(1-frac if experts else 1)/shard+edge
            expert=p*frac/pp/ep if experts else 0
            resident=dense+expert
            state_dp=dp*cp if backend=='megatron' else dp
            expert_dp=dp//ep
            def state(bytes_,level):
                return bytes_*(dense/(state_dp if z>=level else 1)+expert/(expert_dp if z>=level else 1))/GIB
            rows=[]
            def row(key,value,formula,precision):
                width={'weights':2,'gradients':4 if backend=='megatron' else 2,'master':4,'adam_m':4,'adam_v':4,'logits':4,'dispatch':2,'gather':2}.get(key)
                rows.append({'item':key,'per_gpu_gib':value,'calculation':formula,'precision':precision,'source':'analytical_proxy','basis':formula,'whole_instance_gib':None,
                             'bytes':value*GIB,'bytes_per_element':width,'elements':value*GIB/width if width else None})
            grad_bytes=4 if backend=='megatron' else 2
            for key,bytes_,level in [('weights',2,3),('gradients',grad_bytes,2),('master',4,1),('adam_m',4,1),('adam_v',4,1)]:
                row(key,state(bytes_,level),f'{bytes_} × ({dense:,.0f}/{state_dp if z>=level else 1} + {expert:,.0f}/{expert_dp if z>=level else 1}) / 2^30', 'BF16 · 2 bytes' if bytes_==2 else 'FP32 · 4 bytes')
            alive=min(pp,acc)
            saved=(2 if checkpoint=='full' else 24)*b*(s/cp)*d*(L/pp)/tp*alive/GIB
            temp=12*b*(s/cp)*max(d,dims['ffn'])/tp/GIB
            # Conservative unchunked logits; every rank budgeted as worst stage.
            logits=4*b*(s/cp)*dims['vocab']/tp/GIB
            dispatch=4*b*s*d*topk/GIB if experts else 0
            row('saved_activations',saved,f'{2 if checkpoint=="full" else 24} × B{b} × S{s}/CP{cp} × d{d} × L{L}/PP{pp}/TP{tp} × live microbatches {alive} / 2^30','BF16 boundaries' if checkpoint=='full' else 'mixed activation proxy · coefficient 24')
            row('transient',temp,f'12 × {b} × {s}/{cp} × max({d},{dims["ffn"]})/{tp}/2^30','mixed proxy')
            row('logits',logits,f'4 × {b} × {s}/{cp} × V{dims["vocab"]}/{tp}/2^30','FP32')
            row('activation_extra',max(0,floor-saved-temp-logits),f'max(0, {floor} − structural activations)','manual lower bound')
            row('dispatch',dispatch,f'MoE send/receive: 4 × {b} × {s} × {d} × {topk} / 2^30 = {dispatch:.6f} GiB','BF16 proxy')
            gather=2*resident/(L/pp)/GIB if z==3 and dp>1 else 0
            row('gather',gather,f'2 × {resident:,.3f} / ({L}/{pp}) / 2^30 = {gather:.6f} GiB' if gather else '0: no modeled block gather; prefetch needs separate budget','BF16')
            row('workspace',work,f'workspace {num("workspace_gib",4,.1,1024)} + explicit extras {extra} = {work:.6f} GiB','mixed / manual budget')
            peak=sum(x['per_gpu_gib'] for x in rows)
            if peak>usable:rejected['memory_capacity']+=1;continue
            tokens=b*s*dp*acc*packing
            # Bytes on each rank; deliberately no assumed overlap in upper bound.
            comm={
              'tp':8*b*s*d*(L/pp)*acc*(tp-1)/tp/GIB if tp>1 else 0,
              'cp':4*b*s*d*(L/pp)*acc*(cp-1)/cp/GIB if cp>1 else 0,
              'pp':4*b*s*d/tp/cp*acc/GIB if pp>1 else 0,
              'dp':2*grad_bytes*resident*(state_dp-1)/state_dp/GIB if state_dp>1 else 0,
              'ep':8*b*s*d*topk*(L/pp)*acc*(ep-1)/ep/GIB if ep>1 else 0,
              'zero_gather':4*resident*acc*(dp-1)/dp/GIB if z==3 and dp>1 else 0}
            # Traffic changes with ZeRO scheduling; levels 1/2 add a BF16 weight gather.
            if z in (1,2) and state_dp>1:comm['dp']+=2*resident*(state_dp-1)/state_dp/GIB
            flops=(6*active*b*s*dp*acc+12*b*dp*acc*L*s*s*d)*(4/3 if checkpoint=='full' else 1)
            seconds_compute=flops/n/(compute*1e12) if compute else None
            # Lower bound on state traffic, NOT all activation traffic or a roofline proof.
            hbm_lower=resident*16*acc/GIB/hbm if hbm else None
            seconds_comm=0.;network_known=True
            nodes=math.ceil(n/node)
            for axis,traffic in comm.items():
                if not traffic:continue
                bandwidth=intra if nodes==1 or axis in ('tp','ep') else inter/node
                if not bandwidth:network_known=False;continue
                seconds_comm+=traffic/bandwidth
            bubble=(pp-1)/acc
            # Two sensitivity endpoints, not a statistical confidence interval.
            time_range=None;hours=None;cost=None
            if seconds_compute is not None and network_known:
                base=max(seconds_compute,hbm_lower or 0)
                time_range=[max(base,seconds_comm)*(1+bubble), (base+seconds_comm)*(1+bubble)]
                if total:
                    steps=math.ceil(total/tokens)
                    hours=[v*steps*(1+overhead)/3600 for v in time_range]
                    if price:cost=[v*n*price for v in hours]
            walls=[]
            if peak/usable>.9:walls.append('memory_capacity')
            if seconds_compute and network_known and seconds_comm>seconds_compute*.3:walls.append('communication')
            if hbm_lower and seconds_compute and hbm_lower>seconds_compute:walls.append('HBM_bandwidth_lower_bound')
            if bubble>.25:walls.append('pipeline_bubble')
            if not walls:walls.append('unprofiled')
            candidates.append({'id':f'{n}-{tp}-{pp}-{cp}-{ep}-{backend}-{z}-{checkpoint}',
                'gpus':n,'nodes':nodes,'gpu':gpu['name'],'tp':tp,'pp':pp,'cp':cp,'dp':dp,'ep':ep,
                'zero_stage':z,'state_strategy':'distributed optimizer (ZeRO-1-like)' if backend=='megatron' and z==1 else f'ZeRO-{z}',
                'backend_family':backend,'sequence_parallel':tp>1,'checkpoint':checkpoint,
                'microbatch':b,'gradient_accumulation':acc,'effective_batch_tokens':tokens,
                'peak_gib':peak,'headroom_gib':usable-peak,'rows':rows,
                'embedding_stage_allowance_parameters':edge,'comm_gib_per_rank_step':comm,
                'compute_seconds':seconds_compute,'communication_seconds_proxy':seconds_comm if network_known else None,
                'hbm_seconds_lower_bound':hbm_lower,'pipeline_bubble_ratio':bubble,
                'step_seconds_range':time_range,'hours_range':hours,'cost_range':cost,
                'steps':math.ceil(total/tokens) if total else None,'bottlenecks':walls,
                'timing_basis':'user-input analytical sensitivity; not measured',
                'launch_ready':False,
                'configuration_blueprint':{
                    'status':'partial settings only; model recipe and version preflight required',
                    'tensor_model_parallel_size':tp,'pipeline_model_parallel_size':pp,
                    'context_parallel_size':cp,'expert_model_parallel_size':ep,
                    'sequence_parallel':tp>1,'micro_batch_size':b,
                    'global_batch_size':b*dp*acc,'seq_length':s,
                    'recompute':checkpoint,'bf16':True,
                    'use_distributed_optimizer':backend=='megatron' and z==1,
                    'deepspeed_zero_stage':z if backend=='deepspeed' else None}})
    result['rejected']=dict(rejected);result['examined']=examined;result['feasible_count']=len(candidates)
    if not candidates:
        result['blockers'].append('No candidate in the bounded search. Check topology, batch, memory or raise search ceiling; this is not proof no solution exists.')
        return result
    # Stable, bounded objective: minimal hardware first. Communication proxy only tie-breaks.
    def capacity_score(x):return (x['gpus'],sum(x['comm_gib_per_rank_step'].values()),x['pipeline_bubble_ratio'],x['zero_stage'],x['checkpoint']!='none',x['peak_gib'],x['id'])
    minimum=min(candidates,key=capacity_score)
    timed=[x for x in candidates if x['hours_range']]
    priced=[x for x in timed if x['cost_range']]
    cheapest=min(priced,key=lambda x:x['cost_range'][1]) if priced else None
    within=[x for x in timed if deadline and x['hours_range'][1]<=deadline]
    ontime=min(within,key=lambda x:(x['cost_range'][1] if x['cost_range'] else x['gpus'],x['hours_range'][1])) if within else None
    result.update(status='estimated',recommendations={'minimum_capacity':minimum,'lowest_estimated_cost':cheapest,'deadline':ontime})
    result['candidates']=sorted(candidates,key=capacity_score)[:24]
    result['selection_note']='Minimum across enumerated candidates only; costs/times require tokens and effective hardware measurements. No timing means no fastest/cheapest claim.'
    return result
