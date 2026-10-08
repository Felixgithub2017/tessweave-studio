"""Architecture-independent state accounting for audited floating checkpoints.

Produces conditional topology blueprints with activation headroom, not a claim
that arbitrary hybrid kernels/frameworks can train the checkpoint.
"""
import math

GIB=2**30

def structural_capacity(body, legacy, inventory):
    from .resources import number
    def num(k,d,lo=0,hi=4096):return number(body,k,d,lo,hi,True)
    p=inventory['total_elements'];layers=inventory['layers'];L=len(layers)
    gpu=legacy['selected'];capacity=gpu['vram_gib']
    usable=capacity*(1-legacy['reserve_percent']/100)
    node=num('gpus_per_node',8,1,16);fixed=num('gpu_count',0);maximum=num('max_search_gpus',256,1)
    maximum=fixed or maximum
    E=inventory['expert_count']
    work=number(body,'workspace_gib',4,.1,1024)+sum(number(body,k,0,0,100000) for k in ('runtime_gib','communication_gib','cuda_graph_gib','other_models_gib','rollout_gib'))
    outside=sum(inventory['outside_decoder'].values())
    largest=max([x['elements'] for x in layers]+[outside])
    results=[];candidate_count=0
    def assemble(n,pp,dp,ep,z,backend,rank_rows):
        nonlocal candidate_count
        peak=max(x['accounted_gib'] for x in rank_rows)
        if peak>=usable:return
        for rank in rank_rows:
            rank['ledger']=[]
            for key,bits in [('weights',16),('gradients',16 if z==3 else 32),('master',32),('adam_m',32),('adam_v',32),('gather',16)]:
                bytes_=rank[key+'_gib']*GIB
                quantity=bytes_/(bits/8)
                if z==3:
                    term=f'{p:,} / {dp}' if key!='gather' else (f'{largest:,}' if dp>1 else '0')
                else:
                    term=rank['resident_expression'] if key in ('weights','gradients') else rank['optimizer_expression'] if key!='gather' else '0'
                rank['ledger'].append({'item':key,'precision':'BF16' if bits==16 else 'FP32',
                    'bytes_per_element':bits/8,'elements':quantity,'bytes':bytes_,'gib':bytes_/GIB,
                    'formula':f'({term}) × {bits//8} bytes / 1,073,741,824 = {bytes_/GIB:.6f} GiB',
                    'basis':'balanced state shard estimate; padding/allocator overhead excluded' if z==3 else 'header-derived placement; optimizer shards averaged'})
            rank['ledger'].append({'item':'workspace','precision':'mixed / budget','elements':None,
                'bytes_per_element':None,'bytes':work*GIB,'gib':work,
                'formula':' + '.join(f'{k}={float(body.get(k,d)):.3f}' for k,d in [('workspace_gib',4),('runtime_gib',0),('communication_gib',0),('cuda_graph_gib',0),('other_models_gib',0),('rollout_gib',0)])+f' = {work:.6f} GiB',
                'basis':'explicit per-rank budgets; not measured tensor allocations'})
        candidate_count+=1
        results.append({'gpus':n,'nodes':math.ceil(n/node),'tp':1,'cp':1,'pp':pp,'dp':dp,'ep':ep,
            'zero_stage':z,'backend_family':backend,'accounted_peak_gib':peak,
            'remaining_for_activations_gib':usable-peak,'ranks':rank_rows,
            'status':'state_fits_only','training_feasible':None,'launch_ready':False,
            'mapping':'PP stages contiguous; EP nested in DP; all non-decoder modules conservatively placed on first stage',
            'batch_constraint':'Global training samples = microbatch × DP × accumulation; EP not an extra multiplier'})
        results.sort(key=lambda x:(x['gpus'],-x['remaining_for_activations_gib'],x['pp'],x['ep']))
        del results[8:]
    # Pure data-parallel ZeRO-3: conditional adapter required, but state arithmetic is architecture independent.
    sizes=[fixed] if fixed else list(range(1,min(node,maximum)+1))+list(range(node*2,maximum+1,node))
    for n in sizes:
        state=p*16/n/GIB;gather=2*largest/GIB if n>1 else 0
        rank={'state_gib':state,'gather_gib':gather,'workspace_gib':work,'accounted_gib':state+gather+work,
              'weights_gib':2*p/n/GIB,'gradients_gib':2*p/n/GIB,'master_gib':4*p/n/GIB,'adam_m_gib':4*p/n/GIB,'adam_v_gib':4*p/n/GIB,
              'layer_start':0,'layer_end':L,'expert_ids':'all, parameters sharded by DP',
              'formula':f'16 × {p:,} / DP {n} / 2^30 + gather {gather:.3f} + workspace {work:.3f}',
              'precision':'BF16 weights/gradients; FP32 master/m/v; all stored elements assumed trainable'}
        assemble(n,1,n,1,3,'deepspeed-like (adapter unverified)',[dict(rank,rank=i,pp_rank=0,dp_rank=i,ep_rank=0) for i in range(n)])
        # Larger pure DP allocations are represented by the first fitting allocation unless fixed.
        if results:break
    # Enumerate actual layer-count divisors, not only powers of two (e.g. 45 -> 3,5,9,15).
    for pp in [x for x in range(1,L+1) if L%x==0]:
      width=L//pp
      partitions={}
      for stage in range(pp):
        part=layers[stage*width:(stage+1)*width]
        dense=sum(x['elements']-x['routed_elements'] for x in part)+(outside if stage==0 else 0)
        for ep in range(1,min(E or 1,node)+1):
          partitions[stage,ep]=(dense,[sum(sum(v for k,v in x['experts'].items() if int(k)%ep==lane) for x in part) for lane in range(ep)])
      for dp in range(1,maximum//pp+1):
       n=pp*dp
       if fixed and n!=fixed:continue
       if n>node and n%node:continue
       if dp>node and dp%node:continue
       for ep in [x for x in range(1,min(E or 1,node,dp)+1) if dp%x==0 and (not E or E%x==0) and node%x==0]:
        ranks=[]
        for stage in range(pp):
          dense,expert_parts=partitions[stage,ep]
          for lane in range(dp):
            expert=expert_parts[lane%ep]
            resident=dense+expert;opt=dense/dp+expert/(dp//ep)
            state=(6*resident+12*opt)/GIB
            ranks.append({'rank':stage*dp+lane,'pp_rank':stage,'dp_rank':lane,'ep_rank':lane%ep,
                'layer_start':stage*width,'layer_end':(stage+1)*width,
                'expert_ids':f'id % {ep} == {lane%ep}',
                'weights_gib':2*resident/GIB,'gradients_gib':4*resident/GIB,'master_gib':4*opt/GIB,
                'resident_expression':f'{dense:,} + {expert:,}',
                'optimizer_expression':f'{dense:,}/{dp} + {expert:,}/{dp//ep}',
                'adam_m_gib':4*opt/GIB,'adam_v_gib':4*opt/GIB,'state_gib':state,
                'gather_gib':0,'workspace_gib':work,'accounted_gib':state+work,
                'formula':f'(6 × ({dense:,}+{expert:,}) + 12 × ({dense:,}/{dp}+{expert:,}/{dp//ep})) / 2^30 + workspace {work:.3f}',
                'precision':'BF16 weights; FP32 gradients/master/m/v; distributed optimizer'})
        assemble(n,pp,dp,ep,1,'megatron-like (adapter unverified)',ranks)
    results.sort(key=lambda x:(x['gpus'],-x['remaining_for_activations_gib'],x['pp'],x['ep']))
    return {'status':'partial','kind':'header_based_state_capacity','candidates':results[:8],
        'candidate_count':candidate_count,'training_feasible':None,'seq':body.get('seq',4096),
        'unmodeled':['Saved activations and transient kernel buffers for linear/sparse attention, mHC, MTP and vision.',
                     'Expert token imbalance, communication peak buffers, pipeline scheduling and backend model support.'],
        'explanation':'Counts and state partitioning come from all tensor headers. Remaining capacity is a budget for unmeasured activations, NOT free memory or proven training feasibility.',
        'empty_reason':None if results else 'No state-capacity candidate within the selected hardware/search ceiling; full training needs still more memory.',
        'checkpoint_dtype_elements':inventory['dtypes']}
