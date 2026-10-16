'use strict';
function decisionTraceView(parent,design){
  if(!design.decision_trace)return;const tr=(en,zh)=>language==='zh'?zh:en;
  const section=el('section',undefined,'decision-trace');section.append(el('h3',tr('Why this plan?','为什么得到这个方案？')),el('p',tr('Actual solver path. Source-backed principles are not universal recipes. Expand a step for inputs and exclusions.','实际求解路径。报告支持的是原则，不是通用配方；展开步骤查看输入与淘汰原因。')));parent.append(section);
  const flow=el('div',undefined,'decision-steps');section.append(flow);
  for(const step of design.decision_trace){const d=el('details',undefined,'decision-step');d.dataset.status=step.status;
    d.append(el('summary',(language==='zh'?step.title_zh:step.title_en)+' · '+step.status));
    for(const rule of step.rules){d.append(el('p',language==='zh'?rule.zh:rule.en));const links=el('div');for(const id of rule.sources){const source=design.knowledge_evidence.sources.find(s=>s.id===id);if(source){const a=el('a',source.vendor+' · '+id);a.href=source.url;a.target='_blank';a.rel='noopener noreferrer';links.append(a,document.createTextNode(' '));}}d.append(links);}
    table(d,[tr('Input / result','输入／结果'),tr('Value','数值')],Object.entries(step.facts).map(([k,v])=>[k,v===null?'—':typeof v==='object'?JSON.stringify(v,null,2):String(v)]));flow.append(d);
  }
}
function knowledgeEvidenceView(parent,design){
  const e=design?.knowledge_evidence;if(!e)return;const tr=(en,zh)=>language==='zh'?zh:en;
  const box=el('details');box.append(el('summary',tr('Research basis and applicability','技术报告依据与适用边界')+' · '+e.version));parent.append(box);
  box.append(el('p',tr('Curated offline evidence, not an automatic reproduction of the vendor run.','离线核查的证据，不代表自动复现厂商训练。')));
  table(box,[tr('Rule','规则'),tr('Planning consequence','规划影响')],e.rules.map(x=>[x.id,language==='zh'?x.zh:x.en]));
  if(e.published_reference){const r=e.published_reference;box.append(el('p',tr('Published reference: ','公开训练参考：')+r.lengths_label+' · '+r.locator),el('p',language==='zh'?r.note_zh:r.note_en));}
  for(const s of e.sources){const a=el('a',s.vendor+' · '+s.title);a.href=s.url;a.target='_blank';a.rel='noopener noreferrer';const p=el('p');p.append(a,el('small',' · '+s.scope));box.append(p);}
  box.append(el('small','Rules SHA256: '+e.rules_sha256));
}
function lengthCurriculumView(parent,r){
  const c=r.length_curriculum;if(!c)return;const tr=(en,zh)=>language==='zh'?zh:en;
  const box=el('section',undefined,'plan-basis');parent.append(box);
  box.append(el('h3',tr('Staged sequence-length plan','分阶段训练长度规划')),
    el('p',tr('Engineering starting point, not the model’s official training history. Token shares apply to this job only; they are not paper-prescribed ratios.','工程起点，不是模型官方训练历史。数据占比只针对本次任务，并非论文规定比例。')),
    el('p',`${tr('Config context limit','配置声明的上下文上限')}: ${c.declared_context_limit?.toLocaleString()||'—'} · ${tr('This job target','本次目标')}: ${c.effective_target.toLocaleString()} · ${c.published_reference?tr('Published stages: see reference below','公开阶段长度：见下方参考'):tr('Official training length: unknown','官方训练长度：未确认')}`));
  if(c.published_reference){const ref=c.published_reference;box.append(el('p',tr('Published release reference (not this job): ','公开版本参考（不是本次任务）：')+ref.lengths_label+' · '+ref.locator),el('p',language==='zh'?ref.note_zh:ref.note_en));}
  if(!c.stages.length){box.append(el('p',tr('Preference/RL training needs a separate role plan: prompt + response, chosen/rejected, group size and rollout KV. No single-sequence GPU recommendation is implied.','偏好／强化学习需要单独的角色规划：提示词＋回答、正负样本、组大小和 rollout KV；不能套用单序列推荐。')));return;}
  const details=el('div');
  table(box,[tr('Stage','阶段'),tr('Sequence tokens','序列 token'),tr('Token budget share','token 预算占比'),tr('Effective tokens','有效 token'),tr('Resource plan','资源方案')],c.stages.map(s=>{
    const b=el('button',tr('View recalculated GPUs','查看重新计算的 GPU 方案'));b.type='button';b.onclick=()=>{details.replaceChildren();details.append(el('h3',(language==='zh'?s.name_zh:s.name_en)+' · '+s.seq),el('p',language==='zh'?s.advance_gate_zh:s.advance_gate_en));trainingDesignView(details,{...r,training_design:s.resource_design,assumptions:{...r.assumptions,seq:s.seq}});};
    return [language==='zh'?s.name_zh:s.name_en,s.seq.toLocaleString(),fmt(s.token_share_percent)+'%',s.tokens==null?'—':s.tokens.toLocaleString(),b];}));
  box.append(el('p',tr('Validate real data lengths first. Long stages retain short examples; do not pad every sample to the cap. Hybrid-model state-only candidates may match across lengths: their activation peaks remain unknown, not constant.','先验证真实数据长度分布；长阶段保留短样本，不把所有样本填充到上限。混合模型的状态容量候选可能各阶段相同：这表示激活峰值未知，不是它不随长度增长。')));
  const evidence=el('details');evidence.append(el('summary',tr('Rules and references','规则与参考依据')));box.append(evidence);for(const source of c.references){const a=el('a',source.title);a.href=source.url;a.target='_blank';a.rel='noopener noreferrer';evidence.append(el('p'),a);}raw(evidence,{basis:c.basis,warnings:c.warnings});box.append(details);
}
function trainingDesignView(parent,r){
  const tr=(en,zh)=>language==='zh'?zh:en,design=r.training_design,id=r.model_identity||{};
  const component=k=>k==='saved_activations'?tr('Saved activations','保存的激活'):k==='dispatch'?tr('Expert dispatch buffers','专家分发缓冲'):t(k);
  const box=el('section',undefined,'plan-basis');parent.append(box);
  box.append(el('h3',tr('Joint training planner','训练联合规划器')+' · '+(id.official_name||id.display_name||r.parameters/1e9+' B')),
    el('p',tr('Analytical candidates, not validated launch recipes. No rental or training is started.','解析估算候选，尚非已验证启动配方。不会租机或启动训练。')));
  box.append(el('p',tr('Budget parameter count: ','账本参数计数：')+r.parameters.toLocaleString()+' · '+r.parameter_basis+(id.official?' · '+tr('Official nominal / active: ','官方标称／激活：')+id.official.total_parameters/1e9+' B / '+id.official.active_parameters/1e9+' B':'')));
  decisionTraceView(box,design);
  if(design.blockers.length){for(const text of design.blockers)box.append(el('p',text,'danger'));}
  if(design.structural_capacity){structuralCapacityView(box,r,design);return;}
  const rec=design.recommendations,choices=el('div',undefined,'actions'),detail=el('div');box.append(choices,detail);
  for(const [key,label]of [['minimum_capacity',tr('Minimum capacity','最低容量')],['lowest_estimated_cost',tr('Lowest estimated cost','预计最低成本')],['deadline',tr('Meet deadline','满足期限')]]){
    const b=el('button',label);b.type='button';b.disabled=!rec[key];b.onclick=()=>show(rec[key]);choices.append(b);
  }
  box.append(el('p',tr('Cost/time require total tokens, effective TFLOPS, relevant link bandwidth and price. Blank results mean unknown or no qualifying candidate, not zero.','时间／成本需要总 token、有效算力、相关链路带宽与单价。空缺表示未知或没有满足条件的候选，不是零。')));
  function show(c){
    detail.replaceChildren();
    detail.append(el('h3',`${c.gpus} × ${c.gpu} · ${c.nodes} ${tr('nodes','节点')}`),el('p',`TP ${c.tp} / PP ${c.pp} / DP ${c.dp} / CP ${c.cp} / EP ${c.ep} · ${c.backend_family} · ${c.state_strategy}`,'placement-equation'));
    detail.append(el('p',`${tr('Microbatch / accumulation','微批／累积')} ${c.microbatch} / ${c.gradient_accumulation} · ${tr('Effective tokens/step','有效 token/步')} ${c.effective_batch_tokens.toLocaleString()} · ${tr('Recompute','重计算')} ${c.checkpoint} · SP=${c.sequence_parallel}`));
    const range=x=>x?x.map(v=>v.toFixed(2)).join(' – '):'—';
    table(detail,[tr('Peak/card GiB','每卡峰值 GiB'),tr('Steps','步数'),tr('Hours','小时'),tr('Cost (input currency)','费用（输入币种）'),tr('Bottleneck signals','瓶颈信号')],[[fmt(c.peak_gib),c.steps??'—',range(c.hours_range),range(c.cost_range),c.bottlenecks.join(', ')]]);
    detail.append(el('p',tr('Ranges are overlap sensitivity scenarios, not measured confidence intervals. Memory uses the worst-stage budget on every card.','区间是通信重叠敏感性情景，不是实测置信区间。所有卡保守展示最重阶段预算。')));
    const nodes=el('div',undefined,'placement-nodes'),nav=el('div',undefined,'actions'),ledger=el('div',undefined,'placement-detail');detail.append(nav,nodes,ledger);
    let page=0;const per=Number(r.assumptions.gpus_per_node||8);
    function pick(rank){ledger.replaceChildren();const tp=rank%c.tp,cp=Math.floor(rank/c.tp)%c.cp,dp=Math.floor(rank/(c.tp*c.cp))%c.dp,pp=Math.floor(rank/(c.tp*c.cp*c.dp));
      ledger.append(el('h4',`Rank ${rank}: TP${tp} / CP${cp} / DP${dp} / PP${pp} / EP${dp%c.ep}`),el('p',`Layers [${pp*r.dims.layers/c.pp}, ${(pp+1)*r.dims.layers/c.pp}) · ${tr('Local context tokens','本卡上下文 token')} ${design.assumptions.seq/c.cp}`));
      ledger.append(el('p',tr('EP is nested in DP; optimizer expert-DP = DP/EP. State-shard divisors and all precision assumptions are expanded below.','EP 嵌套于 DP；专家优化器分片组为 DP/EP。具体分片除数和精度见下方公式。')));
      table(ledger,['component',tr('Precision / bytes per element','精度／每元素字节'),tr('Elements/card (estimate)','每卡元素数（估算）'),tr('Bytes/card','每卡字节'),'rankGib',tr('Substituted formula','代入公式')],c.rows.map(x=>[component(x.item),x.precision+(x.bytes_per_element?' · '+x.bytes_per_element+' B':''),x.elements==null?tr('Mixed/budget; not a tensor count','混合／预算，不伪造元素数'):x.elements.toLocaleString(undefined,{maximumFractionDigits:3}),x.bytes.toLocaleString(undefined,{maximumFractionDigits:3}),fmt(x.per_gpu_gib),el('code',x.calculation,'memory-formula')]));
      ledger.append(el('p',`${r.selected.vram_gib} − ${fmt(c.peak_gib)} − ${fmt(r.selected.vram_gib*r.reserve_percent/100)} = ${fmt(c.headroom_gib)} GiB`));
    }
    function draw(){nodes.replaceChildren();nav.replaceChildren();const previous=el('button',tr('Previous nodes','上一组节点')),next=el('button',tr('Next nodes','下一组节点'));previous.disabled=page===0;next.disabled=(page+1)*4>=c.nodes;previous.onclick=()=>{page--;draw();};next.onclick=()=>{page++;draw();};nav.append(previous,el('span',`${page+1}/${Math.ceil(c.nodes/4)}`),next);
      for(let n=page*4;n<Math.min(c.nodes,page*4+4);n++){const node=el('div',undefined,'placement-node'),grid=el('div',undefined,'placement-gpus');node.append(el('h4','Node '+n),grid);nodes.append(node);
        for(let rank=n*per;rank<Math.min(c.gpus,(n+1)*per);rank++){const card=el('button',undefined,'gpu-tile'),bar=el('div',undefined,'gpu-memory');card.type='button';card.append(el('strong',`GPU ${rank%per} · R${rank}`),el('span',`${fmt(c.peak_gib)} / ${r.selected.vram_gib} GiB`),bar);for(const x of c.rows){const seg=el('span');seg.style.width=(x.per_gpu_gib/r.selected.vram_gib*100)+'%';seg.style.background=({weights:'#3976de',gradients:'#9b65cb',master:'#b285d5',adam_m:'#ce86af',adam_v:'#ab6b9b',saved_activations:'#e49a30',transient:'#c98123',logits:'#ecba59',dispatch:'#df6f63'})[x.item]||'#647891';seg.title=component(x.item)+' '+fmt(x.per_gpu_gib)+' GiB';bar.append(seg);}const reserve=el('span');reserve.style.width=r.reserve_percent+'%';reserve.style.background='#cbd5e1';reserve.title=tr('Safety reserve','安全预留');bar.append(reserve);card.onclick=()=>pick(rank);grid.append(card);}
      }
    }draw();pick(0);
    const comm=el('details');comm.append(el('summary',tr('Communication, timing and validation','通信、时间与验证依据')));detail.append(comm);
    table(comm,[tr('Communication','通信'),'GiB / rank / step'],Object.entries(c.comm_gib_per_rank_step).map(([k,v])=>[k,fmt(v)]));
    raw(comm,{compute_seconds:c.compute_seconds,communication_seconds_proxy:c.communication_seconds_proxy,hbm_seconds_lower_bound:c.hbm_seconds_lower_bound,pipeline_bubble_ratio:c.pipeline_bubble_ratio,step_seconds_range:c.step_seconds_range,assumptions:design.assumptions,validation:design.validation});
  }
  if(rec.minimum_capacity)show(rec.minimum_capacity);
  const evidence=el('details');evidence.append(el('summary',tr('Search bounds, exclusions and source rules','搜索边界、淘汰原因与规则来源')));box.append(evidence);
  for(const url of design.sources){const a=el('a',url);a.href=url;a.target='_blank';a.rel='noopener noreferrer';evidence.append(el('p'),a);}
  raw(evidence,design);
}
function structuralCapacityView(box,r,design){
  const tr=(en,zh)=>language==='zh'?zh:en,i=design.architecture_inventory,s=design.structural_capacity;
  box.append(el('h3',tr('Architecture-derived state capacity · activation peak pending','结构驱动容量方案 · 激活峰值待验证')),
    el('p',`${i.layers.length} ${tr('decoder layers','解码层')} · ${i.expert_count} experts / top-${i.expert_topk} · ${Object.entries(i.attention_layers).map(([k,v])=>k+' × '+v).join(' / ')}`),
    el('p',tr('All headers scanned; auxiliary/MTP and vision weights retained. These are storage elements, conservatively treated as trainable.','已扫描全部张量头；保留视觉和 MTP 等辅助权重。按存储元素保守计作可训练参数。')));
  table(box,[tr('Component','组成'),tr('B elements','十亿元素')],[[tr('Decoder routed experts','解码器路由专家'),fmt(i.routed_elements/1e9)],[tr('Other modules (incl. auxiliary experts)','其余模块（含辅助专家）'),fmt(i.non_routed_elements/1e9)],...Object.entries(i.outside_decoder).map(([k,v])=>[k,fmt(v/1e9)])]);
  box.append(el('p',tr('Conditional state-capacity only, not a runnable recommendation. TP=CP=1; no unverified hybrid attention partitioning. Time/cost await kernel and communication measurements.','这里只给有条件的状态容量方案，不是已验证可运行推荐。TP=CP=1，不假定混合注意力支持切分。耗时／成本需补充算子与通信实测。')));
  const select=el('select'),detail=el('div');box.append(select,detail);
  s.candidates.forEach((c,j)=>{const o=el('option',`${c.gpus} GPU · PP${c.pp}/DP${c.dp}/EP${c.ep} · ZeRO-${c.zero_stage} · ${fmt(c.accounted_peak_gib)} GiB`);o.value=j;select.append(o);});
  function show(){detail.replaceChildren();const c=s.candidates[Number(select.value)];if(!c){detail.append(el('p',tr('No state-capacity fit inside the search ceiling. Increase the ceiling or change hardware.','搜索上限内没有状态容量满足的候选；请调整上限或硬件。')));return;}
    detail.append(el('h4',`TP1 / PP${c.pp} / DP${c.dp} / CP1 / EP${c.ep} · ZeRO-${c.zero_stage}`),el('p',tr('EP is nested in DP. Non-decoder modules are conservatively placed on stage 0. Blank capacity is the budget for unknown activations/buffers, not guaranteed free VRAM.','EP 嵌套于 DP。非解码模块保守放在第 0 阶段。空余部分是未知激活／缓冲的预算，不是保证可用的显存。')));
    const grid=el('div',undefined,'placement-gpus'),ledger=el('div'),pages=el('select');detail.append(pages,grid,ledger);
    const per=Number(r.assumptions.gpus_per_node||8);for(let n=0;n<c.nodes;n++){const o=el('option','Node '+n);o.value=n;pages.append(o);}
    const keys=['weights','gradients','master','adam_m','adam_v','gather','workspace'];
    function pick(x){ledger.replaceChildren();ledger.append(el('h4',`Rank ${x.rank} · PP${x.pp_rank} / DP${x.dp_rank} / EP${x.ep_rank}`),el('p',`Layers [${x.layer_start}, ${x.layer_end}) · experts: ${x.expert_ids}`),el('p',x.precision),el('code',x.formula,'memory-formula'));table(ledger,[tr('Component','分项'),tr('Precision / B per element','精度／每元素字节'),tr('Elements/card (estimate)','每卡元素数（估算）'),tr('Bytes/card','每卡字节'),'GiB',tr('Substituted formula','代入公式')],(x.ledger||[]).map(row=>[t(row.item),row.precision+(row.bytes_per_element?' · '+row.bytes_per_element+' B':''),row.elements==null?tr('Budget, not tensor count','预算，非张量元素数'):row.elements.toLocaleString(undefined,{maximumFractionDigits:3}),row.bytes.toLocaleString(undefined,{maximumFractionDigits:3}),fmt(row.gib),el('code',row.formula,'memory-formula')]));ledger.append(el('p',tr('Balanced shard estimates exclude padding/allocator overhead. Activations and unmodeled communication remain unknown, not zero.','均匀分片估算不含 padding／分配器开销。激活与未建模通信仍为未知，不是零。')));ledger.append(el('p',`${r.selected.vram_gib} − ${fmt(x.accounted_gib)} − ${fmt(r.selected.vram_gib*r.reserve_percent/100)} = ${fmt(r.selected.vram_gib*(1-r.reserve_percent/100)-x.accounted_gib)} GiB · ${tr('unmeasured activation budget','待实测激活预算')}`));}
    function draw(){grid.replaceChildren();for(const x of c.ranks.slice(Number(pages.value)*per,(Number(pages.value)+1)*per)){const b=el('button',undefined,'gpu-tile'),bar=el('div',undefined,'gpu-memory');b.append(el('strong',`GPU ${x.rank%per} · R${x.rank}`),el('span',`${fmt(x.accounted_gib)} GiB · PP${x.pp_rank} / EP${x.ep_rank}`),bar);keys.forEach((k,j)=>{const seg=el('span');seg.style.width=x[k+'_gib']/r.selected.vram_gib*100+'%';seg.style.background=['#3976de','#9b65cb','#b285d5','#ce86af','#ab6b9b','#e49a30','#647891'][j];seg.title=t(k)+' '+fmt(x[k+'_gib'])+' GiB';bar.append(seg);});b.onclick=()=>pick(x);grid.append(b);}}
    pages.onchange=draw;draw();pick(c.ranks[0]);
  }select.onchange=show;show();const evidence=el('details');evidence.append(el('summary',tr('Inventory, assumptions and validation','张量清单、假设与验证要求')));box.append(evidence);raw(evidence,design);
}
function resourcePlanBasis(parent,r){
  const tr=(en,zh)=>language==='zh'?zh:en,s=r.selected,top=s.topology,rec=r.zero_recommendation;
  const box=el('section',undefined,'plan-basis');parent.append(box);
  const identity=r.model_identity||{},official=identity.official;
  box.append(el('h3',identity.official_name||identity.display_name||tr('Model identity unavailable','模型身份待确认')));
  box.append(el('p',(identity.repo?identity.repo+' · ':'')+tr('Scanned/reported elements: ','扫描／报告元素数：')+(identity.stored_parameter_count==null?'—':identity.stored_parameter_count.toLocaleString()+' ('+(identity.stored_parameter_count/1e9).toFixed(3)+' B)')));
  if(official)box.append(el('p',tr('Official nominal parameters: ','官方标称参数：')+official.total_parameters/1e9+' B · '+tr('active per token: ','每 token 激活：')+official.active_parameters/1e9+' B'));
  box.append(el('small',tr('Metadata match, not checkpoint authenticity verification. Stored tensor elements may include buffers or shared-weight differences; not an exact trainable parameter count.','元数据匹配不等于权重真实性验证。存储元素可能包含缓冲区、共享权重差异，不等于严格可训练参数数。')));
  const reference=el('details');reference.append(el('summary',tr('Official training setup vs this estimate','官方训练配置与本方案对照')));box.append(reference);
  const unknown=official?tr('Not found in checked sources','已核查来源未披露'):tr('No verified source yet','尚无已核查来源'),training=official?.training||{};
  table(reference,[tr('Item','项目'),tr('Official release','官方版本'),tr('This estimate','当前估算')],[
    [tr('Stage / objective','阶段／目标'),training.phase||unknown,r.task+' / '+r.mode],
    [tr('Training tokens','训练 token 总量'),training.tokens?.toLocaleString()||unknown,tr('Not budgeted: capacity only','未核算：本页仅算容量')],
    ['GPU',training.gpu||unknown,s.name],
    [tr('GPU count','GPU 数'),training.gpu_count??unknown,s.allocated_gpus],
    [tr('Parallelism','并行策略'),training.parallelism||unknown,top?`TP=${top.tp}, PP=${top.pp}, DP=${top.dp}, CP=1`:'—'],
    ['ZeRO',training.zero??unknown,r.zero_stage??'N/A'],
    [tr('Training precision','训练精度'),training.precision||unknown,r.task==='train'?(r.mode==='qlora'?'4-bit base / BF16 adapters / FP32 optimizer':'BF16 + FP32 optimizer'):tr('Inference only','仅推理')],
    [tr('Batch / sequence','批量／序列长度'),training.batch&&training.sequence_length?training.batch+' / '+training.sequence_length:unknown,`${r.assumptions.batch||1} / ${r.assumptions.seq||4096}`]
  ]);
  reference.append(el('p',tr('No similarity score: full pretraining clusters and minimum-capacity experiments have different goals. Unknown hardware, precision or global batch prevents a meaningful closeness claim. BF16 release weights do not prove BF16 training.','不计算相似度：完整预训练集群与最低容量实验目标不同；硬件、精度或全局批量缺失时，不能声称方案接近。BF16 发布权重不代表训练全程使用 BF16。')));
  if(official)for(const source of official.sources){const link=el('a',source.title+' · '+source.checked);link.href=source.url;link.target='_blank';link.rel='noopener noreferrer';reference.append(link);}
  else reference.append(el('p',tr('No verified release-specific training reference in the current catalog. Do not infer it from the directory name.','当前参考库尚无已核查的该版本训练资料；不会从目录名猜测。')));
  if(official?.scope_note)reference.append(el('p',official.scope_note));
  if(r.planning_scope?.limited)box.append(el('p',tr('Limited capacity candidate, NOT a production recommendation: training TP/PP/CP/EP and architecture-specific activation memory are not searched. Do not rent this GPU count without validating those alternatives.','受限容量候选，不是生产训练推荐：未搜索训练 TP／PP／CP／EP，也未准确建模专用架构激活。请勿直接按此卡数租机，需先验证这些替代方案。'),'danger'));
  box.append(el('h3',s.feasible_under_assumptions?
    (r.planning_scope?.limited?tr('Limited capacity candidate','受限容量候选'):tr('Recommended plan','推荐方案'))+' · '+s.allocated_gpus+' × '+s.name:
    tr('No feasible plan under these assumptions','当前假设下暂无可行方案')));
  box.append(el('p',(r.task==='train'?'ZeRO-'+(rec?.stage??r.zero_stage):tr('Inference','推理'))+
    (top?` · TP ${top.tp} / PP ${top.pp} / DP ${top.dp} / CP 1`:'')+
    ` · B=${r.assumptions.batch||1} · S=${r.assumptions.seq||4096}`,'placement-equation'));
  box.append(el('p',r.task==='train'?
    (r.mode==='qlora'?tr('4-bit frozen base; BF16 adapters/gradients; FP32 master + Adam','4-bit 冻结基座；BF16 适配器／梯度；FP32 主权重＋Adam'):
    tr('BF16 weights/gradients · 2 bytes; FP32 master/Adam · 4 bytes','BF16 权重／梯度 · 每元素 2 字节；FP32 主权重／Adam · 每元素 4 字节')):
    `${r.assumptions.bits||16}-bit `+tr('weights; BF16 KV · 2 bytes/element','权重；BF16 KV · 每元素 2 字节')));
  box.append(el('p',tr('Estimate, not measurement. Fewest cards, then least sharding; not a proven speed optimum. Select a GPU below to inspect the arithmetic.','估算，非实测。优先更少卡数、较低分片阶段，不代表速度最优。点击下方 GPU 查看逐项计算。')));
  const more=el('details');more.append(el('summary',tr('Why this plan? Alternatives and full assumptions','为什么推荐？备选方案与完整假设')));box.append(more);
  if(rec)table(more,['zeroStage','minimum','allocated','peak','headroom','status'],rec.alternatives.map(x=>[
    'ZeRO-'+x.stage,x.minimum_gpus??'—',x.allocated_gpus,fmt(x.peak_gib),fmt(x.headroom_gib),
    x.selected?tr('Recommended','推荐'):x.feasible?tr('Feasible alternative','可行备选'):tr('Not feasible / unsupported','不可行／不支持')]));
  table(more,['fieldName','value'],[
    [tr('Model / source','模型／参数来源'),(r.assumptions.model_path||'—')+' · '+r.parameter_basis],
    [tr('Parameters','参数量'),r.parameters.toLocaleString()],
    [tr('Task / tuning','任务／训练方式'),r.task+' / '+r.mode],
    [tr('Structure','结构维度'),Object.entries(r.dims).map(([k,v])=>k+'='+(v||'?')).join(' · ')],
    [tr('Reserve','安全预留'),r.reserve_percent+'%'],
    [tr('Checkpointing','激活重计算'),r.activation_basis],
    ['Offload / EP / CP',tr('Not modeled; no automatic launch configuration','未纳入模型；不会自动修改启动配置')],
    [tr('Network','网络'),s.intra_node+' · '+s.inter_node]
  ]);
  more.append(txt('zeroAutomatic'));
  for(const note of r.warnings)more.append(el('p',note));
}
// Resource charts consume the same per-rank ledger as the numeric report.
// No animation is used: these are capacity estimates, not live GPU telemetry.
function resourcePlacement(parent,r){
  const tr=(en,zh)=>language==='zh'?zh:en;
  const map=r.placement,s=r.selected,b=r.detailed_budget;
  const box=el('section',undefined,'placement');parent.append(box);
  box.append(el('h3',tr('Cluster → node → GPU allocation','集群 → 节点 → GPU 分配')));
  box.append(el('p',tr('Estimated logical placement · not a measured topology or an executable launch plan. Equal layer count does not imply equal memory or compute.','逻辑分配估算 · 不是实测硬件拓扑，也不是可直接执行的启动计划。层数相同不代表显存或计算量相同。')));
  if(!map?.ranks?.length){box.append(el('p',s.reason||tr('No valid placement.','暂无有效分配。')));return;}
  if(!s.feasible_under_assumptions)box.append(el('p',tr('OVER BUDGET: this allocation is not feasible under the entered assumptions.','超出预算：按当前假设，此分配不可行。'),'danger'));
  const a=map.axes;
box.append(el('p',`TP ${a.tp} × PP ${a.pp} × DP ${a.dp} × CP ${a.cp} = ${map.ranks.length} GPU · ${r.zero_stage==null?tr("Inference","推理"):"ZeRO-"+r.zero_stage}`, 'placement-equation'));
  const conceptFold=el('details');conceptFold.append(el('summary',tr('How parallelism partitions work','并行方式如何切分工作')));box.append(conceptFold);
  const concepts=el('div',undefined,'parallel-concepts');conceptFold.append(concepts);
  for(const [title,body]of [
    [`TP ×${a.tp}`,tr('Split matrix operations within a layer. Ranks work on the same sequence; partial outputs require collectives. KV heads may be replicated, not evenly split.','切同一层的矩阵计算。各卡处理同一序列，局部结果需要集合通信；KV 头可能复制，不能一律平均切分。')],
    [`PP ×${a.pp}`,tr('Split consecutive layers into stages. Activations flow forward and gradients backward during training. Microbatch scheduling affects pipeline bubbles.','按连续层切成阶段。前向传激活，训练反向传梯度；微批调度决定流水线空泡。')],
    [`DP ×${a.dp}`,tr('Different training samples per rank. State sharding follows the selected ZeRO stage above; DP alone does not imply parameter sharding.','训练每卡处理不同样本。状态是否分片由上方 ZeRO 阶段决定；DP 本身不代表权重已经分片。')],
    ['CP ×1',tr('Not enabled in this plan. Context parallelism splits a sequence, exchanges attention context, and does not automatically shard weights or optimizer state.','当前方案未启用。上下文并行切分序列，并交换注意力上下文；不会自动分片权重或优化器状态。')]
  ]){const c=el('div');c.append(el('strong',title),el('p',body));concepts.append(c);}
  if(r.task==='train')box.append(el('p',tr('Global microbatch before gradient accumulation: ','梯度累积前的全局微批：')+`${map.global_microbatch} = ${r.assumptions.batch||1} × DP ${a.dp}`));
  const colors={weights:'#3976de',frozen:'#3976de',quant_meta:'#6e91d5',gradients:'#9b65cb',master:'#b285d5',adam_m:'#ce86af',adam_v:'#ab6b9b',kv:'#10a896',saved_activations:'#e49a30',transient:'#c98123',logits:'#ecba59',activation_extra:'#a97939',gather:'#df6f63',workspace:'#647891',runtime_gib:'#647891',communication_gib:'#df6f63',cuda_graph_gib:'#647891',other_models_gib:'#647891',rollout_gib:'#647891'};
  const active=b.rows.filter(x=>x.per_gpu_gib>0);
  function cardLedger(){
    const list=el('dl',undefined,'gpu-ledger');
    for(const [name,keys]of [
      [tr('Weights + metadata','权重＋元数据'),['weights','frozen','quant_meta']],
      [tr('Gradients','梯度'),['gradients']],
      [tr('Master + optimizer','主权重＋优化器'),['master','adam_m','adam_v']],
      ['KV',['kv']],
      [tr('Activations + logits','激活＋logits'),['saved_activations','transient','logits','activation_extra']],
      [tr('Gather + workspace + other','聚合＋工作区＋其他'),['gather','workspace','runtime_gib','communication_gib','cuda_graph_gib','other_models_gib','rollout_gib']]
    ]){list.append(el('dt',name),el('dd',fmt(b.rows.filter(x=>keys.includes(x.item)).reduce((sum,x)=>sum+(x.per_gpu_gib||0),0))+' GiB'));}
    list.append(el('dt',tr('Safety reserve','安全预留')),el('dd',fmt(b.reserve_gib)+' GiB'),el('dt',tr('Headroom','剩余预算')),el('dd',fmt(b.headroom_gib)+' GiB'));return list;
  }
  const legend=el('div',undefined,'placement-legend');box.append(legend);
  for(const [name,color]of [[tr('Weights','权重'),'#3976de'],[tr('Gradients / optimizer','梯度／优化器'),'#9b65cb'],['KV','#10a896'],[tr('Activations','激活'),'#e49a30'],[tr('Gather / communication','聚合／通信'),'#df6f63'],[tr('Other budgets','其他预算'),'#647891'],[tr('Reserve','安全预留'),'#cbd5e1']]){const item=el('span',name),dot=el('i');dot.style.background=color;item.prepend(dot);legend.append(item);}
  const nav=el('div',undefined,'actions'),nodes=el('div',undefined,'placement-nodes'),detail=el('div',undefined,'placement-detail');
  detail.setAttribute('aria-live','polite');box.append(nav,nodes,detail);
  let page=0,selected=0;const pageSize=4, pages=Math.ceil(s.nodes/pageSize);
  function meter(rank){const bar=el('div',undefined,'gpu-memory');bar.setAttribute('role','img');bar.setAttribute('aria-label',`${rank.peak_gib.toFixed(2)} / ${rank.capacity_gib} GiB`);
    for(const row of [...active,{item:'reserve',per_gpu_gib:rank.reserve_gib}]){const seg=el('span');seg.style.width=(row.per_gpu_gib/rank.capacity_gib*100)+'%';seg.style.background=colors[row.item]||'#cbd5e1';seg.title=`${t(row.item)}: ${row.per_gpu_gib.toFixed(3)} GiB`;bar.append(seg);}return bar;}
  function select(rank){selected=rank.rank;detail.replaceChildren();
    detail.append(el('h4',`Node ${rank.node} / GPU ${rank.local_gpu} · Rank ${rank.rank}`));
    detail.append(el('p',`TP ${rank.tp_rank} · PP ${rank.pp_rank} · DP ${rank.dp_rank} · CP 0`));
detail.append(el('p',tr('Layer interval (zero-based): ','层区间（从 0 计）：')+(rank.layer_start==null?tr('unknown','未知'):`[${rank.layer_start}, ${rank.layer_end_exclusive})`)+tr(' · resident weight fraction: ',' · 驻留权重比例：')+`1/${Math.round(1/rank.weight_fraction)}`));
    detail.append(el('p',tr('Sequences per rank × tokens per sequence: ','每卡序列数 × 每条序列 token 数：')+`${rank.sequences} × ${rank.tokens_per_sequence}`+tr(' · Q / KV heads per TP rank: ',' · 每个 TP rank 的 Q / KV 头：')+`${rank.query_heads??'?'} / ${rank.kv_heads??'?'}`));
    const mapping=el('details');mapping.append(el('summary',tr('Communication groups and mapping caveats','通信组与映射限制')));detail.append(mapping);
    mapping.append(el('p',tr('Standard attention proxy; MLA, hybrid attention and expert dispatch need profiling. Memory share does not measure compute time.','标准注意力近似；MLA、混合注意力和专家路由需实测。显存占比不是计算耗时。')));
    const groups=map.groups.filter(g=>g.ranks.includes(rank.rank));
    for(const g of groups){const members=g.ranks.length>32?g.ranks.slice(0,32).join(', ')+' … ('+g.ranks.length+')':g.ranks.join(', ');mapping.append(el('p',`${g.axis.toUpperCase()} ranks: ${members}`));}
    detail.append(el('h4',tr('Memory arithmetic · click a component to expand','显存怎么算 · 展开分项查看代入过程')));
    detail.append(el('p',tr('Precision below is the planner assumption, not verified tensor dtype. 1 GiB = 1,073,741,824 bytes. Zero budgets are hidden below, not measured zero.','下列精度是估算假设，不是已核验的张量 dtype。1 GiB = 1,073,741,824 字节。零预算项折叠在下方，不代表实测为零。')));
    const meaning={weights:['Trainable weights (training) or resident model weights (inference).','训练时为可训练权重，推理时为驻留模型权重。'],gradients:['One gradient per trainable parameter.','每个可训练参数对应一个梯度。'],master:['FP32 master copy used by the assumed mixed-precision optimizer.','假设混合精度优化器保留的 FP32 主权重副本。'],adam_m:['Adam first moment, one FP32 value per trainable parameter.','Adam 一阶矩：每个可训练参数一个 FP32 值。'],adam_v:['Adam second moment, one FP32 value per trainable parameter.','Adam 二阶矩：每个可训练参数一个 FP32 值。'],kv:['K and V × bytes × sequences × tokens × layers × KV heads × head dimension; then apply PP/head sharding.','K 和 V 两份 × 字节数 × 序列数 × token 数 × 层数 × KV 头数 × 头维度，再按 PP 与 KV 头分片；MLA 使用不同压缩公式。'],saved_activations:['BF16 checkpoint boundaries: 2 × B × S × hidden × layers.','BF16 重计算边界：2 字节 × B × S × 隐藏维度 × 层数。'],transient:['Coefficient 12 is a temporary-tensor proxy, NOT a 12-byte dtype.','系数 12 是临时张量合计近似，不是某种 12 字节数据精度。'],logits:['FP32 materialized logits: 4 × B × S × vocabulary.','FP32 物化 logits：4 字节 × B × S × 词表大小。'],activation_extra:['Top up structural activation estimate to the input budget; do not add the full budget twice.','将结构激活估算补足到输入预算，不把整笔预算重复相加。']};
    function explainRow(x){const fold=el('details');fold.append(el('summary',t(x.item)));if(x.item==='weights')fold.open=true;
      if(meaning[x.item])fold.append(el('p',tr(...meaning[x.item])));
      if(x.source==='parameter_formula'&&r.task==='train'){
        const shard=(['master','adam_m','adam_v'].includes(x.item)&&r.zero_stage>=1)||(x.item==='gradients'&&r.zero_stage>=2)||(['weights','frozen'].includes(x.item)&&r.zero_stage===3);
        fold.append(el('p',`ZeRO-${r.zero_stage}: `+(shard?tr(`divide by DP=${a.dp} GPUs; this state is sharded.`,`除以 DP=${a.dp} 张卡；此项状态参与分片。`):tr('divide by 1; every GPU retains this complete state.','除以 1；每卡保留完整的这项状态。'))));
      }
      fold.append(el('code',x.calculation||x.basis,'memory-formula'));
      fold.append(el('p',({parameter_formula:tr('Parameter formula; sharding divisor follows ZeRO/TP/PP.','参数公式；除数由 ZeRO／TP／PP 决定。'),shape_proxy:tr('Structural approximation, not measured tensor liveness.','结构近似，不是实测张量生命周期。'),format_proxy:tr('Format-dependent approximation; inspect quantization metadata.','格式相关近似，须核对量化元数据。'),manual_budget:tr('Input/default budget, not inferred from model precision.','用户输入或默认预算，不是由模型精度推导。')})[x.source]||x.basis));
      return [fold,x.precision||'—',fmt(x.per_gpu_gib)];}
    table(detail,['component',tr('Precision / bytes','精度／字节数'),'rankGib'],active.map(explainRow));
    const zeroRows=b.rows.filter(x=>!x.per_gpu_gib),zeroFold=el('details');zeroFold.append(el('summary',tr('Zero / unknown budget items','零值／未知预算项')));table(zeroFold,['component',tr('Precision / bytes','精度／字节数'),'rankGib'],zeroRows.map(explainRow));detail.append(zeroFold);
    detail.append(el('p',`${tr('Capacity','容量')} ${rank.capacity_gib} − ${tr('estimated peak','估算峰值')} ${fmt(rank.peak_gib)} − ${tr('reserve','预留')} ${fmt(rank.reserve_gib)} = ${tr('headroom','剩余预算')} ${fmt(rank.headroom_gib)} GiB`));
    for(const btn of nodes.querySelectorAll('button'))btn.setAttribute('aria-pressed',String(Number(btn.dataset.rank)===selected));
  }
  function draw(){nav.replaceChildren();nodes.replaceChildren();
    const prev=el('button',tr('Previous nodes','上一组节点')),next=el('button',tr('Next nodes','下一组节点'));prev.type=next.type='button';prev.disabled=page===0;next.disabled=page>=pages-1;prev.onclick=()=>{page--;draw();};next.onclick=()=>{page++;draw();};nav.append(prev,el('span',`${page+1} / ${pages} · ${s.nodes} ${tr('nodes','节点')}`),next);
    for(let node=page*pageSize;node<Math.min(s.nodes,(page+1)*pageSize);node++){const panel=el('div',undefined,'placement-node');panel.append(el('h4',`Node ${node} · ${s.name}`));const grid=el('div',undefined,'placement-gpus');panel.append(grid);nodes.append(panel);
for(const rank of map.ranks.filter(x=>x.node===node)){const btn=el('button',undefined,'gpu-tile');btn.type='button';btn.dataset.rank=rank.rank;btn.setAttribute('aria-pressed',String(selected===rank.rank));btn.append(el('strong',`GPU ${rank.local_gpu} · R${rank.rank}`),el('span',`TP${rank.tp_rank} / PP${rank.pp_rank} / DP${rank.dp_rank}`),meter(rank),el('span',`${fmt(rank.peak_gib)} / ${rank.capacity_gib} GiB`));btn.append(cardLedger());btn.onclick=()=>select(rank);grid.append(btn);}
    }
  }draw();select(map.ranks[0]);
  const arithmetic=el('details');arithmetic.append(el('summary',tr('Selected GPU · formulas and mapping','选中 GPU · 显存公式与映射')),detail);box.append(arithmetic);
  const flow=el('details');flow.append(el('summary',tr('Tensor and pipeline data flow','张量与流水线数据流')));box.append(flow);
  flow.append(el('p',tr('Schematic, not a timeline or a throughput measurement. TP is a partition within a stage; PP passes intermediate activations between stages.','这是结构示意，不是时间线或吞吐实测。TP 在阶段内部切分计算；PP 在阶段之间传递中间激活。')));
  const pipeline=el('div',undefined,'parallel-flow');flow.append(pipeline);
  for(let stage=0;stage<Math.min(a.pp,16);stage++){
    if(stage)pipeline.append(el('span',tr('→ activations →','→ 激活 →')));
    const block=el('div',undefined,'parallel-stage');block.append(el('strong',`PP ${stage}`));
    const first=map.ranks[stage*a.tp];block.append(el('p',first.layer_start==null?tr('Layers unknown','层数未知'):`Layers [${first.layer_start}, ${first.layer_end_exclusive})`));
    const shards=el('div',undefined,'matrix-shards');block.append(shards);
    for(let lane=0;lane<a.tp;lane++)shards.append(el('span',`W${lane} · R${stage*a.tp+lane}`));
    block.append(el('p',a.tp>1?tr('Matrix shards → local GEMM → collective','矩阵分片 → 本地 GEMM → 集合通信'):tr('Full layer computation','完整层计算')));pipeline.append(block);
  }
  if(a.pp>16)flow.append(el('p',tr('First 16 stages shown; select individual GPUs above for the complete mapping.','仅展示前16个阶段；完整映射可在上方逐卡查看。')));
  if(r.task==='train')flow.append(el('p',tr('All DP ranks execute all layers on different samples. ZeRO-3 gathers layer parameters; ZeRO-2/3 shard gradients; ZeRO-1/2/3 shard optimizer state. DP is not a pipeline.','所有 DP rank 在不同样本上执行全部层。ZeRO-3 聚合层参数；ZeRO-2/3 分片梯度；ZeRO-1/2/3 分片优化器状态。DP 不是流水线。')));
  const cp=el('details');cp.append(el('summary',tr('Explore context splitting (illustration only)','探索上下文切分（仅原理示意）')));box.append(cp);
  cp.append(el('p',tr('Changing this illustration does NOT change the capacity plan or launch commands. A real CP implementation must account for communication buffers, saved activations, attention layout and backend support.','调整此示意不会改变上方容量方案或启动命令。真正启用 CP 必须核算通信缓冲、保存的激活、注意力布局和后端支持。')));
  const label=el('label','CP '),input=el('select'),split=el('div',undefined,'context-split');label.append(input);cp.append(label,split);
  for(const v of [1,2,4,8]){const opt=el('option',v);opt.value=v;input.append(opt);}
  function context(){const c=Number(input.value),seq=map.ranks[0].tokens_per_sequence;split.replaceChildren();for(let i=0;i<c;i++){const start=Math.floor(i*seq/c),end=Math.floor((i+1)*seq/c),part=el('div',`CP${i}: tokens [${start}, ${end})`);split.append(part);}split.append(el('p',tr('Illustrative contiguous chunks; causal load balancing may use different token layouts. Attention still needs cross-chunk K/V exchange. Weights are not divided by CP.','这里示意连续切块；因果注意力负载均衡可能采用不同 token 布局。注意力仍需跨块交换 K/V，权重不因 CP 自动减少。')));}
  input.onchange=context;context();
}
const $=id=>document.getElementById(id);
const hash=new URLSearchParams(location.hash.slice(1));
const returnOrigin=safeWorkbenchOrigin(hash.get('return_origin')||'');
if(returnOrigin)sessionStorage.setItem('workbench-return-origin',returnOrigin);
let session=hash.get('token')||sessionStorage.getItem('workbench-session')||'';
if(session)sessionStorage.setItem('workbench-session',session);
if(hash.has('token'))history.replaceState(null,'',location.pathname+'#home');
const S={downloadTimer:null,traceEvents:[],traceLog:'',replayTimer:null,view:'home',info:{},forms:{},models:[],services:[],plans:{},reports:[],chat:[],chatReasoning:'',chatMetrics:{},chatId:null,chatCursor:0,chatTimer:null,chatService:null,chatEpoch:0,benchId:null,benchTimer:null,catalogResult:null,cloudDetail:null,manifest:null,selectedFiles:new Set(),inspect:null,tensorOffset:0,folder:null,scan:null};
const defaults={models:{folder:'',path:'',tensorSearch:''},discover:{vendor:'',min_b:'',max_b:'',after:'',before:'',modality:'',sort:'downloads',limit:50,source:'huggingface',author:'',search:'',repo:'',revision:'main',destination:''},data:{model_path:'',task:'sft',path:''},resources:{model_path:'',runtime_gib:0,communication_gib:0,cuda_graph_gib:0,other_models_gib:0,rollout_gib:0,parameters_b:7,task:'train',mode:'full',gpu:'h100',seq:4096,batch:1,gpus_per_node:8,extra_gpus:0,gpu_count:0,reserve_percent:20,bits:16,layers:0,hidden:0,heads:0,kv_heads:0,ffn:0,vocab:0,activation_gib:8,workspace_gib:4},train:{model_path:'',dataset:'',task:'sft',backend:'swift',tuner:'lora',steps:20,batch:1,grad_acc:8,learning_rate:.00001,gpus:1,tp:1,seq:2048,port:8000,profile:'none',backend_python:'',execution:'native'},optimize:{model_path:'',dataset:'',method:'awq',task:'quantize',gpus:1,tp:1,seq:2048,profile:'none',backend_python:'',execution:'native'},deploy:{model_path:'',task:'inference',backend:'vllm',gpus:1,tp:1,seq:2048,port:8000,profile:'none',backend_python:'',execution:'native'},connect:{name:'',url:'http://127.0.0.1:8000',model:'workbench'},playground:{service_id:'',system:'',temperature:.7,max_tokens:512,message:''},evaluate:{service_id:'',prompt:'Explain attention in three sentences.',requests:20,concurrency:1,max_tokens:128,label:'baseline'},remote:{alias:''}};
Object.assign(defaults.resources,{training_phase:'cpt',node_fabric:'unknown',train_tokens_b:0,global_batch_tokens:1048576,max_search_gpus:256,deadline_hours:0,packing_efficiency:1,effective_tflops:0,intra_gib_s:0,inter_gib_s:0,hbm_gib_s:0,gpu_hour_price:0,overhead_percent:15,expert_count:0,expert_topk:0,expert_parameter_fraction:0});
defaults.resources.curriculum_target_seq=32768;
for(const [k,v]of Object.entries(defaults))S.forms[k]={...v};
// Independent stage drafts. Import copies values; never alias another stage's
// object or carry its task type / execution approval into the current stage.
S.resourceDrafts={};
function plannerConfigImport(parent){
  const tr=(en,zh)=>language==='zh'?zh:en,box=el('details'),select=el('select');
  box.append(el('summary',tr('Reuse inputs from another stage (optional)','调取其他环节配置（可选）')));
  select.dataset.configSource='';select.setAttribute('aria-label',tr('Configuration source','配置来源'));
  for(const key of Object.keys(S.resourceDrafts).filter(key=>key!==S.view)){const o=el('option',t(key));o.value=key;select.append(o);}
  const importButton=button(tr('Copy inputs into this stage','复制输入到当前环节'),()=>{
    const source=S.resourceDrafts[select.value];if(!source)return;
    const task=S.forms.resources.task;
    S.resourceDrafts[S.view]={...source,task};render();
  },false,true);importButton.dataset.configImport='';importButton.disabled=!select.options.length;
  box.append(el('p',tr('Copies the model, hardware and estimate inputs, replacing this draft. Task type stays unchanged; recalculate here. No navigation or execution. Drafts last until reload.','复制模型、硬件及估算输入，替换当前草稿；保留当前任务类型，需在本页重新计算。不会跳页或执行。草稿仅保留到刷新页面前。')),select,importButton);parent.append(box);
}
function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);if(cls)n.className=cls;return n;}
function txt(k){return el('p',t(k));}
function contentViewer(parent,value,kind='json'){
  const box=el('div'),controls=el('div',undefined,'actions'),output=el('div');
  let mode=localStorage.getItem('workbench-view-'+kind)||'parsed';
  const parsed=el('button',t('parsedView')),plain=el('button',t('plainView'));
  parsed.type=plain.type='button';parsed.onclick=()=>draw('parsed');plain.onclick=()=>draw('plain');
  controls.setAttribute('role','group');controls.setAttribute('aria-label',t('viewMode'));
  controls.append(parsed,plain);box.append(controls,output);parent.append(box);
  function draw(next){mode=next;localStorage.setItem('workbench-view-'+kind,mode);for(const [b,v]of [[parsed,'parsed'],[plain,'plain']]){b.setAttribute('aria-pressed',String(mode===v));b.classList.toggle('primary',mode===v);}
    output.replaceChildren();
    if(mode==='plain'){output.append(el('pre',kind==='json'?JSON.stringify(value,null,2):value,'model-card-preview'));return;}
    if(kind==='markdown'){output.append(renderSafeMarkdown(value));return;}
    function tree(v,depth=0){if(v===null||typeof v!=='object')return el('span',typeof v==='string'?v:JSON.stringify(v));if(depth>=12)return el('pre',JSON.stringify(v,null,2));const list=el('dl',undefined,'structured-report');for(const [key,item]of Object.entries(v)){list.append(el('dt',key));const dd=el('dd');dd.append(tree(item,depth+1));list.append(dd);}return list;}
    output.append(tree(value));
  }draw(mode);
}
function renderSafeMarkdown(source){
  const root=el('div',undefined,'markdown-preview');
  let text=String(source).replace(/\r\n/g,'\n');
  text=text.replace(/^---\n[\s\S]*?\n---(?:\n|$)/,'');
  const fragment=DOMPurify.sanitize(marked.parse(text,{gfm:true,breaks:false}),{
    RETURN_DOM_FRAGMENT:true,FORBID_TAGS:['script','style','iframe','form','input','button','video','audio','object','embed'],
    FORBID_ATTR:['style','srcset'],ALLOW_DATA_ATTR:false
  });
  const m=S.cloudDetail;
  for(const a of fragment.querySelectorAll('a')){
    try{const u=new URL(a.getAttribute('href')||'',m?.url+'/');if(!['https:','http:'].includes(u.protocol))throw Error();a.href=u.href;a.target='_blank';a.rel='noopener noreferrer';}
    catch(e){a.removeAttribute('href');}
  }
  // Never fetch arbitrary remote media automatically; replace with an explicit safe link.
  for(const img of fragment.querySelectorAll('img')){
    const label=img.alt||t('imageLink'),a=el('a',label+' ↗');
    try{const base=m?.source==='modelscope'?m.url+'/':(m?.url+'/resolve/'+encodeURIComponent(m?.revision||'main')+'/');const u=new URL(img.getAttribute('src')||'',base);if(u.protocol!=='https:')throw Error();a.href=u.href;a.target='_blank';a.rel='noopener noreferrer';img.replaceWith(a);}
    catch(e){img.replaceWith(el('span',label));}
  }
  root.append(fragment);return root;
}
async function runSpeedTest(parent){
  let result=parent.querySelector('.speed-result');if(!result){result=el('div',undefined,'speed-result');parent.querySelector('.actions')?.after(result);}result.setAttribute('role','status');result.setAttribute('aria-live','polite');
  const started=Date.now(),source=S.forms.discover.source;const status=el('p');result.replaceChildren(status);const update=()=>status.textContent=t('speedRunning')+' · '+source+' · '+Math.floor((Date.now()-started)/1000)+' s';update();result.scrollIntoView({block:'nearest'});const timer=setInterval(update,1000);
  try{const r=await api('/api/hub/speed',{manifest_id:S.manifest.id,source});result.replaceChildren(el('h3',t(r.ok?'speedDone':'speedFailed')));if(r.ok){cards(result,{source:r.source,speedRate:r.mib_per_s.toFixed(2)+' MiB/s',speedBytes:r.bytes.toLocaleString()+' bytes',speedTime:r.seconds.toFixed(2)+' s',speedFirst:r.first_response_s.toFixed(2)+' s'});result.append(txt('speedCaveat'));}else result.append(el('p',r.error,'danger'));raw(result,r);}catch(e){result.replaceChildren(el('h3',t('speedFailed')),el('p',String(e.message),'danger'));}finally{clearInterval(timer);}
}
function raw(parent,obj){const d=el('details');d.append(el('summary',t('raw')));contentViewer(d,obj);parent.append(d);}
function fail(e){if(e.status===401){showSessionRecovery();return;}$('alert').hidden=false;$('alert').replaceChildren(el('strong',t('error')));raw($('alert'),{diagnostic:String(e.message||e)});}
async function api(path,body){const r=await fetch(path,{method:body===undefined?'GET':'POST',headers:{'X-Workbench-Token':session,...(body===undefined?{}:{'Content-Type':'application/json'})},body:body===undefined?undefined:JSON.stringify(body)});const v=await r.json();if(!r.ok){const e=Error(v.error||r.status);e.status=r.status;throw e;}return v;}
function showSessionRecovery(){
  S.info={};$('connection').textContent=t('sessionExpired');
  if(document.getElementById('session-recovery'))return;
  for(const d of document.querySelectorAll('dialog[open]'))d.close();
  const dialog=el('dialog',undefined,'directory-picker');dialog.id='session-recovery';
  dialog.setAttribute('aria-label',t('reconnect'));
  const input=el('input');input.type='password';input.autocomplete='off';input.spellcheck=false;
  input.setAttribute('aria-label',t('sessionLink'));
  const label=el('label',t('sessionLink'));label.append(input);
  const error=el('p',undefined,'danger');
  dialog.append(el('h2',t('sessionExpired')),txt('sessionHelp'),label,error);
  actions(dialog,button('cancel',()=>dialog.close()),button('reconnect',async()=>{
    let candidate=input.value.trim();
    if(candidate.includes('://')){
      try{const url=new URL(candidate);if(url.origin!==location.origin)throw Error();candidate=new URLSearchParams(url.hash.slice(1)).get('token')||'';}
      catch(e){error.textContent=t('sessionWrongHost');return;}
    }
    if(!candidate||candidate.length>512){error.textContent=t('sessionInvalid');return;}
    const previous=session;session=candidate;
    try{const info=await api('/api/info');sessionStorage.setItem('workbench-session',candidate);input.value='';dialog.close();$('alert').hidden=true;await initializeWorkspace(info);}
    catch(e){session=previous;error.textContent=t(e.status===401?'sessionInvalid':'sessionUnavailable');}
  },true));
  dialog.addEventListener('close',()=>dialog.remove(),{once:true});document.body.append(dialog);dialog.showModal();input.focus();
}
function button(k,fn,primary=false,literal=false){const b=el('button',literal?k:t(k),primary?'primary':'');b.type='button';b.onclick=async()=>{b.disabled=true;$('alert').hidden=true;try{await fn();}catch(e){fail(e);}finally{b.disabled=false;}};return b;}
function panel(k,parent=$('view')){const p=el('div',undefined,'panel');if(k)p.append(el('h2',t(k)));parent.append(p);return p;}
function actions(parent,...buttons){const a=el('div',undefined,'actions');a.append(...buttons);parent.append(a);return a;}
function fields(parent){const g=el('div',undefined,'grid');parent.append(g);return g;}
function field(parent,group,key,label,type='text',options=null){const f=S.forms[group],l=el('label',t(label));const i=document.createElement(options?'select':type==='textarea'?'textarea':'input');i.id=group+'_'+key;i.name=key;if(options)for(const [v,name]of options){const o=el('option',M[name]?t(name):name);o.value=v;i.append(o);}else if(type!=='textarea'){i.type=type;if(type==='number')i.step='any';}i.value=f[key]??'';const capture=()=>{f[key]=type==='number'&&i.value!==''?Number(i.value):i.value;if(S.plans[group]){delete S.plans[group];document.querySelectorAll('[data-run="'+group+'"]').forEach(b=>b.disabled=true);}};i.addEventListener('input',capture);i.addEventListener('change',capture);l.append(i);parent.append(l);return i;}
function table(parent,headers,rows){const wrap=el('div',undefined,'table'),tb=el('table'),h=tb.createTHead().insertRow();for(const x of headers)h.append(el('th',M[x]?t(x):x));const body=tb.createTBody();for(const row of rows){const tr=body.insertRow();for(const value of row){const td=tr.insertCell();if(value instanceof Node)td.append(value);else td.textContent=String(value??t('none'));}}wrap.append(tb);parent.append(wrap);return tb;}
function cards(parent,values){const g=el('div',undefined,'cards');for(const [k,v]of Object.entries(values)){const d=el('div',undefined,'card');d.append(el('small',M[k]?t(k):k),el('strong',v??t('none')));g.append(d);}parent.append(g);return g;}
const fmt=n=>Number.isFinite(n)?n.toFixed(2):t('none');const bytes=n=>Number.isFinite(n)?(n/2**30).toFixed(3)+' GiB':t('none');
function navigate(v){if(!['home','trainFlow','deployFlow','trainingPlan','deploymentPlan','academy','infrastructure','models','discover','cloudDetail','data','resources','train','optimize','deploy','playground','evaluate','runs'].includes(v))v='home';if(v!=='playground')clearInterval(S.replayTimer);S.view=v;location.hash=v;render();}
function shell(){document.documentElement.lang=language==='zh'?'zh-CN':'en';$('language').value=language;$('title').textContent=t(S.view);$('edition').textContent=t('preview');$('footer').textContent=t('footer');$('runsButton').textContent=t('runs');$('traceLabel').textContent=t('trace');$('traceRefresh').textContent=t('refresh');$('connection').textContent=t(Object.keys(S.info).length?'connected':'connecting');$('nav').replaceChildren();for(const [group,views]of [['',['home']],['workflowPlanning',['trainingPlan','deploymentPlan']],['assets',['academy','models','discover','data','resources']],['build',['train','optimize']],['serve',['deploy','playground','evaluate']]]){if(group)$('nav').append(el('small',t(group)));for(const v of views){const b=button(v,()=>navigate(v));b.classList.toggle('active',S.view===v||(v==='trainingPlan'&&S.view==='trainFlow')||(v==='deploymentPlan'&&S.view==='deployFlow'));$('nav').append(b);}}}
function render(){clearTimeout(S.downloadTimer);clearTimeout(S.deployTimer);clearTimeout(S.infrastructureTimer);shell();renderExecutionHost().catch(fail);if(S.info.hostname)$('connection').textContent=S.info.hostname;const infraNav=button('infrastructure',()=>navigate('infrastructure'));infraNav.classList.toggle('active',S.view==='infrastructure');$('nav').append(infraNav);$('view').replaceChildren();if(S.view!=='home')workflowNavigation();({home:homeView,trainFlow:()=>plannerEntry('train',true),deployFlow:()=>plannerEntry('inference',true),trainingPlan:()=>plannerEntry('train',false),deploymentPlan:()=>plannerEntry('inference',false),academy:academyView,infrastructure:infrastructureView,models:modelsView,discover:discoverView,cloudDetail:cloudDetailView,data:dataView,resources:resourcesView,train:()=>jobView('train'),optimize:()=>jobView('optimize'),deploy:deployView,playground:playgroundView,evaluate:evaluateView,runs:runsView})[S.view]();}
function environmentDefaults(recipe='download'){
  const c=S.info.conda_installations?.[0];
  return {manager:'conda',conda:c?.executable||'',python_version:'3.11',path:c?c.envs_dir+'/workbench-'+recipe:'',python:S.info.python||'',recipe,version:''};
}
function configureEnvironment(group='deploy'){
  const recipe=group==='train'?'train':group==='optimize'?(S.forms.optimize.method==='mlx4'?'mlx':'train'):S.forms.deploy.backend;
  S.forms.envSetup={...environmentDefaults(recipe),...(S.forms.envSetup||{}),recipe,version:recipe==='train'?'4.5.3':''};
  navigate('infrastructure');$('envSetup_path')?.scrollIntoView({block:'center'});
}
function infrastructureView(){
  S.forms.envSetup??=environmentDefaults();S.forms.cloudSetup??={alias:'',root:'',local_port:8876,remote_port:8876};
  const host=panel('executionHost');host.append(el('p',S.info.hostname||S.info.execution_host),txt('hostBoundary'));
  environmentPicker(host,'deploy',true);
  actions(host,button('probe',async()=>raw(host,await api('/api/hardware'))));
  const cloud=panel('cloudSetup');cloud.append(txt('cloudSetupHelp'));const cg=fields(cloud);for(const k of ['alias','root'])field(cg,'cloudSetup',k,k==='root'?'remoteRoot':'alias');for(const k of ['local_port','remote_port'])field(cg,'cloudSetup',k,k,'number');
  actions(cloud,button('remote',async()=>raw(cloud,await api('/api/remote',{alias:S.forms.cloudSetup.alias}))));
  const env=panel('envSetup');env.append(txt('envSetupHelp'));const eg=fields(env);field(eg,'envSetup','manager','envManager','text',[['conda','Conda · Miniforge / Miniconda'],['venv','Python venv']]);field(eg,'envSetup','conda','condaExecutable','text',S.info.conda_installations?.length?S.info.conda_installations.map(c=>[c.executable,c.executable]):undefined);field(eg,'envSetup','python_version','condaPython','text',[['3.10','3.10'],['3.11','3.11'],['3.12','3.12'],['3.13','3.13']]);field(eg,'envSetup','recipe','envRecipe','text',[['download','Download SDKs'],['vllm','vLLM'],['sglang','SGLang'],['train','ms-swift / PyTorch'],['mlx','MLX LM']]);field(eg,'envSetup','path','envPath');field(eg,'envSetup','python','python');field(eg,'envSetup','version','envVersion');
  const managerFields=()=>{const isConda=S.forms.envSetup.manager==='conda';$('envSetup_conda').disabled=!isConda;$('envSetup_python_version').disabled=!isConda;$('envSetup_python').disabled=isConda;};managerFields();
  $('envSetup_manager').addEventListener('change',()=>{const f=S.forms.envSetup,c=S.info.conda_installations?.find(x=>x.executable===f.conda);f.path=f.manager==='conda'?(c?c.envs_dir+'/workbench-'+f.recipe:''):S.info.state+'/env-'+f.recipe;$('envSetup_path').value=f.path;managerFields();});
  $('envSetup_conda').addEventListener('change',()=>{const f=S.forms.envSetup,c=S.info.conda_installations?.find(x=>x.executable===f.conda);if(c){f.path=c.envs_dir+'/workbench-'+f.recipe;$('envSetup_path').value=f.path;}});
  const jobs=panel('infrastructureJobs'),list=el('div'),links=el('div'),logs=el('pre',t('empty'));jobs.append(links,list,logs);let selected=null;
async function preview(parent,group,route){parent.querySelectorAll('.environment-preview').forEach(x=>x.remove());const snapshot=JSON.stringify(S.forms[group]);const status=el('div',undefined,'environment-preview');parent.append(status);status.append(txt('loading'));status.scrollIntoView({block:'center'});if(group==='envSetup'&&!S.forms[group].path.trim()){status.replaceChildren(el('p',t('envPathRequired'),'danger'));return;}let job;try{job=await api(route,S.forms[group]);}catch(e){status.replaceChildren(el('p',t('envPreviewFailed')+String(e.message||e),'danger'));return;}status.replaceChildren();selected=job.id;const box=el('div',undefined,'environment-preview');parent.append(box);box.append(el('h3',t('planPreview')),el('pre',JSON.stringify(job.plan.steps,null,2)));raw(box,job.plan);const run=button('run',async()=>{if(JSON.stringify(S.forms[group])!==snapshot){box.replaceChildren(el('p',t('envChanged'),'danger'));return;}if(!confirm(t('infraConfirm')))return;await api('/api/start',{id:job.id,confirm:job.id,native:!!S.info.native_terminal});run.disabled=true;await refresh();},true);run.disabled=!!job.plan.blockers.length;box.append(run);}
  actions(cloud,button('planPreview',()=>preview(cloud,'cloudSetup','/api/cloud/plan')));actions(env,button('planPreview',()=>preview(env,'envSetup','/api/environment/plan')));
  async function refresh(){
    if(S.view!=='infrastructure')return;
    const rows=(await api('/api/jobs')).filter(x=>['cloud-workspace','environment'].includes(x.task));
    if(S.view!=='infrastructure')return;
    list.replaceChildren();table(list,['name','task','status','actions'],rows.map(r=>{
      const a=el('div',undefined,'actions');a.append(button('details',()=>{selected=r.id;refresh();}));
      if(r.task==='cloud-workspace'&&r.status==='running')a.append(button('openCloud',async()=>{
        const x=await api('/api/cloud/access',{id:r.id}),link=el('a',t('openCloud')+' · '+x.root);
        link.href='http://127.0.0.1:'+x.port+'/#token='+encodeURIComponent(x.token);link.target='_blank';link.rel='noopener noreferrer';links.replaceChildren(link);
      }));
      if(r.task==='environment'&&r.status==='succeeded')a.append(button('useEnvironment',async()=>{
        const job=await api('/api/job?id='+r.id),py=job.plan.request.env_python||job.plan.request.path+'/bin/python';
        for(const group of ['train','deploy','optimize'])S.forms[group].backend_python=py;
        links.replaceChildren(el('p',t('environmentSelected')+' '+py));
      }));
      if(['running','starting'].includes(r.status))a.append(button('stop',async()=>{if(confirm(t('infraStop')))await api('/api/stop',{id:r.id});await refresh();}));
      return [r.id.slice(0,8),r.task,t(r.status),a];
    }));
    if(selected){const result=await api('/api/logs?id='+selected);if(S.view==='infrastructure')logs.textContent=result.text||t('empty');}
    clearTimeout(S.infrastructureTimer);if(S.view==='infrastructure')S.infrastructureTimer=setTimeout(()=>refresh().catch(fail),1500);
  }
  refresh().catch(fail);
}
function environmentPicker(parent,group,remote=false){
  const box=el('section'),result=el('div'),path=el('input');path.placeholder=t('envSearchPath');path.setAttribute('aria-label',t('envSearchPath'));
  box.append(el('h3',t('envScan')),txt('envScanHelp'),path);parent.append(box);
  const recipe=()=>group==='train'?'train':group==='optimize'?(S.forms.optimize.method==='mlx4'?'mlx':'train'):S.forms.deploy.backend;
  const target=el('p',t('envTarget')+' '+recipe());box.prepend(target);
  setTimeout(()=>{for(const key of ['backend','method'])$(group+'_'+key)?.addEventListener('change',()=>{target.textContent=t('envTarget')+' '+recipe();result.replaceChildren(txt('envChanged'));});},0);
  async function scan(alias=''){
    result.replaceChildren(txt('loading'));local.disabled=true;if(remoteButton)remoteButton.disabled=true;
    try{
      const report=await api('/api/environment/scan',{recipe:recipe(),alias,paths:!alias&&path.value.trim()?[path.value.trim()]:[]});
      result.replaceChildren(el('p',report.hostname+' · '+report.os+' / '+report.machine));
      if(alias)result.append(txt('envRemoteNote'));
      if(!report.environments.some(x=>x.candidate))result.append(txt('envMissing'));
      table(result,['python','envPackages','status','actions'],report.environments.map(row=>{
        const use=button('envSelect',()=>{S.forms[group].backend_python=row.path;delete S.plans[group];const input=$(group+'_backend_python');if(input)input.value=row.path;document.querySelectorAll('[data-run="'+group+'"]').forEach(b=>b.disabled=true);result.append(el('p',t('environmentSelected')+' '+row.path));});
        use.disabled=!!alias||!row.candidate;
        return [row.path+' · '+(row.version||[]).join('.'),Object.entries(row.packages||{}).filter(([,v])=>v).map(([k,v])=>k+' '+v).join(' · ')||'—',row.candidate?t('envCandidate'):row.reasons.join('; '),use];
      }));raw(result,report);
    }catch(e){result.replaceChildren(el('p',String(e),'danger'));throw e;}
    finally{local.disabled=false;if(remoteButton)remoteButton.disabled=false;}
  }
  const local=button('envScan',()=>scan());let remoteButton=null;
  actions(box,local,button('envConfigure',()=>configureEnvironment(group)));
  if(remote){remoteButton=button('envScanRemote',()=>{const alias=S.forms.cloudSetup?.alias?.trim();if(!alias)throw Error(t('alias'));return scan(alias);});actions(box,remoteButton);}
  box.append(result);
}
function formatParameters(n){return Number.isFinite(n)&&n>0?(n/1e9).toFixed(3)+' B · '+n.toLocaleString():t('unknownParameters');}
async function openCloudDetail(source,repo){S.cloudDetail=await api('/api/hub/detail',{source,repo,revision:S.forms.discover.revision||'main'});navigate('cloudDetail');}
function cloudDetailView(){
  const m=S.cloudDetail,p=panel('cloudDetail');
  actions(p,button('backCatalog',()=>navigate('discover')));
  if(!m){p.append(txt('chooseCloudModel'));return;}
  p.append(el('h2',m.repo));
  cards(p,{publisherParameters:(m.parameter_claims||[]).join(' · ')||t('none'),hubElements:formatParameters(m.parameters),modality:m.pipeline_tag||t('none'),license:typeof m.license==='string'?m.license:JSON.stringify(m.license),library:m.library_name||t('none')});
  p.append(txt('parameterMeaning'));
  const link=el('a',t('publisherPage'));link.href=m.url;link.target='_blank';link.rel='noopener noreferrer';p.append(link);
  actions(p,button('downloadFiles',async()=>{S.forms.discover.source=m.source;S.forms.discover.repo=m.repo;S.forms.discover.revision=m.revision;S.manifest=await api('/api/hub/manifest',{source:m.source,repo:m.repo,revision:m.revision});S.selectedFiles.clear();navigate('discover');},true));
  table(p,['fieldName','value'],[['repo',m.repo],['source',m.source],['revision',m.revision],['created_at',m.created_at],['last_modified',m.last_modified],['downloadCount',m.downloads],['likes',m.likes],['gated',m.gated]].map(([k,v])=>[M[k]?t(k):k,v]));
  const structure=panel('cloudArchitecture'),c=m.config||{},text=c.text_config||c;
  table(structure,['fieldName','value'],['model_type','architectures','num_hidden_layers','hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','vocab_size','max_position_embeddings','num_experts','num_local_experts','num_experts_per_tok','torch_dtype','dtype','quantization_config'].map(k=>[k,JSON.stringify(text[k]??c[k])??t('none')]));
  raw(structure,c);
  const card=panel('modelCard');card.append(txt('cardPreviewNote'));contentViewer(card,m.readme||t('cardUnavailable'),'markdown');raw(card,m.card);if(m.errors?.length)raw(card,m.errors);
}
// Navigation is not execution or a completion ledger. Existing preflight and
// explicit execution confirmation remain the only way to start a job.
function workflowNavigation(){
  const tr=(en,zh)=>language==='zh'?zh:en;
  const box=el(S.view==='home'?'section':'details',undefined,'workflow-map');box.setAttribute('aria-label',tr('Workflow navigation','流水线导航'));
  if(S.view!=='home')box.append(el('summary',tr('Switch stage (optional; not required for this tool)','切换环节（可选，当前工具不要求跳转）')));
  box.append(el('h2',tr('One workflow, enter at any step','流程清晰，每一环都可独立进入')),el('p',tr('Steps are suggested order, not completion status. Bring existing inputs or jump to any tool.','步骤表示建议顺序，不代表已完成。已有模型、数据或方案时，可直接进入对应环节。')));
  const lines=[['Training','训练',[['models','选择模型'],['trainingPlan','训练规划'],['data','准备数据'],['train','预检与训练'],['runs','日志与产物']]],['Deployment','部署',[['models','选择模型'],['deploymentPlan','部署规划'],['optimize','可选压缩'],['deploy','预检与部署'],['playground','试用'],['evaluate','服务测试']]]];
  for(const [en,zh,steps]of lines){const row=el('nav',undefined,'workflow-steps');row.setAttribute('aria-label',tr(en,zh));row.append(el('strong',tr(en,zh)));
    steps.forEach(([route,label],i)=>{const current=S.view===route||(route==='trainingPlan'&&S.view==='trainFlow')||(route==='deploymentPlan'&&S.view==='deployFlow');const a=el('a',`${i+1}. ${language==='zh'?label:t(route)}`);a.href='#'+route;if(current)a.setAttribute('aria-current','step');row.append(a);});box.append(row);}
  $('view').append(box);
}
function plannerEntry(task,execution){
  const tr=(en,zh)=>language==='zh'?zh:en;
  const p=panel(S.view);
  p.classList.add('planner-intro');
  p.append(el('p',execution?tr('Start with a plan; proceed to preflight and execution when ready. An existing plan? Jump directly below.','先规划，再预检与执行。已有方案？可以直接进入执行环节。'):tr('Planning only: inspect assumptions, hardware and budgets. No training or service will be started.','仅制定方案：理解假设、硬件与预算，不启动训练或服务。')));
  if(execution)actions(p,button(task==='train'?'train':'deploy',()=>navigate(task==='train'?'train':'deploy')));
  resourcesView(task);
}
function homeView(){
  const tr=(en,zh)=>language==='zh'?zh:en,p=panel();p.classList.add('hero');
  p.append(el('h2',tr('What would you like to do?','你想完成什么？')),el('p',tr('Choose a goal first. Plan only, or continue to a confirmed execution.','先选择目标：只做方案，或者沿流程完成预检与执行。')));
  const grid=el('div',undefined,'intent-grid');
  for(const [route,en,zh]of [
    ['trainFlow','Plan → data → preflight → training → logs and artifacts','训练规划 → 数据准备 → 预检训练 → 日志与产物'],
    ['deployFlow','Plan → optional compression → preflight → serving → test','部署规划 → 可选压缩 → 预检部署 → 试用与测试'],
    ['trainingPlan','Understand training objectives, data requirements, hardware and memory budgets. No execution.','明确训练目标、数据要求、硬件和显存预算。只做方案，不执行。'],
    ['deploymentPlan','Estimate hardware, concurrency, context limits and serving constraints. No execution.','评估硬件、并发、上下文和服务约束。只做方案，不执行。']]){
    const a=el('a',undefined,'intent-card');a.href='#'+route;a.dataset.intent=route;a.append(el('h3',t(route)),el('p',tr(en,zh)),el('span',tr('Open planner →','进入规划器 →')));grid.append(a);
  }p.append(grid);workflowNavigation();
  const tools=panel('guided');tools.append(el('p',tr('Use a tool independently; completing earlier steps is not required.','独立使用工具，无需先完成前置页面；实际执行仍需满足输入与环境检查。')));actions(tools,...['academy','models','data','resources','train','deploy','runs'].map(v=>button(v,()=>navigate(v))));
}
function remember(m){S.models=S.models.filter(x=>x.path!==m.path);S.models.push(m);}
async function inspect(path){const m=await api('/api/scan',{path});S.inspect=m;remember(m);S.forms.models.path=m.path;S.tensorOffset=0;if(S.view==='models')render();}
function modelsView(){const p=panel('local');p.append(txt('hostPaths'));field(p,'models','folder','folder');actions(p,button('selectDirectory',()=>openDirectoryPicker(),true),button('scanFolder',async()=>{S.scan=await api('/api/discover',{path:S.forms.models.folder});for(const m of S.scan.models)if(m.fingerprint)remember(m);render();}),button('discover',()=>navigate('discover')));const roots=el('div',undefined,'actions');for(const r of S.info.roots||[])roots.append(button(r,async()=>{S.forms.models.folder=r;S.folder=await api('/api/folders',{path:r});render();},false,true));p.append(roots);if(S.folder){const children=el('div',undefined,'folder-list');if(S.folder.parent)children.append(button('parent',async()=>{S.folder=await api('/api/folders',{path:S.folder.parent});S.forms.models.folder=S.folder.path;render();}));for(const d of S.folder.directories)children.append(button(d.name,async()=>{S.folder=await api('/api/folders',{path:d.path});S.forms.models.folder=S.folder.path;render();},false,true));p.append(children);}p.append(txt('scanLimit'));if(S.scan){table(p,['name','architecture','weightSize','status','actions'],S.scan.models.map(m=>[m.path,m.model_type||m.pipeline,bytes(m.weight_bytes),m.errors?.length?t('blocked'):t('inspect'),button('inspect',()=>inspect(m.path))]));raw(p,S.scan);}const detail=panel('structure');field(detail,'models','path','path');actions(detail,button('scan',()=>inspect(S.forms.models.path),true));const m=S.inspect;if(!m){detail.append(txt('scanFirst'));return;}cards(detail,{name:m.name,architecture:m.model_type||m.pipeline,modality:m.modality,weightSize:bytes(m.weight_bytes),parameters:m.parameter_estimate?.toLocaleString(language==='zh'?'zh-CN':'en-US')});actions(detail,...['train','optimize','deploy'].map(v=>button(v,()=>{S.forms[v].model_path=m.path;delete S.plans[v];navigate(v);})));table(detail,['component','parameters'],(m.structure?.components||[]).map(c=>[c.name||'.',c.stored_elements]));raw(detail,m);const tensors=panel('files');field(tensors,'models','tensorSearch','search');const target=el('div');async function load(){const r=await api('/api/tensors',{path:m.path,offset:S.tensorOffset,search:S.forms.models.tensorSearch});target.replaceChildren();table(target,['name','shape','dtype'],r.tensors.map(x=>[x.name,Array.isArray(x.shape)?(x.shape.length?x.shape.join(' × '):t('scalarShape')):t('none'),x.dtype]));target.append(el('p',`${r.offset} / ${r.total}`));}actions(tensors,button('search',async()=>{S.tensorOffset=0;await load();}),button('previous',async()=>{S.tensorOffset=Math.max(0,S.tensorOffset-100);await load();}),button('next',async()=>{S.tensorOffset+=100;await load();}));tensors.append(target);load().catch(fail);}
async function openDirectoryPicker(options={}){
  if(!S.info.roots?.length){showSessionRecovery();return;}
  const dialog=el('dialog',undefined,'directory-picker');
  dialog.setAttribute('aria-labelledby','directory-picker-title');
  const title=el('h2',t('selectDirectory'));title.id='directory-picker-title';
  const description=txt('directoryHelp'), path=el('input'), label=el('label',t('folder'));
  path.setAttribute('aria-label',t('folder'));label.append(path);
  const error=el('p',undefined,'danger'), listing=el('div',undefined,'directory-entries');
  let current=null,browseVersion=0;
  const choose=button('useDirectory',()=>{if(!current)return;if(options.onSelect)options.onSelect(current.path);else{S.forms.models.folder=current.path;S.folder=null;S.scan=null;}dialog.close();render();},true);
  choose.disabled=true;
  async function browse(value){
    const version=++browseVersion;
    choose.disabled=true;current=null;listing.replaceChildren();error.textContent='';
    try{
      const r=await api('/api/folders',{path:value});if(version!==browseVersion||!dialog.open)return;current=r;path.value=r.path;
      if(r.parent)listing.append(button('parent',()=>browse(r.parent)));
      for(const d of r.directories)listing.append(button('📁 '+d.name,()=>browse(d.path),false,true));
      if(!r.directories.length)listing.append(txt('noSubfolders'));
      choose.disabled=false;
    }catch(e){if(e.status===401){showSessionRecovery();return;}if(version===browseVersion&&dialog.open)error.textContent=t('directoryDenied')+' '+String(e.message||e);}
  }
  path.oninput=()=>{browseVersion++;current=null;choose.disabled=true;};
  path.onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();browse(path.value);}};
  const roots=el('div',undefined,'actions');for(const root of S.info.roots||[])roots.append(button(root,()=>browse(root),false,true));
  dialog.append(title,description,label);actions(dialog,button('browse',()=>browse(path.value)));
  if(options.create){const name=el('input'),l=el('label',t('newFolderName'));name.setAttribute('aria-label',t('newFolderName'));l.append(name);dialog.append(l);actions(dialog,button('createFolder',async()=>{if(!current)return;try{const r=await api('/api/folders/create',{parent:current.path,name:name.value});name.value='';await browse(r.path);}catch(e){error.textContent=String(e.message||e);}}));}
  dialog.append(txt('roots'),roots,error,listing);actions(dialog,button('cancel',()=>dialog.close()),choose);
  dialog.addEventListener('close',()=>dialog.remove(),{once:true});document.body.append(dialog);dialog.showModal();
  await browse(options.start||S.forms.models.folder||S.info.roots?.[0]||'');
}
function discoverView(){const p=panel('discover');p.append(txt('sourceNote'));const g=fields(p);field(g,'discover','source','source','text',Object.entries(S.info.download_sources||{}).map(([k])=>[k,k==='modelscope'?'ModelScope / 魔搭':k]));const vendor=field(g,'discover','vendor','vendor','text',[['',t('chooseVendor')],['Qwen','Alibaba · Qwen'],['all',t('allVendors')],['custom',t('customVendor')],...Object.entries(S.info.model_vendors||{}).filter(([k])=>k!=='Qwen')]);vendor.addEventListener('change',()=>render());if(S.forms.discover.vendor==='custom'){field(g,'discover','author','author');}else if(S.forms.discover.vendor&&S.forms.discover.vendor!=='all'){g.append(el('p',t('organizationId')+': '+S.forms.discover.vendor));}field(g,'discover','search','keyword');field(g,'discover','min_b','minB','number');field(g,'discover','max_b','maxB','number');field(g,'discover','after','dateAfter','date');field(g,'discover','before','dateBefore','date');field(g,'discover','modality','modality','text',[['',t('allModalities')],...['text-generation','image-text-to-text','text-to-image','text-to-video','image-to-video','text-to-speech','automatic-speech-recognition','audio-text-to-text'].map(x=>[x,x])]);field(g,'discover','sort','sort','text',[['downloads',t('sortPopular')],['lastModified',t('sortUpdated')]]);field(g,'discover','limit','perVendorLimit','number');p.append(txt('filterScope'));const catalog=el('div',undefined,'catalog');actions(p,button('sync',async()=>{const vendor=S.forms.discover.vendor;if(!vendor)throw Error(t('chooseVendor'));const author=vendor==='custom'?S.forms.discover.author.trim():vendor;if(!author)throw Error(t('enterOrganization'));const r=await api('/api/catalog/sync',{...S.forms.discover,author});S.catalogResult=r;drawCatalog();},true));p.append(catalog);function drawCatalog(){const r=S.catalogResult;if(!r)return;catalog.replaceChildren(el('p',t('foundCount')+r.models.length+' / '+r.fetched_count));for(const m of r.models){const a=el('article');a.append(button(m.id,()=>openCloudDetail(r.source,m.id),false,true),el('h3',t('parameters')+': '+formatParameters(m.parameters)),el('p',(m.pipeline_tag||'—')+' · '+(m.license||'—')),button('cloudDetail',()=>openCloudDetail(r.source,m.id)));catalog.append(a);}if(r.errors?.length)raw(catalog,r.errors);}drawCatalog();const q=panel('manifest');const gf=fields(q);field(gf,'discover','repo','repo');field(gf,'discover','revision','revision');actions(q,button('manifest',async()=>{S.manifest=await api('/api/hub/manifest',S.forms.discover);S.selectedFiles.clear();render();},true));if(S.manifest){q.append(el('p',S.manifest.repo+' · '+S.manifest.commit+' · '+bytes(S.manifest.total_bytes)));actions(q,button('selectAll',()=>{S.selectedFiles=new Set(S.manifest.files.map(x=>x.name));render();}),button('speed',async()=>{await runSpeedTest(q);}));const list=el('div',undefined,'file-list');for(const f of S.manifest.files){const l=el('label'),c=el('input');c.type='checkbox';c.checked=S.selectedFiles.has(f.name);c.onchange=()=>{c.checked?S.selectedFiles.add(f.name):S.selectedFiles.delete(f.name);};l.append(c,el('span',f.name+' · '+bytes(f.size)));list.append(l);}q.append(list);field(q,'discover','destination','destination');q.append(txt('downloadFolderHelp'));actions(q,button('chooseDownloadFolder',()=>openDirectoryPicker({create:true,start:S.forms.discover.download_parent,onSelect:path=>{S.forms.discover.download_parent=path;S.forms.discover.destination=path.replace(/\/$/,'')+'/'+S.forms.discover.repo.split('/').pop();}})));actions(q,button('download',async()=>{if(!confirm(t('confirm')))return;const r=await api('/api/downloads/prepare',{manifest_id:S.manifest.id,destination:S.forms.discover.destination,files:[...S.selectedFiles]});await api('/api/downloads/start',{id:r.id,confirm:r.id,source:S.forms.discover.source});await downloadList();},true));}const d=panel('downloads'),dl=el('div');d.append(dl);async function downloadList(){clearTimeout(S.downloadTimer);const rs=await api('/api/downloads');dl.replaceChildren();for(const r of rs){const a=el('article');a.append(el('h3',r.repo+' · '+t(r.status)),el('p',r.destination));const pr=el('progress');pr.max=Math.max(1,r.total_bytes);pr.value=r.downloaded_bytes;a.append(pr,el('p',bytes(r.downloaded_bytes)+' / '+bytes(r.total_bytes)+' · '+(r.bytes_per_s/1048576||0).toFixed(2)+' MiB/s · '+(r.total_bytes?100*r.downloaded_bytes/r.total_bytes:0).toFixed(1)+'%'));if(['downloading','verifying'].includes(r.status))a.append(button('pause',async()=>{await api('/api/downloads/pause',{id:r.id});await downloadList();}));if(['paused','failed','planned'].includes(r.status))a.append(button('resume',async()=>{if(!confirm(t('confirm')))return;await api('/api/downloads/start',{id:r.id,confirm:r.id,source:S.forms.discover.source});await downloadList();}));if(r.status==='completed')a.append(button('inspect',async()=>{await inspect(r.destination);navigate('models');}));if(r.current_file)a.append(el('p',r.current_file));if(!['downloading','verifying','pausing'].includes(r.status))a.append(button('removeRecord',async()=>{if(!confirm(t('removeRecordConfirm')))return;await api('/api/downloads/remove',{id:r.id});await downloadList();}));if(r.error)raw(a,r.error);dl.append(a);}if(S.view==='discover'&&dl.isConnected)S.downloadTimer=setTimeout(()=>downloadList().catch(fail),1500);}actions(d,button('refresh',downloadList));downloadList().catch(fail);}
function dataView(){
  const entry=panel('academy');actions(entry,button('academy',()=>{S.academyTab='training';navigate('academy');}));
  const p=panel('contract'),f=S.forms.data;p.append(txt('dataNote'),txt('dataScope'));
  const g=fields(p);
  if(S.models.length){const m=field(g,'data','known_model','knownModel','text',[['','manual'],...S.models.map(m=>[m.path,m.name+' · '+m.model_type])]);m.addEventListener('change',()=>{f.model_path=m.value;$('data_model_path').value=m.value;load();});}
  field(g,'data','model_path','path');
  const task=field(g,'data','task','task','text',[['cpt','CPT / Pretraining'],['sft','SFT'],['dpo','DPO'],['ppo','PPO'],['grpo','GRPO'],['rlhf','RLHF']]);
  task.addEventListener('change',load);
  const guide=el('div');let revision=0;
  async function load(){const rev=++revision;try{const r=await api('/api/data/guide',{task:f.task,model_path:f.model_path||'',language});if(rev!==revision)return;guide.replaceChildren();cards(guide,{name:r.model,architecture:r.architecture,modality:r.modality});guide.append(el('p',r.flow,'note'));table(guide,['fieldName','fieldType','fieldMeaning'],r.fields.map(x=>[x.name,x.type,x.meaning]));guide.append(el('h3',t('sample')),el('pre',JSON.stringify(r.sample,null,2)),el('h3','JSONL'),el('pre',r.jsonl));for(const note of r.notes)guide.append(el('p',note));const source=el('a',t('source'));source.href=r.source;source.target='_blank';source.rel='noopener noreferrer';guide.append(source);}catch(e){guide.replaceChildren();fail(e);}}
  actions(p,button('dataGuide',load,true));p.append(guide);field(p,'data','path','dataset');
  actions(p,button('validate',async()=>{if(f.task==='rlhf')throw Error(t('chooseRlhfStage'));raw(p,await api('/api/data',f));}),button('train',()=>{if(f.task==='rlhf')throw Error(t('chooseRlhfStage'));S.forms.train.dataset=f.path;S.forms.train.task=f.task;if(f.model_path)S.forms.train.model_path=f.model_path;delete S.plans.train;navigate('train');}));load();
}
function resourcesView(fixedTask=null){
  S.resourceDrafts[S.view]??={...defaults.resources};
  S.forms.resources=S.resourceDrafts[S.view];
  if(fixedTask)S.forms.resources.task=fixedTask;
  const tr=(en,zh)=>language==='zh'?zh:en;
  const h=panel('hardware'),hd=el('details');hd.append(el('summary',tr('Inspect local / SSH hardware (optional)','查看本机／SSH 硬件（可选）')));h.append(hd);
  actions(hd,button('probe',async()=>raw(hd,await api('/api/hardware'))));
  field(hd,'remote','alias','alias');actions(hd,button('remote',async()=>raw(hd,await api('/api/remote',S.forms.remote))));hd.append(txt('remoteNote'));
  const p=panel('planner');plannerConfigImport(p);resourceModelPicker(p);
  const g=fields(p);
  for(const [k,label]of [['parameters_b','totalB'],['seq','seq'],['batch','batch'],['gpu_count','fixed']])field(g,'resources',k,label,'number');
  const taskSelect=field(g,'resources','task','task','text',[['train','train'],['inference','deploy']]);
  if(fixedTask)taskSelect.disabled=true; // Dedicated routes cannot silently become the other planner.
  field(g,'resources','mode','mode','text',[['full','Full'],['lora','LoRA'],['qlora','QLoRA']]);
  field(g,'resources','gpu','gpu','text',Object.entries(S.info.gpu_catalog||{}).map(([k,v])=>[k,v.name]));
  S.forms.resources.deployment_target??='gpu';
  const target=field(g,'resources','deployment_target',tr('Deployment device','部署设备'),'text',[['gpu',tr('Discrete GPU','独立 GPU')],...Object.entries(S.info.edge_catalog||{}).map(([k,v])=>[k,v.name])]);
  const quant=field(g,'resources','bits',tr('Weight precision / quantization estimate','权重精度／量化容量估算'),'text',[[16,'FP16 / BF16 · 16-bit'],[8,'8-bit 权重（格式待核验）'],[4,'4-bit 权重（格式待核验）']]);
  const edgeInputs=el('div',undefined,'grid');p.append(edgeInputs);
  for(const [key,label]of [['edge_memory_gib','板卡总内存 GiB（必填，不按芯片猜测）'],['edge_system_gib','系统／其他应用预留 GiB'],['edge_runtime_gib','运行时／激活额外预算 GiB']]){
    S.forms.resources[key]??=key==='edge_memory_gib'?0:key==='edge_system_gib'?2:1;
    field(edgeInputs,'resources',key,label,'number');
  }
  const quantNote=el('p','权重位宽不是量化格式：AWQ/GPTQ、GGUF、MLX、RKLLM、QNN 产物不能互换；本页不执行转换，KV 仍按 BF16 估算。','note');p.append(quantNote);
  const planning=el('details');planning.open=S.forms.resources.task==='train';planning.append(el('summary',tr('Training objective and measured hardware','训练目标与硬件实测输入')));p.append(planning);const gp=fields(planning);
  field(gp,'resources','training_phase',tr('Training phase','训练阶段'),'text',[['cpt','CPT'],['pretrain','Pretraining'],['sft','SFT'],['dpo','DPO (separate role planner required)'],['grpo','GRPO (separate role planner required)'],['ppo','PPO (separate role planner required)']]);
  field(gp,'resources','curriculum_target_seq',tr('Staged plan target length (not official history)','分阶段计划的目标长度（非官方历史）'),'number');
  field(gp,'resources','node_fabric',tr('Node interconnect','节点内互联'),'text',[['unknown',tr('Unknown (no TP search)','未知（不搜索 TP）')],['pcie','PCIe'],['nvlink','NVLink / NVSwitch']]);
  for(const [k,en,zh]of [['train_tokens_b','Total effective training tokens / B','本次训练有效 token 总量／十亿'],['global_batch_tokens','Target padded tokens / update','目标每步填充后 token 数'],['max_search_gpus','Search GPU ceiling','搜索卡数上限'],['deadline_hours','Deadline hours (0 unknown)','完成期限／小时（0 未设）']])field(gp,'resources',k,tr(en,zh),'number');
  const timing=el('details');timing.append(el('summary',tr('Timing, price and MoE inputs (optional)','时间、价格和 MoE 输入（可选）')));planning.append(timing);const gt=fields(timing);
  for(const [k,en,zh]of [['packing_efficiency','Effective / padded token ratio','有效／填充后 token 比例'],['effective_tflops','Effective compute TFLOPS / GPU','单卡有效算力 TFLOPS'],['intra_gib_s','Effective intra-node GiB/s','节点内有效通信 GiB/s'],['inter_gib_s','Effective inter-node GiB/s / node','节点间有效通信 GiB/s／节点'],['hbm_gib_s','Effective HBM GiB/s','显存有效带宽 GiB/s'],['gpu_hour_price','GPU hourly price (same currency)','单卡小时价格（统一币种）'],['overhead_percent','Evaluation / checkpoint overhead %','评估／检查点时间附加 %'],['expert_count','Routed expert count','路由专家数'],['expert_topk','Experts per token','每 token 专家数'],['expert_parameter_fraction','Routed-expert parameter fraction','路由专家参数占比（0–1）']])field(gt,'resources',k,tr(en,zh),'number');
  planning.append(el('p',tr('0 for unknown throughput/bandwidth/price. Do not enter peak marketing TFLOPS as effective compute. Assumptions are not measurements.','吞吐、带宽、价格未知填 0。不要把宣传峰值 TFLOPS 当成有效算力。手填假设不等于实测。')));
  const taskVisibility=()=>{const training=S.forms.resources.task==='train',edge=!training&&target.value!=='gpu';planning.hidden=!training;$('resources_mode').parentElement.hidden=!training;target.parentElement.hidden=training;quant.parentElement.hidden=training;quantNote.hidden=training;edgeInputs.hidden=!edge;$('resources_gpu').parentElement.hidden=edge;$('resources_gpu_count').parentElement.hidden=edge;
    for(const key of ['gpus_per_node','extra_gpus','activation_gib','workspace_gib','kv_gib','gather_gib','adapter_million','runtime_gib','communication_gib','cuda_graph_gib','other_models_gib','rollout_gib']){const input=$('resources_'+key);if(input)input.parentElement.hidden=edge;}
  };
  target.addEventListener('change',taskVisibility);
  $('resources_task').addEventListener('change',taskVisibility);taskVisibility();
  // Frequently adjusted capacity assumptions are not hidden behind a fold.
  // Keep long derivations / raw evidence collapsible, not essential inputs.
  const capacity=el('section',undefined,'control-section');capacity.append(el('h3',tr('Context and capacity budgets','上下文与容量预算')));p.append(capacity);const limits=fields(capacity);
  const adv=el('details');adv.append(el('summary',tr('Model structure and detailed memory overrides','模型结构与详细显存覆盖值')));p.append(adv);const ga=fields(adv);
  field(ga,'resources','model_path','path').addEventListener('input',()=>{S.forms.resources.model_key='';});
  field(limits,'resources','context_limit',tr('Verified context cap (blank = model config; 0 unknown)','模型/后端上下文上限（留空读配置，0 未知）'),'number');
  field(limits,'resources','kv_block_tokens',tr('KV page tokens (blank = 16)','KV 页 token 数（留空默认 16）'),'number');
  field(limits,'resources','request_mix','混合长度对照（总tokens:请求数，如 2048:8,8192:2）','text');
  limits.append(el('p','上方显存图按“驻留序列数 × 每请求上下文”计算，默认1路。每请求上下文包括输入＋最大输出。混合长度是同卡数/同TP与PP的独立对照，不改写上方等长方案。'));
  for(const [k,label]of [['gpus_per_node','nodeSize'],['extra_gpus','extra'],['reserve_percent','reserve']])field(limits,'resources',k,label,'number');
  for(const k of ['layers','hidden','heads','kv_heads','ffn','vocab','activation_gib','workspace_gib','kv_gib','gather_gib','adapter_million','runtime_gib','communication_gib','cuda_graph_gib','other_models_gib','rollout_gib'])field(ga,'resources',k,({kv_heads:'kvHeads',activation_gib:'activation',workspace_gib:'workspace'})[k]||k,'number');
  taskVisibility();
  p.classList.add('planner-dashboard');
  const inputs=el('div',undefined,'planner-inputs');
  while(p.firstChild)inputs.append(p.firstChild);p.append(inputs);
  inputs.append(hd);h.remove();
  const output=el('div',undefined,'planner-output'),status=el('p',undefined,'planner-status');
  status.setAttribute('role','status');status.setAttribute('aria-live','polite');
  output.append(el('h2',tr('PLAN / LIVE ESTIMATES','方案总览 · 实时估算')),status);p.append(output);
  const result=el('div');output.append(result);let planRevision=0,timer=null;
  const form=S.forms.resources;
  function schedule(){clearTimeout(timer);timer=setTimeout(()=>{if(p.isConnected)calculate();},400);}
  const invalidate=event=>{if(result.contains(event.target))return;if(!event.target.id?.startsWith('resources_')&&!event.target.hasAttribute('data-resource-model-picker'))return;planRevision++;result.classList.add('stale');status.textContent=tr('Inputs changed · updating…','输入已变更 · 等待更新…');schedule();};
  p.addEventListener('input',invalidate);p.addEventListener('change',invalidate);
  async function calculate(){
    clearTimeout(timer);const currentRevision=++planRevision;
    output.setAttribute('aria-busy','false');
    if(p.querySelector('[data-resource-model-picker][data-loading="true"]')){status.textContent=tr('Reading model metadata…','正在读取模型元数据…');return;}
    if((!(Number(form.parameters_b)>0)&&!(form.task==='inference'&&form.model_key))||!(Number(form.seq)>0)||!(Number(form.batch)>0)){status.textContent=tr('Enter positive parameters, context and concurrency.','请填写有效的正数参数量、上下文和并发。');result.classList.add('stale');return;}
    status.textContent=tr('Calculating · previous results are stale','正在计算 · 旧结果暂不可用');output.setAttribute('aria-busy','true');result.classList.add('stale');
    try{
    const r=await api('/api/resources',Object.fromEntries(Object.entries(form).filter(([k,v])=>(v!==''||k==='parameters_b')&&!(form.model_path&&['layers','hidden','heads','kv_heads','ffn','vocab'].includes(k)&&v===0))));
    if(currentRevision!==planRevision||!p.isConnected)return;
    if(r.route_only){
      result.replaceChildren();renderDeploymentRoute(result,r.deployment_route);result.append(el('p',r.reason,'note'));
      result.classList.remove('stale');status.textContent=r.deployment_route.label;return;
    }
    if(r.edge_plan){
      renderEdgePlan(result,r.edge_plan);renderDeploymentRecipe(result,r.deployment_recipe);result.classList.remove('stale');status.textContent=tr('Updated · changes recalculate automatically','已更新 · 修改参数自动重算');return;
    }
    renderReport(r);
    const detail=el('details');detail.append(el('summary',tr('Expand formulas, topology and service planning','展开原理、逐卡拓扑与服务规划')));
    while(result.firstChild)detail.append(result.firstChild);
    const partial=r.planning_coverage?.status==='partial';
    const hasRecipe=r.deployment_route?.status==='upstream_recipe';
    if(r.deployment_route)renderDeploymentRoute(result,r.deployment_route);
    const unsupported=r.deployment_support?.status==='adapter_required'||partial;
    const env=unsupported?null:r.deployment_envelope,s=unsupported?{}:r.selected||{};
    if(partial){
      const coverage=r.planning_coverage;
      cards(result,{[tr('Weight payload only · GiB','仅权重有效载荷 · GiB')]:fmt(coverage.weight_payload_gib),[tr('Assumed weight bits','假设权重位宽')]:coverage.weight_bits,[tr('Plan coverage','方案覆盖')]:hasRecipe?'已有上游配方 · 当前容量待测':tr('Partial · not runnable','部分估算 · 非运行方案')});
      result.append(el('p',tr('Payload = parameters × bits / 8. Excludes quantization metadata, KV, activations, gradients, optimizer and runtime. Nominal parameters may differ from checkpoint size; quantization availability is not verified.','权重有效载荷 = 参数量 × 位宽 / 8；不含量化元数据、KV、激活、梯度、优化器与运行时。标称参数量可能不同于检查点大小；不代表已有对应量化权重或运行后端。'),'note'));
      if(hasRecipe)result.append(el('p','当前输入的逐卡峰值、上下文并发仍待架构预算验证；请使用上面的独立配方，不把它与下方权重载荷混算。','note'));
      else for(const reason of coverage.blockers)result.append(el('p',reason,'danger'));
    }else cards(result,form.task==='train'?{[tr('Conditional minimum GPUs','条件最少 GPU')]:s.conditional_min_gpus??'—',[tr('Allocated GPUs','分配 GPU')]:s.allocated_gpus??'—',[tr('Peak GiB / GPU','逐卡峰值 GiB')]:fmt(s.topology?.peak_gib),[tr('Host RAM guidance GiB','主存建议 GiB')]:fmt(s.host_ram_gib_guidance)}:{[tr('Conditional minimum GPUs','条件最少 GPU')]:env?.minimum_gpus_target??s.conditional_min_gpus??'—',[tr('Resident sequences · memory bound','显存驻留并发上限')]:env?.memory_concurrency??'—',[tr('Context at target concurrency','给定并发上下文上限')]:env?.context_at_target_concurrency??'—',[tr('Peak GiB / GPU','逐卡峰值 GiB')]:fmt(env?.target?.peak_gib??s.topology?.peak_gib)});
    result.append(el('p',tr('Estimates under current assumptions · not measured throughput or an SLO guarantee.','基于当前输入的条件估算 · 非实测吞吐，也不是 SLO 保证。'),'note'));
    if(r.deployment_support?.model_type==='gemma3_text'&&r.deployment_support.analytic_supported)result.append(el('p',r.deployment_support.reason,'note'));
    if(env?.hybrid_cache){
      const h=env.hybrid_cache,box=el('section',undefined,'hybrid-cache control-section');result.append(box);
      const gdn=h.family==='qwen_gdn';
      box.append(el('h3',gdn?'Qwen混合 · Gated DeltaNet + 全注意力规划':'Kimi Linear · KDA + MLA + MoE 多卡规划'));
      cards(box,{'分配 GPU':env.allocated_gpus,'张量并行 TP':env.comparison_layout.tp,[gdn?'全注意力KV GiB/卡':'MLA缓存 GiB/卡']:fmt(h.attention_gib),[gdn?'GDN状态 GiB/卡':'KDA状态 GiB/卡']:fmt(h.state_gib)});
      box.append(el('p',gdn?`全注意力层KV随上下文增长、按KV头分片；GDN循环状态随驻留请求数增长、按头分片。仅文本请求，不包含图片/视频编码峰值。分块prefill上限 ${h.prefill_chunk_tokens} tokens。`:`MLA随每路上下文增长，压缩缓存在TP卡间复制；KDA随驻留请求数增长、按头分片。全部MoE专家权重驻留，不以激活3B代替总参数。分块prefill上限 ${h.prefill_chunk_tokens} tokens。`));
      for(const assumption of env.assumptions)box.append(el('p',assumption));
      const link=el('a','官方部署入口（外部后端，非原生核）');link.href=gdn?'https://huggingface.co/Qwen/Qwen3.5-9B':'https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base';link.target='_blank';link.rel='noopener noreferrer';box.append(link);
      box.append(el('p','部署顺序：固定模型revision与兼容vLLM/SGLang版本 → 配置上述TP/上下文/驻留并发与分块prefill → 校验各卡加载及prefill/decode峰值 → 验证输出质量和延迟后开放服务。PP/EP、量化产物与多节点路线未在此适配器认证。'));
    }
    if(env?.cache_adapter==='layered_full_history'){
      const box=el('section',undefined,'control-section');result.append(box);
      box.append(el('h3','逐层混合注意力容量规划'));
      box.append(el('p','逐层累加 KV 头数 × (K维度 + V维度) × 2字节；滑窗按全历史保守预算。PP取缓存最大的阶段，不用平均值。仅文本，外部后端待验证。'));
      cards(box,{'层数':env.layered_cache.layers.length,'TP':env.layered_cache.tp,'PP':env.layered_cache.pp,'每token字节/最忙卡':env.kv_bytes_per_token_per_rank});
    }
    if(env?.cache_adapter==='mla'){
      const box=el('section',undefined,'mla-cache control-section');box.append(el('h3','MLA / DSA · 多卡容量规划'));
      for(const note of env.assumptions)box.append(el('p',note));result.append(box);
    }
    for(const warning of env?.blockers||[])result.append(el('p',warning,'danger'));
    if(r.deployment_support?.status==='adapter_required'&&!hasRecipe)result.append(el('p',r.deployment_support.reason,'danger'));
    // Keep the existing colored per-GPU ledger visible; only derivations fold.
    const placement=detail.querySelector('.placement');if(placement&&!partial){result.append(el('p',`当前彩色图：${form.batch} 路同时驻留 × 每路 ${form.seq} tokens（输入＋最大输出）；共享权重只计一次，KV逐请求计。`));result.append(placement);}
    renderDeploymentRecipe(result,r.deployment_recipe);
    if(env?.concurrency_memory){
      const memory=el('section',undefined,'concurrency-memory control-section');result.append(memory);
      memory.append(el('h3',tr('Concurrency × context · VRAM per GPU','并发 × 上下文 · 每卡显存')));
      const declared=env.declared_context;
      if(declared)memory.append(el('p',tr('Model-declared context: ','模型声明上下文：')+declared.context_tokens.toLocaleString()+' tokens · '+declared.concurrency+tr(' residents · estimated peak ',' 路驻留并发 · 估算峰值 ')+fmt(declared.peak_gib)+' GiB / GPU · '+(declared.fits_memory?tr('Within budget','预算内'):tr('Exceeds budget','超出预算'))));
      memory.append(el('p',tr('Current planning length: ','当前规划长度：')+env.target.context_tokens.toLocaleString()+' tokens · TP '+env.comparison_layout.tp+' / PP '+env.comparison_layout.pp));
      memory.append(el('p',`当前 ${env.target.concurrency} 路；固定 ${env.allocated_gpus} 张 GPU、每路 ${env.target.context_tokens} tokens 的显存容量上限：${env.memory_concurrency} 路。${env.target.within_context_limit===false?'但当前长度超过模型上限，不可准入。':env.target.within_context_limit==null?'模型上下文上限未知，暂不可批准准入。':''} 满足响应延迟的并发需实测，不能由显存推断。`));
      const mix=env.mixed_requests;
      if(mix){const section=el('section',undefined,'mixed-request-memory');memory.append(section);section.append(el('h3','长短请求混合 · 同一GPU配置对照'));
        table(section,['每请求输入＋最大输出','请求数','逐请求页对齐tokens'],mix.groups.map(g=>[g.context_tokens,g.concurrency,g.padded_tokens]));
        cards(section,{'总驻留请求':mix.concurrency,'预留KV tokens':mix.reserved_tokens,'KV GiB':fmt(mix.kv_gib),'总峰值 GiB/卡':fmt(mix.peak_gib),'按此比例整组扩容的容量上限':mix.maximum_concurrency_same_mix});
        const bar=el('div',undefined,'gpu-memory');bar.setAttribute('aria-label','混合请求每卡显存预算');section.append(bar);
        for(const [label,amount,color]of [['固定权重/运行时',mix.fixed_gib,'#3976de'],['混合KV',mix.kv_gib,'#10a896'],['激活代理',mix.activation_gib,'#e49a30']]){const span=el('span');span.style.width=100*amount/Math.max(mix.usable_gib,mix.peak_gib)+'%';span.style.background=color;span.title=label+' '+fmt(amount)+' GiB';bar.append(span);}
        section.append(el('p',`蓝色固定权重/运行时 ${fmt(mix.fixed_gib)} GiB；绿色KV ${fmt(mix.kv_gib)} GiB；橙色激活代理 ${fmt(mix.activation_gib)} GiB。可用预算 ${fmt(mix.usable_gib)} GiB/卡（已扣安全预留）；图条按总需求与可用预算较大者缩放。`));
        section.append(el('p',mix.formula),el('p',`${mix.fits_memory?'显存预算内':'超过显存预算'}；${mix.within_context_limit===true?'各请求长度在模型上限内':'长度超过模型上限或上限未知，不准入'}。整组上限保持上述比例，非任意请求组合的绝对最大值${mix.search_ceiling_reached?'，已达搜索封顶':''}。激活按总token代理估算，实际调度/prefill峰值需验证。`));}
      table(memory,[tr('Concurrency','并发'),tr('Fixed GiB','固定 GiB'),tr('KV GiB','KV GiB'),tr('Activation proxy GiB','激活代理 GiB'),tr('Peak GiB/GPU','每卡峰值 GiB'),tr('Extra vs 1','比单路增加 GiB'),tr('Memory check','显存检查')],env.concurrency_memory.map(x=>[x.concurrency,fmt(x.fixed_gib),fmt(x.kv_gib),fmt(x.activation_gib),fmt(x.peak_gib),fmt(x.extra_vs_one_gib),x.fits_memory?tr('Within budget','预算内'):tr('Over budget','超预算')]));
      memory.append(el('p',tr('Fixed GPU count and TP/PP. Weights are shared; KV is per resident request, reserving the full input + output length. Memory occupancy is not compute utilization or throughput. Activation is a proxy; context limits still apply.','各行固定卡数和TP/PP：权重共享，KV按驻留请求分别占用，每路预留完整输入＋输出长度。显存占用不是计算利用率或吞吐；激活为代理估算，仍须遵守上下文限制。'),'note'));
    }
    const delta=env?.context_memory_delta;
    if(delta){
      const context=el('section',undefined,'context-memory-delta');result.append(context);
      context.append(el('h3',tr('Context length alone · extra VRAM / GPU','仅上下文长度变化 · 每卡额外显存')));
      context.append(el('p',`${delta.baseline_tokens} → ${delta.target_tokens} tokens · ${tr('Resident sequences','驻留并发')} ${delta.concurrency} · TP ${delta.tp} / PP ${delta.pp} · KV ${delta.kv_precision}`));
      const signed=n=>(n>=0?'+':'')+Number(n).toFixed(3)+' GiB';
      cards(context,{[tr('KV cache delta','KV Cache 增量')]:signed(delta.kv_delta_gib),[tr('Prefill activation proxy delta','Prefill 激活代理增量')]:signed(delta.activation_delta_gib),[tr('Estimated peak delta','估算峰值增量')]:signed(delta.peak_delta_gib)});
      context.append(el('p',`KV: ${delta.baseline_kv_gib.toFixed(3)} → ${delta.target_kv_gib.toFixed(3)} GiB`),el('code',delta.formula,'memory-formula'));
      context.append(el('p',tr('Same weights, concurrency, precision, TP/PP and budgets; only context changes. Negative means savings. Context includes prompt + generated tokens. KV is page-rounded; activation is an approximation, not measured prefill peak. Baseline is a comparison, not an approved configuration.','固定权重、并发、精度、TP/PP 与其他预算，只改变上下文。负数表示节省；上下文含输入与生成 tokens。KV 按页取整，激活是代理估算，不是实测 prefill 峰值。2K 仅为比较基准，不代表已批准配置。'),'note'));
    }
    result.append(detail);result.classList.remove('stale');status.textContent=hasRecipe?'已有上游部署配方 · 请核对硬件/精度；当前输入容量尚未认证':partial?tr('Partial estimate · complete planning unavailable; see requirements below','部分估算 · 暂不能生成完整方案，请查看下方缺失条件'):tr('Updated · changes recalculate automatically','已更新 · 修改参数自动重算');
    }catch(e){if(currentRevision===planRevision&&p.isConnected){status.textContent=tr('Cannot calculate: ','无法计算：')+e.message;result.classList.add('stale');}}
    finally{if(currentRevision===planRevision)output.setAttribute('aria-busy','false');}
  }
  actions(inputs,button('calculate',calculate,true));schedule();
  function renderReport(r){
    result.replaceChildren();let auditParent=result;knowledgeEvidenceView(result,r.training_design);lengthCurriculumView(result,r);
    if(r.training_design){trainingDesignView(result,r);auditParent=el('details');auditParent.append(el('summary',tr('Legacy ZeRO-only audit and official model reference (not the joint recommendation)','旧版仅 ZeRO 容量审计及官方模型资料（不是联合推荐）')));result.append(auditParent);}
    if(r.deployment_support?.status!=='adapter_required'){resourcePlanBasis(auditParent,r);resourcePlacement(auditParent,r);}
    if(S.forms.resources.task==='inference')servingPlannerView(result,r);
    const more=el('details');more.append(el('summary',tr('Other GPUs, host resources and audit evidence','其他 GPU、主机资源与审计依据')));auditParent.append(more);
    if(r.deployment_support?.status==='adapter_required'){
      more.append(el('p','以下仅保留未适配通用公式的诊断记录，卡数、主存、磁盘与逐卡账本均不得作为租卡或启动依据。','danger'));
      raw(more,r);return;
    }
    table(more,['gpu','zeroStage','minimum','allocated','peak'],r.candidates.map(c=>[c.name,c.recommended_zero_stage==null?'—':'ZeRO-'+c.recommended_zero_stage,c.conditional_min_gpus,c.allocated_gpus,fmt(c.topology?.peak_gib)]));
    cards(more,{hostRam:r.selected.host_ram_gib_guidance,diskSpace:r.selected.storage_gib_guidance});
    table(more,['fieldName','basis'],r.derivation.map(x=>[x.step,x.calculation]));
    for(const note of r.detailed_budget.notes)more.append(el('p',note));
    raw(more,r);
  }
}
function renderDeploymentRoute(parent,r){
  const box=el('section',undefined,'deployment-routes control-section');parent.append(box);
  box.append(el('h3',r.label));
  if(r.recipes.length){
    box.append(el('p','以下是独立的上游方案，不继承左侧统一位宽、上下文或并发；参数变化不会把配方变成已验证方案。','note'));
    const show=(parent,rows)=>{for(const recipe of rows){
      const card=el('section',undefined,'control-section');parent.append(card);
      card.append(el('h4',`${recipe.gpus} × ${recipe.hardware.toUpperCase()} · ${recipe.nodes} 节点 · ${recipe.strategy}`),
        el('p',`${recipe.repo} · ${recipe.precision} · ${recipe.matches_selected_hardware?'匹配所选硬件':'替代硬件路线'} · ${recipe.upstream_verified?'上游标记已验证':'上游未验证'} / 本平台未实测`));
      card.append(el('p',recipe.runtime));
      const link=el('a','固定版本上游配方与完整依赖');link.href=recipe.source;link.target='_blank';link.rel='noopener noreferrer';card.append(link);
      const args=el('details');args.append(el('summary','查看上游参数（仅展示，不执行）'),el('pre',[...recipe.env,...recipe.flags].join('\n')));card.append(args);
    }};
    show(box,r.recipes.slice(0,4));
    if(r.recipes.length>4){const rest=el('details');rest.append(el('summary',`其余 ${r.recipes.length-4} 个硬件/策略配方`));show(rest,r.recipes.slice(4));box.append(rest);}
    for(const note of r.warnings)box.append(el('p',note,'note'));
  }
  const steps=el('ol');for(const step of r.steps)steps.append(el('li',step));box.append(steps);
}
function renderDeploymentRecipe(parent,r){
  if(!r)return;
  const section=el('section',undefined,'deployment-recipe');section.append(el('h3','当前模型部署路线 · '+r.backend.name));
  section.append(el('p','外部工具链候选，未实机认证；容量以当前主规划图为准，以下步骤不自动执行。'));
  const list=el('ol');r.steps.forEach(s=>list.append(el('li',s)));section.append(list);
  parent.append(section);
}
function renderEdgePlan(parent,e){
  parent.replaceChildren();parent.append(el('h3',e.device.name+' · 单设备内存规划'));
  parent.append(el('p',`当前图：${e.concurrency} 路驻留请求 × 每路 ${e.context_tokens} tokens（输入＋最大输出）；权重共享。显存容量上限：${e.memory_concurrency??'未知'} 路，仅固定运行时预算的容量代理，不是延迟承诺或准入批准。`));
  if(e.request_mix_note)parent.append(el('p',e.request_mix_note,'danger'));
  cards(parent,{'权重 GiB':fmt(e.weights_gib),'KV GiB（BF16）':e.kv_gib==null?'未知':fmt(e.kv_gib),'含额外预算 GiB':e.estimated_gib==null?'未知':fmt(e.estimated_gib),'容量判断':e.fits_budget==null?'待补参数':e.fits_budget?'预算内 · 未认证':'超预算'});
  const bar=el('div',undefined,'gpu-memory');bar.setAttribute('aria-label','端侧内存估算分布');parent.append(bar);
  const rows=[['权重',e.weights_gib,'#3976de'],['量化元数据',e.quant_metadata_gib,'#9b65cb'],['KV',e.kv_gib,'#10a896'],['运行时／激活',e.runtime_gib,'#e49a30'],['系统',e.system_gib,'#647891'],['安全预留',e.reserve_gib,'#cbd5e1']];
  if(e.memory_gib)for(const [label,value,color]of rows){if(value==null)continue;const s=el('span');s.style.width=(value/e.memory_gib*100)+'%';s.style.background=color;s.title=label+': '+fmt(value)+' GiB';bar.append(s);}
  table(parent,['内存分项','GiB'],rows.map(([k,v])=>[k,v==null?'未知':fmt(v)]));
  parent.append(el('p',`上下文 ${e.baseline_tokens} → ${e.context_tokens} tokens · 并发 ${e.concurrency} · 仅 KV 增量 ${e.context_delta_gib==null?'未知':e.context_delta_gib.toFixed(3)+' GiB'}（运行时预算固定；未建模 prefill 额外峰值）`));
  parent.append(el('h3','转换与运行时路线（外部生态，非 TessWeave 原生能力）'),el('p',e.device.runtime));
  const link=el('a','官方参考 · 核对 '+e.checked);link.href=e.device.source;link.target='_blank';link.rel='noopener noreferrer';parent.append(link);
  for(const note of e.notes)parent.append(el('p',note,'note'));
}
function servingPlannerView(parent,capacity){
  const tr=(en,zh)=>language==='zh'?zh:en,s=capacity.selected;
  if(capacity.deployment_support){
    const support=capacity.deployment_support,view=panel(null,parent);
    view.append(el('h2','部署证据与架构支持'),el('p',support.reason,'note'));
    for(const recipe of support.recipes){
      view.append(el('h3',recipe.repo),el('p',recipe.deployment),el('p',recipe.caveat));
      const link=el('a','上游配方 · '+recipe.checked+' · '+recipe.evidence);link.href=recipe.source;link.target='_blank';link.rel='noopener noreferrer';view.append(link);
      view.append(el('p','仅精确仓库名匹配；未核验本地权重或固定 revision，非本平台实测，不自动执行。'));
    }
    if(!support.recipes.length)view.append(el('p','没有精确匹配的已登记配方；不按目录名或同系列模型继承配置。'));
    for(const item of support.required_evidence)view.append(el('p','待核验：'+item));
  }
  const envelope=capacity.deployment_envelope;
  if(envelope){
    const view=panel(null,parent);view.append(el('h2',tr('Deployment capacity envelope','部署容量边界：GPU × 上下文 × 并发')));
    cards(view,{[tr('Minimum GPUs · one sequence','单序列条件最小 GPU')]:envelope.minimum_gpus_one_request??'—',[tr('Minimum GPUs · target concurrency','目标并发条件最小 GPU')]:envelope.minimum_gpus_target??'—',[tr('Resident sequences · memory bound','显存允许驻留序列上限')]:envelope.memory_concurrency??'—',[tr('Context at requested concurrency','给定并发的上下文上限')]:envelope.context_at_target_concurrency??'未知',[tr('SLO concurrency','满足延迟目标的并发')]:tr('Requires load test','需压测，不能由显存推断')});
    view.append(el('p',envelope.minimum_basis,'note'));
    for(const note of envelope.blockers)view.append(el('p',note,'danger'));
    if(envelope.target)cards(view,{[tr('Estimated peak / rank GiB','估算逐卡峰值 GiB')]:fmt(envelope.target.peak_gib),[tr('Budget remaining GiB','预算剩余 GiB')]:fmt(envelope.target.headroom_gib)});
    table(view,[tr('Context tokens','上下文 tokens'),tr('Memory ceiling','显存并发上限'),tr('Context limit check','上下文限制检查')],envelope.curve.map(x=>[x.context_tokens,x.memory_concurrency+(x.search_ceiling_reached?'（搜索封顶）':''),x.within_context_limit===null?'未知':x.within_context_limit?'范围内（非实测）':'超上限，不准入']));
    const detail=el('details');detail.append(el('summary',tr('Formulas and production checklist','计算依据与生产服务检查清单')));view.append(detail);
    table(detail,[tr('Item','分项'),'GiB',tr('Formula','公式')],(envelope.ledger||[]).map(x=>[x.item,fmt(x.gib),x.formula]));
    for(const note of [...envelope.assumptions,...envelope.operations])detail.append(el('p',note));
    detail.append(el('p','vLLM / SGLang 提供推理服务及相关生态组件，不等于完整托管平台。TessWeave 当前输出规划与审计，不宣称已实现生产自动扩缩容、故障迁移或 SLA。云厂商提供硬件/网络/存储；自建集群也可运行，平台仍需网关、身份鉴权、健康检查、监控与编排。'));
    raw(detail,envelope);
  }
  const box=panel(null,parent);box.classList.add('serving-planner');
  box.append(el('h2',tr('Traffic → replicas → failure reserve','业务负载 → 服务副本 → 故障余量')),
    el('p',tr('First qualify one replica above. Then enter joint, open-loop load-test capacity for this exact model, precision, topology and length distribution. This is a scenario, not an SLO guarantee.','先验证上方单副本容量，再填入同模型、精度、拓扑和长度分布的开环压测结果。这是情景规划，不是 SLA 保证。'),'note'));
  if(!s.feasible_under_assumptions){box.append(el('p',tr('Resolve single-replica memory constraints first.','请先解决单副本显存约束。'),'danger'));return;}
  if(envelope?.blockers.length){box.append(el('p',tr('Resolve deployment envelope blockers before sizing replicas.','请先解决部署容量边界的阻断项，再计算服务副本。'),'danger'));return;}
  // Calibration must never silently migrate to another model/topology/context.
  const context=JSON.stringify([capacity.model_identity,capacity.parameters,s.name,s.topology,S.forms.resources.model_key,S.forms.resources.model_path,S.forms.resources.seq,S.forms.resources.batch,S.forms.resources.bits]);
  if(S.servingContext!==context){S.forms.serving={daily_requests:100000,peak_factor:10,peak_rps:0,calls_per_request:1,mean_input_tokens:2000,mean_output_tokens:500,max_output_tokens:2048,headroom_percent:20,gpus_per_replica:s.allocated_gpus,replicas_per_domain:1,failed_domains:1,first_text_slo_s:2,total_slo_s:30,error_budget_percent:1,replica_rps:0,observed_first_text_p95_s:0,observed_total_p95_s:0,observed_error_percent:0,gpu_hour_price:0};S.servingContext=context;}
  const f=S.forms.serving;
  box.append(el('p',`${s.name} · ${s.allocated_gpus} GPU / replica · TP ${s.topology.tp} / PP ${s.topology.pp}`));
  const traffic=fields(box);
  for(const [k,en,zh]of [['daily_requests','Business requests / day','每日业务请求'],['peak_factor','Peak / mean ratio','峰均比'],['peak_rps','Explicit peak business RPS (0 = ratio)','显式峰值业务 RPS（0 用峰均比）'],['calls_per_request','Model calls / business request','每业务请求调用本模型次数'],['mean_input_tokens','Mean input tokens','平均输入 token'],['mean_output_tokens','Mean output tokens','平均输出 token'],['max_output_tokens','Output admission limit','输出准入上限'],['first_text_slo_s','p95 first-text budget / s','p95 首段文本预算／秒'],['total_slo_s','p95 total budget / s','p95 完整响应预算／秒']])field(traffic,'serving',k,tr(en,zh),'number');
  const calibration=el('details');calibration.open=true;calibration.append(el('summary',tr('Calibration and reliability assumptions','压测校准与可靠性假设')));box.append(calibration);const inputs=fields(calibration);
  for(const [k,en,zh]of [['replica_rps','Tested joint RPS / replica (0 unknown)','单副本联合实测 RPS（0 未知）'],['observed_first_text_p95_s','Observed p95 first text / s','实测 p95 首段文本／秒'],['observed_total_p95_s','Observed p95 total / s','实测 p95 完整响应／秒'],['observed_error_percent','Observed errors %','实测失败率 %'],['error_budget_percent','Allowed errors %','允许失败率 %'],['headroom_percent','Capacity headroom %','容量余量 %'],['replicas_per_domain','Max replicas / failure domain','每故障域最多副本数'],['failed_domains','Simultaneously failed domains','同时失效故障域数'],['gpu_hour_price','GPU hourly price (0 unknown)','单卡小时价格（0 未知）']])field(inputs,'serving',k,tr(en,zh),'number');
  calibration.append(el('p',tr('A domain means a node or disjoint node group, not one GPU. Do not multiply standalone prefill and decode peaks. Manual calibration is not verified benchmark evidence.','故障域是节点或不重叠节点组，不是单张 GPU。不能将单独测得的 prefill/decode 峰值同时计入。手填校准值不等于已验证压测证据。')));
  const out=el('div');let revision=0;
  for(const event of ['input','change'])box.addEventListener(event,()=>{revision++;out.replaceChildren();});
  actions(box,button(tr('Calculate serving scenario','计算服务副本方案'),async()=>{
    const before=revision;const report=await api('/api/serving/plan',{...f});if(before!==revision)return;out.replaceChildren();
    cards(out,{[tr('Peak model RPS','峰值模型 RPS')]:fmt(report.peak_model_rps),[tr('Input tokens/s demand','输入 token/s 需求')]:fmt(report.peak_input_tokens_s),[tr('Output tokens/s demand','输出 token/s 需求')]:fmt(report.peak_output_tokens_s)});
    for(const b of report.blockers)out.append(el('p',b,'danger'));
    if(report.normal){table(out,[tr('Scenario','情景'),tr('Replicas','副本数'),'GPU',tr('Hourly GPU cost','每小时 GPU 费用')],[[tr('Normal peak','正常高峰'),report.normal.replicas,report.normal.gpus,report.normal.hourly_gpu_cost??'—'],[tr('Failure reserve included','包含故障余量'),report.resilient.replicas,report.resilient.gpus,report.resilient.hourly_gpu_cost??'—']]);
      const nodes=el('div',undefined,'placement-nodes');out.append(nodes);for(let i=0;i<Math.min(12,report.resilient.domains);i++){const n=el('div',undefined,'placement-node');n.append(el('h4',tr('Failure domain ','故障域 ')+i),el('p',`${f.replicas_per_domain} replicas × ${f.gpus_per_replica} GPU`));nodes.append(n);}if(report.resilient.domains>12)out.append(el('p',tr('Showing first 12 domains. Full count: ','仅展示前 12 个故障域，总数：')+report.resilient.domains));}
    table(out,[tr('Decision','决策'),tr('Formula','公式'),tr('Result','结果')],report.decision_trace.map(x=>[x.step,x.formula,fmt(x.value)]));
    const evidence=el('details');evidence.append(el('summary',tr('Assumptions and JSON evidence','假设与 JSON 证据')));out.append(evidence);for(const c of report.caveats)evidence.append(el('p',c));raw(evidence,{...report,capacity_context:JSON.parse(context)});
  },true,true));box.append(out);
}
function resourceModelPicker(parent){
  const tr=(en,zh)=>language==='zh'?zh:en;
  const box=el('div'),label=el('label',t('resourceModel')),select=el('select'),note=el('p'),search=el('input'),source=el('select');
  box.dataset.resourceModelPicker='';
  search.type='search';search.placeholder=tr('Search other models by name or organization','搜索其他模型名称或发布组织');search.setAttribute('aria-label',search.placeholder);
  for(const id of ['modelscope','hf-mirror','huggingface']){const o=el('option',id);o.value=id;source.append(o);}source.setAttribute('aria-label',tr('Search source','检索源'));
  select.setAttribute('aria-label',t('resourceModel'));select.hidden=true;select.style.display='none';label.append(select);box.append(label,note);parent.append(box);
  const open=el('button',tr('Choose model · browse and search','选择模型 · 分类浏览与搜索'));open.type='button';open.dataset.openModelChooser='';label.append(open);
  const dialog=el('dialog',undefined,'model-chooser'),heading=el('h2',tr('Model library','选择模型')),close=el('button',tr('Close','关闭')),toolbar=el('div',undefined,'model-chooser-filters'),results=el('div',undefined,'model-chooser-results'),count=el('p');
  heading.id='model-chooser-title';dialog.setAttribute('aria-labelledby',heading.id);close.type='button';close.onclick=()=>dialog.close();dialog.append(heading,close,toolbar,count,results);box.append(dialog);
  const category=el('select'),vendor=el('select'),size=el('select');
  for(const [value,name]of [['','所有类型'],['text','文本 / 推理 / 代码'],['multimodal','多模态 / 视觉语言'],['image','图像生成'],['video','视频生成'],['audio','语音 / 音频'],['embedding','向量 / 重排'],['unknown','类型待确认']]){const o=el('option',name);o.value=value;category.append(o);}
  category.setAttribute('aria-label','模型类型');vendor.setAttribute('aria-label','模型厂商');size.setAttribute('aria-label','参数范围');
  for(const [value,name]of [['','所有参数规模'],['50','低于 50B（已知参数）']]){const o=el('option',name);o.value=value;size.append(o);}
  toolbar.append(search,category,vendor,size,source);
  const manual=el('button','手动输入参数');manual.type='button';manual.onclick=()=>{select.value='';open.textContent='选择模型 · 分类浏览与搜索';dialog.close();choose();};toolbar.append(manual);
  const modality=r=>{const tag=r.pipeline_tag||r.modality||'';if(/image-text|vision-language|any-to-any/.test(tag)||r.config?.vision_config)return 'multimodal';if(/text-to-video|image-to-video/.test(tag))return 'video';if(/text-to-image|image-to-image/.test(tag))return 'image';if(/audio|speech/.test(tag))return 'audio';if(/feature-extraction|sentence-similarity|ranking/.test(tag))return 'embedding';if(/text-generation|text2text|^text$/.test(tag)||['qwen2','qwen3','llama','mistral','phi3'].includes(r.config?.model_type))return 'text';return 'unknown';};
  function drawCards(){const q=search.value.trim().toLowerCase();results.replaceChildren();const rows=records.filter(r=>{const b=r.parameter_estimate/1e9||r.nominal_parameters_b;return (!q||JSON.stringify([r.repo,r.name,r.model_card_title]).toLowerCase().includes(q))&&(!category.value||modality(r)===category.value)&&(!vendor.value||(r.repo||'').split('/')[0]===vendor.value)&&(!size.value||(b>0&&b<50));});
    count.textContent=`${rows.length} 个匹配 / 本地目录 ${records.length} 个。目录不是全站全集；可联网检索。模型可选 ≠ 规划/运行时已适配，开放权重不等于无许可限制。`;
    for(const r of rows){const card=el('button',undefined,'model-choice');card.type='button';card.dataset.modelKey=r.key;const b=r.parameter_estimate/1e9||r.nominal_parameters_b;card.append(el('strong',r.model_card_title||r.repo||r.name),el('span',`${(r.repo||'Local').split('/')[0]} · ${modality(r)} · ${b?b.toLocaleString(undefined,{maximumFractionDigits:2})+'B':'参数未知'}`),el('small',`${r.kind==='local'?'本地权重':'仅元数据'} · ${r.metadata_status||'架构待读取'}${r.license?' · '+r.license:''}`));card.onclick=()=>{select.value=r.key;open.textContent=r.repo||r.name;dialog.close();choose();};results.append(card);}
    for(const card of results.querySelectorAll('.model-choice')){
      const r=rows.find(m=>m.key===card.dataset.modelKey),support=r?.planning_support;
      card.append(el('small',support?.label||'需读取结构元数据','model-planning-support'));
    }
    if(!rows.length)results.append(el('p','没有匹配项。尝试清除筛选，或点击“搜索云端模型”。'));}
  open.onclick=()=>{drawCards();dialog.showModal();search.focus();};
  for(const control of [category,vendor,size])control.onchange=drawCards;
  let records=[];
  function draw(){const current=select.value||S.forms.resources.model_key||'',q=search.value.trim().toLowerCase();select.replaceChildren();const manual=el('option',t('manual'));manual.value='';select.append(manual);for(const r of records){const p=r.parameter_estimate/1e9||r.nominal_parameters_b;const o=el('option',(r.kind==='local'?'Local':r.featured?tr('Common','常见'):'Hub')+' · '+(r.model_card_title||r.repo||r.name)+' · '+(p?p.toLocaleString(undefined,{maximumFractionDigits:3})+' B'+(r.parameter_estimate?'':tr(' nominal',' 标称')):'—'));o.value=r.key;select.append(o);}select.value=current;}
  async function refresh(){records=await api('/api/modelchoices',{include_featured:true});const previous=vendor.value;vendor.replaceChildren(el('option','所有厂商'));vendor.firstChild.value='';for(const author of [...new Set([...Object.keys(S.info.model_vendors||{}),...records.map(r=>(r.repo||'').split('/')[0]).filter(Boolean)])].sort()){const o=el('option',(S.info.model_vendors||{})[author]||author);o.value=author;vendor.append(o);}vendor.value=previous;draw();drawCards();const chosen=records.find(r=>r.key===S.forms.resources.model_key);if(chosen)open.textContent=chosen.repo||chosen.name;}
  search.oninput=()=>{draw();drawCards();};
  const find=el('button',tr('Search cloud models','搜索云端模型'));find.type='button';find.onclick=async()=>{const q=search.value.trim();find.disabled=true;count.textContent='正在检索公开元数据，不下载权重…';try{const result=await api('/api/catalog/sync',{source:source.value,author:vendor.value||(source.value==='modelscope'?'all':(q?'':'all')),search:q,limit:100,sort:'downloads'});await refresh();count.textContent+=' '+result.scope+(result.errors?.length?' 部分来源失败：'+result.errors.map(x=>x.author+': '+x.error).join('; '):'');}catch(e){count.textContent='检索失败：'+e.message;}finally{find.disabled=false;}};toolbar.append(find);
  let revision=0;
  async function choose(refreshMetadata=false){
    const rev=++revision,f=S.forms.resources,key=select.value;
    box.dataset.loading=String(!!key);box.dispatchEvent(new Event('change',{bubbles:true}));
    f.model_key='';f.model_path='';f.parameters_b='';
    f.context_limit='';if($('resources_context_limit'))$('resources_context_limit').value='';
    for(const k of ['layers','hidden','heads','kv_heads','ffn','vocab']){f[k]=0;if($('resources_'+k))$('resources_'+k).value=0;}
    $('resources_model_path').value='';$('resources_parameters_b').value='';
    note.textContent=key?tr('Reading cached metadata; missing entries may need a direct network request…','正在读取缓存元数据；未缓存的模型可能需要直连获取…'):'';
    if(!key)return;
    try{
      const m=await api('/api/modelchoices/select',{key,source:source.value,refresh:refreshMetadata});
      if(rev!==revision||!box.isConnected||f!==S.forms.resources)return;
      // Metadata completion changes calculation inputs too, not just user events.
      box.dispatchEvent(new Event('change',{bubbles:true}));
      f.model_key=key;f.model_path=m.kind==='local'?m.path:'';
      f.parameters_b=m.parameter_estimate!=null?m.parameter_estimate/1e9:(m.nominal_parameters_b||'');
      const cfg=m.config?.text_config||m.config||{};
      // Config declares the position limit; do not multiply RoPE scaling again.
      const declared=Number(cfg.max_position_embeddings||cfg.model_max_length);
      const known=Number.isSafeInteger(declared)&&declared>=128&&declared<=10000000;
      f.seq=known?declared:4096;
      $('resources_seq').value=f.seq;
      for(const [k,c]of [['layers','num_hidden_layers'],['hidden','hidden_size'],['heads','num_attention_heads'],['kv_heads','num_key_value_heads'],['ffn','intermediate_size'],['vocab','vocab_size']]){f[k]=cfg[c]||0;$('resources_'+k).value=f[k];}
      $('resources_model_path').value=f.model_path;$('resources_parameters_b').value=f.parameters_b;
      const id=m.identity||{},name=id.official_name||id.display_name||m.name;
      const count=m.parameter_estimate!=null?m.parameter_estimate.toLocaleString()+' ('+(m.parameter_estimate/1e9).toFixed(3)+' B)':m.nominal_parameters_b?m.nominal_parameters_b+' B '+tr('publisher nominal, not a weight scan','官方标称，非权重实测'):t('needParameterCount');
      note.replaceChildren(el('span',name+' · '+count));
      if(m.architecture_reference){const a=el('a','Google官方架构参考（不是下载成功的检查点配置）');a.href=m.architecture_reference.source;a.target='_blank';a.rel='noopener noreferrer';note.append(el('p','规划限文本请求；KV采用所有层完整历史的保守上界，未计图片编码激活。'),a);}
      note.append(el('p',known?tr('Declared context: ','模型声明上下文：')+declared.toLocaleString()+tr(' tokens · loaded as planning length (input + output); editable, not a backend or quality guarantee.',' tokens · 已带入规划长度（输入＋输出），可修改，不代表后端支持或质量保证。'):tr('Context unavailable or outside planner range; 4096 is provisional, not the model limit.','上下文未知或超出规划范围；4096为临时输入，不是模型上限。')));
      if(known){const reset=el('button',tr('Use declared context','使用模型声明长度'));reset.type='button';reset.onclick=()=>{f.seq=declared;$('resources_seq').value=declared;box.dispatchEvent(new Event('change',{bubbles:true}));};note.append(reset);}
      if(m.kind==='remote'||m.kind==='hub'){
        note.append(el('br'),el('span',tr('Metadata: ','元数据：')+(m.metadata_origin||'unavailable')+' / '+(m.metadata_source||m.source)));
        if(m.config_url){const link=el('a',tr('Config source','配置来源'));link.href=m.config_url;link.target='_blank';link.rel='noopener noreferrer';note.append(' · ',link);}
        if(m.metadata_status==='unavailable')note.append(el('p',tr('Metadata unavailable. Retry with another source below; no model dimensions were guessed.','暂时无法获取架构元数据。可切换源后重试；未猜测模型维度。')));
        if(m.metadata_status==='partial')note.append(el('p',tr('Config available; exact logical parameter count is unverified. Nominal count is only an estimate input, not proof of training support.','配置已获得；准确逻辑参数量尚未核实。标称量仅作为估算输入，不代表已经支持该模型训练。')));
        if((m.full_config||cfg).quantization_config)note.append(el('p',tr('Published weights are quantized. Full-precision training requires a compatible checkpoint and backend validation.','发布权重含量化配置。全精度训练需要兼容的权重及后端验证，不能直接视为 BF16 权重。')));
        if(m.metadata_errors?.length){const details=el('details');details.append(el('summary',tr('Source diagnostics','源站诊断')),el('pre',JSON.stringify(m.metadata_errors,null,2)));note.append(details);}
      }
      select.selectedOptions[0].textContent=(m.kind==='local'?'Local':'Hub')+' · '+name+' · '+count;
    }catch(e){if(rev!==revision)return;note.textContent=tr('Could not read metadata: ','读取元数据失败：')+e.message;}
    finally{if(rev===revision){box.dataset.loading='false';box.dispatchEvent(new Event('change',{bubbles:true}));}}
  }
  select.onchange=()=>choose();
  const retry=el('button',tr('Refresh metadata from selected source','从所选源刷新元数据'));retry.type='button';
  retry.onclick=async()=>{retry.disabled=true;try{await choose(true);}finally{retry.disabled=false;}};box.append(retry);
  actions(box,button('refresh',refresh));
  box.append(el('p',tr('Select or search here, then calculate below. Local model paths can also be entered under Advanced inputs; leaving this planner is not required.','在这里选择或检索模型，然后在下方计算。本地模型也可在高级输入中填写路径，无需离开规划器。')));
  refresh().catch(e=>{note.textContent=e.message;});
}
function jobForm(group,parent){const f=S.forms[group];if(group!=='deploy')parent.append(txt('inputNote'),txt(group==='train'?'trainPlanHelp':'compressPlanHelp'));if(group==='deploy')deploymentModelPicker(parent);const g=fields(parent);if(group!=='deploy'&&S.models.length){const choose=field(g,group,'known_model','knownModel','text',[['','manual'],...S.models.map(m=>[m.path,m.name])]);choose.onchange=()=>{f.model_path=choose.value;delete S.plans[group];render();};}field(g,group,'model_path','path');if(group!=='deploy')field(g,group,'dataset','dataset');if(group==='train'){field(g,group,'task','task','text',['cpt','sft','dpo','grpo','ppo'].map(x=>[x,x.toUpperCase()]));field(g,group,'tuner','tuner','text',[['lora','LoRA'],['full','Full']]);for(const [k,l]of [['steps','steps'],['batch','batch'],['grad_acc','gradAcc'],['learning_rate','lr']])field(g,group,k,l,'number');}if(group==='deploy')field(g,group,'backend','backend','text',[['vllm','vLLM'],['sglang','SGLang'],['mlx','MLX']]);if(group==='optimize')field(g,group,'method','method','text',[['awq','AWQ / INT4'],['gptq','GPTQ / INT4'],['fp8','FP8'],['mlx4','MLX / 4-bit'],['structured-pruning','Structured pruning'],['sparse-2of4','2:4 sparsity']]);field(g,group,'backend_python','python');for(const k of ['seq','gpus'])field(g,group,k,k,'number');if(group==='deploy')for(const k of ['tp','port'])field(g,group,k,k,'number');field(g,group,'profile','profile','text',[['none','None'],['nsys','Nsight Systems']]);field(g,group,'execution','execution','text',[['native','native'],['managed','managed']]);const explain=el('details');explain.append(el('summary',t('explanation')));const explanationText=el('div');function explainNow(){explanationText.replaceChildren();if(group==='deploy')explanationText.append(txt('inferExplain'),txt(f.backend+'Mechanism'));if(group==='optimize')explanationText.append(txt(['awq','gptq','fp8','mlx4'].includes(f.method)?f.method:'prune'),txt('gainNote'));}explainNow();explain.append(explanationText);if(group!=='train')parent.append(explain);for(const k of ['backend','method'])$(group+'_'+k)?.addEventListener('change',explainNow);parent.append(txt('planNote'));const summary=el('div',undefined,'selected-plan');function showPlan(){summary.replaceChildren();const p=S.plans[group];if(!p)return;summary.append(el('h3',t('preflight')),el('p',t(p.blockers.length?'blocked':'ready'),p.blockers.length?'danger':''),el('h3',t('command')),el('pre',p.argv.map(x=>JSON.stringify(x)).join(' ')));for(const reason of p.blockers)summary.append(el('p',reason,'danger'));for(const warning of p.warnings||[])summary.append(el('p',warning,'note'));raw(summary,p);const b=button('run',async()=>{if(!confirm(t('confirm')))return;await api('/api/start',{id:p.id,confirm:p.id,native:f.execution==='native'});if(group==='deploy')S.lastDeployment=p.id;delete S.plans[group];await refreshServices();navigate(group==='deploy'?'deploy':'runs');},true);b.dataset.run=group;b.disabled=!!p.blockers.length;summary.append(b);}actions(parent,button('planPreview',async()=>{const request={...f};delete request.execution;delete request.known_model;if(group==='optimize')request.backend=f.method==='mlx4'?'mlx':'swift';const r=await api('/api/plan',request);S.plans[group]=r.plan;showPlan();},true));parent.append(summary);showPlan();}
function buildTracePanel(){
  const p=panel('traceTitle');p.classList.add('observatory');p.append(txt('traceScope'));
  const inspectorLink=el('a',language==='zh'?'TessWeave 解码透视 ↗':'TessWeave Decode Inspector ↗');
  const inspectorService=S.services.find(x=>x.id===S.forms.playground.service_id);
  if(inspectorService){try{const target=new URL(inspectorService.url);if(['http:','https:'].includes(target.protocol)&&['127.0.0.1','localhost','[::1]'].includes(target.hostname)){target.pathname='/inspector';target.search='';target.hash='';inspectorLink.href=target.href;inspectorLink.target='_blank';inspectorLink.rel='noopener noreferrer';p.append(inspectorLink,el('p',language==='zh'?'仅 TessWeave Engine 提供此页面；以 --decode-trace 新文件.jsonl 启动。外部 MLX 实验使用独立 HTML 报告。远端先转发引擎端口。':'Available on TessWeave Engine only: start with --decode-trace NEW.jsonl. External MLX experiments export standalone HTML reports. Forward the engine port for remote hosts.'));}}catch(_){/* Invalid/unregistered service URL: do not create a navigation link. */}}
  const head=el('div',undefined,'trace-flow');head.id='traceFlow';p.append(head);
  const values=el('div');values.id='traceValues';p.append(values);
  const engine=el('div');engine.id='engineMetrics';p.append(engine);
  const remoteLabel=el('label',t('traceRemote')),remote=el('input');remote.placeholder='gpu-server';remoteLabel.append(remote);p.append(remoteLabel,txt('traceRemoteHelp'));
  actions(p,button('traceRemoteSave',async()=>{if(S.chatId)throw Error(t('stop'));const s=S.services.find(x=>x.id===S.forms.playground.service_id);if(!s)throw Error(t('noService'));await api('/api/services/register',{...s,telemetry_ssh_alias:remote.value});await refreshServices();remote.value=S.services.find(x=>x.id===s.id).telemetry_ssh_alias||'';}));
  remote.value=S.services.find(x=>x.id===S.forms.playground.service_id)?.telemetry_ssh_alias||'';
  $('playground_service_id').addEventListener('change',()=>{remote.value=S.services.find(x=>x.id===S.forms.playground.service_id)?.telemetry_ssh_alias||'';clearInterval(S.replayTimer);S.traceEvents=[];S.traceLog='';S.traceReplay=false;renderChat();});
  const chart=el('canvas');chart.width=1100;chart.height=190;chart.id='traceChart';chart.setAttribute('aria-label',t('traceChart'));chart.setAttribute('role','img');p.append(txt('traceChart'),chart,txt('traceMatrix'));
  const path=el('p');path.id='tracePath';p.append(path);
  const backend=el('select');backend.setAttribute('aria-label','Profiler backend');for(const name of ['vllm','sglang']){const o=el('option',name);o.value=name;backend.append(o);}p.append(txt('traceCaptureHelp'),backend);
  const captureStatus=el('p');p.append(captureStatus);
  async function capture(action){const id=S.forms.playground.service_id;if(!id)throw Error(t('noService'));if(!confirm(t('traceCaptureConfirm')))return;captureStatus.textContent=t('running');try{const r=await api('/api/services/profile',{service_id:id,backend:backend.value,action,confirm:id});captureStatus.textContent=r.status+' · '+action+' · '+r.response;}catch(e){captureStatus.textContent=t('traceCaptureUncertain');throw e;}}
  actions(p,button('traceCaptureStart',()=>capture('start')),button('traceCaptureStop',()=>capture('stop')));
  const traceInput=el('input');traceInput.placeholder='/path/to/trace.json.gz';traceInput.setAttribute('aria-label',t('traceImport'));p.append(txt('traceImportHelp'),traceInput);
  actions(p,button('traceImport',async()=>{if(S.chatId)throw Error(t('stop'));clearInterval(S.replayTimer);const r=await api('/api/profiler/import',{path:traceInput.value});S.traceEvents=r.events;S.traceLog=r.log_path;S.traceReplay=true;S.chat=[];S.chatMetrics={};S.chatReasoning='';renderChat();await refresh();}));
  const profile=el('div');profile.id='profileTimeline';p.append(profile);
  const select=el('select');select.setAttribute('aria-label',t('traceReplay'));p.append(select);
  async function refresh(){const rows=await api('/api/chat/recordings',{});select.replaceChildren();for(const row of rows){const o=el('option',new Date(row.updated_at*1000).toLocaleString()+' · '+row.id.slice(0,8));o.value=row.id;select.append(o);}}
  actions(p,button('refresh',refresh),button('traceReplay',async()=>{
    if(S.chatId)throw Error(t('stop'));if(!select.value)return;clearInterval(S.replayTimer);
    const r=await api('/api/chat/replay',{id:select.value});S.traceEvents=[];S.traceLog=r.log_path;S.traceReplay=true;S.chat=[];S.chatReasoning='';S.chatMetrics={};
    const events=r.events.sort((a,b)=>(a.elapsed_s||0)-(b.elapsed_s||0));let index=0;const start=performance.now();
    function step(){const seconds=(performance.now()-start)/1000;while(index<events.length&&(events[index].elapsed_s||0)<=seconds){const e=events[index++];S.traceEvents.push(e);if(e.type==='request')S.chat=[...(e.messages||[]).filter(m=>m.role!=='system'),{role:'assistant',content:''}];if(e.type==='content'&&S.chat.length)S.chat[S.chat.length-1].content+=e.text;if(e.type==='reasoning')S.chatReasoning+=e.text;if(e.type==='completed')S.chatMetrics=e.metrics||{};}renderChat();if(index>=events.length)clearInterval(S.replayTimer);}
    S.replayTimer=setInterval(step,100);step();
  }),button('traceStop',()=>{clearInterval(S.replayTimer);}));refresh().catch(fail);drawTrace();
}
function drawTrace(){
  if(!$('traceValues'))return;
  const events=S.traceEvents||[],request=events.find(e=>e.type==='request'),last=[...events].reverse().find(e=>e.response_bytes!=null),sample=[...events].reverse().find(e=>e.type==='telemetry')?.sample,gpu=sample?.gpus?.[0],missing=t('traceMissing');
  $('traceFlow').textContent=(S.traceReplay?t('traceReplayNote'):t('traceLive'))+' · '+t('tracePipeline');$('traceFlow').classList.toggle('streaming',!!S.chatId);
  const out=$('traceValues');out.replaceChildren();cards(out,{traceRequest:request?.request_bytes??'—',traceResponse:last?.response_bytes??S.chatMetrics.response_bytes??0,traceChunks:events.filter(e=>e.type==='content'||e.type==='reasoning').length,traceLoad:sample?fmt(sample.load1)+' / '+sample.cpu_count:missing,traceGpu:gpu?.utilization!=null?gpu.utilization+'%':missing,traceMem:gpu?.used_mib!=null?gpu.used_mib+' MiB':missing});
  $('tracePath').textContent=t('traceLog')+': '+(S.traceLog||'—')+(sample?' · '+sample.host+' · '+sample.scope:'');
  const eng=[...events].reverse().find(e=>e.type==='engine_metrics')?.sample,ep=$('engineMetrics');ep.replaceChildren();
  if(sample?.gpus?.length){table(ep,['GPU','Source','Utilization','Memory'],sample.gpus.map(g=>[g.index,g.source||'nvidia-smi',g.utilization==null?'—':g.utilization+'%',g.used_mib!=null?g.used_mib+' MiB':g.driver_in_use_mib!=null?g.driver_in_use_mib.toFixed(1)+' MiB (driver / unified)':'—']));}
  if(sample?.error)ep.append(el('p',sample.error));
  if(eng){ep.append(el('p',eng.source+' · '+(eng.scope||eng.status)));table(ep,['Metric','Value','Series'],eng.series.map(r=>[r.name,r.value,r.labels]));if(eng.status!=='ok')ep.append(txt('traceMetricsMissing'));}
  const net=sample?.network_counters;if(net&&Object.keys(net).length)table(ep,['Interface','RX bytes','TX bytes'],Object.entries(net).map(([k,v])=>[k,v.rx_bytes,v.tx_bytes]));
  const profile=events.find(e=>e.type==='profiler')?.trace;drawProfile(profile);
  const canvas=$('traceChart'),ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#081525';ctx.fillRect(0,0,1100,190);ctx.strokeStyle='#203449';for(let y=30;y<190;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(1100,y);ctx.stroke();}
  const points=events.filter(e=>e.response_bytes!=null),maxX=Math.max(1,...points.map(e=>e.elapsed_s)),maxY=Math.max(1,...points.map(e=>e.response_bytes));ctx.strokeStyle='#35e1b8';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(15,165);for(const e of points)ctx.lineTo(15+e.elapsed_s/maxX*1060,165-e.response_bytes/maxY*135);ctx.stroke();ctx.fillStyle='#9dd8ed';ctx.font='13px monospace';ctx.fillText(maxY+' bytes / '+maxX.toFixed(2)+' s',20,20);
}
function drawProfile(trace){
  const box=$('profileTimeline');if(!box)return;
  if(box.dataset.hash===(trace?.sha256||''))return;box.dataset.hash=trace?.sha256||'';box.replaceChildren();if(!trace)return;
  box.append(el('h3',trace.file),txt('traceProfileScope'),el('p',`${trace.events.length} / ${trace.total_events} events · ${(trace.duration_us/1000).toFixed(2)} ms`));
  const range=el('input');range.type='range';range.min=0;range.max=1000;range.value=1000;range.setAttribute('aria-label',t('traceScrub'));box.append(range);
  const canvas=el('canvas');canvas.width=1100;canvas.height=250;canvas.setAttribute('role','img');canvas.setAttribute('aria-label',t('traceTimeline'));box.append(canvas);
  const detail=el('div');box.append(detail);const lanes=[...new Set(trace.events.map(e=>e.lane))].slice(0,8),colors=['#35e1b8','#7ab5ff','#ffc476','#d7a5ff'];
  function paint(){const cutoff=Number(range.value)/1000*trace.duration_us,ctx=canvas.getContext('2d');ctx.fillStyle='#081525';ctx.fillRect(0,0,1100,250);ctx.font='12px monospace';
    lanes.forEach((lane,i)=>{ctx.fillStyle='#a7ccdf';ctx.fillText(lane,8,i*28+20);for(const e of trace.events.filter(e=>e.lane===lane&&e.start_us<=cutoff)){ctx.fillStyle=colors[i%colors.length];ctx.fillRect(130+e.start_us/Math.max(1,trace.duration_us)*950,i*28+6,Math.max(1,Math.min(e.duration_us,cutoff-e.start_us)/Math.max(1,trace.duration_us)*950),16);}});
    ctx.strokeStyle='#fff';ctx.beginPath();ctx.moveTo(130+range.value/1000*950,0);ctx.lineTo(130+range.value/1000*950,250);ctx.stroke();
    detail.replaceChildren();table(detail,['Operator / kernel','Category','Start ms','Duration µs','Input shapes'],trace.events.filter(e=>e.start_us<=cutoff).slice(-80).map(e=>[e.name,e.category,(e.start_us/1000).toFixed(3),e.duration_us,e.input_shapes==null?t('traceMissing'):JSON.stringify(e.input_shapes)]));
  }range.oninput=paint;paint();
}
function deploymentModelPicker(parent){
  environmentPicker(parent,'deploy');
  const box=el('div',undefined,'deployment-picker'),label=el('label',t('deployLocalModel')),select=el('select'),note=el('p');select.setAttribute('aria-label',t('deployLocalModel'));label.append(select);box.append(label,txt('deployLocalHelp'),note);parent.append(box);
  let revision=0;
  async function load(){const rows=(await api('/api/modelchoices',{})).filter(m=>m.kind==='local');select.replaceChildren();const empty=el('option',t('choose'));empty.value='';select.append(empty);for(const m of rows){const o=el('option',m.name+' · '+m.path);o.value=m.path;select.append(o);}select.value=S.forms.deploy.model_path;note.textContent=rows.length?t('deploySelectOrBrowse'):t('deployNoModels');}
  async function pick(path){const rev=++revision;note.textContent=t('running');try{const m=await api('/api/scan',{path});if(rev!==revision)return;remember(m);S.forms.deploy.model_path=m.path;delete S.plans.deploy;$('deploy_model_path').value=m.path;document.querySelectorAll('[data-run="deploy"]').forEach(b=>b.disabled=true);await load();note.textContent=m.name+' · '+(m.model_type||m.pipeline)+' · '+bytes(m.weight_bytes)+(m.errors?.length?' · '+m.errors.join('; '):'');}catch(e){if(rev===revision){note.textContent=String(e.message);fail(e);}}}
  select.onchange=()=>{if(select.value)pick(select.value);};
  actions(box,button('selectDirectory',()=>openDirectoryPicker({start:S.forms.deploy.model_path,onSelect:path=>{S.forms.deploy.model_path=path;delete S.plans.deploy;}})),button('scanSelectedModel',()=>pick(S.forms.deploy.model_path)),button('refresh',load),button('envSetup',()=>configureEnvironment('deploy')));
  load().catch(fail);
}
function deploymentProgress(){
  const p=panel('deployProgress'),status=el('p'),log=el('pre',t('empty'));p.append(status,log);
  async function poll(){if(S.view!=='deploy')return;if(!S.lastDeployment){const rows=await api('/api/jobs');S.lastDeployment=rows.find(j=>j.task==='inference')?.id;}if(!S.lastDeployment){status.textContent=t('deployNoTask');return;}
    const id=S.lastDeployment,job=await api('/api/job?id='+id),logs=await api('/api/logs?id='+id);if(S.view!=='deploy')return;status.textContent=id.slice(0,8)+' · '+t(job.status)+' · '+t('deployReadiness');log.textContent=logs.text||t('empty');
    if(['running','starting','stopping'].includes(job.status))S.deployTimer=setTimeout(()=>poll().catch(fail),1500);
  }actions(p,button('refresh',poll),button('runs',()=>navigate('runs')));poll().catch(fail);
}
function jobView(group){const p=panel(group);if(group==='optimize')p.append(txt('compressionScope'));environmentPicker(p,group);jobForm(group,p);}
async function refreshServices(){S.services=await api('/api/services');}
function serviceSelect(parent,group){return field(parent,group,'service_id','service','text',[['','choose'],...S.services.map(s=>[s.id,s.name+' · '+s.model+' · '+s.url])]);}
async function verifyService(id){const notice=el('p',t('checking-generation'),'busy');$('view').prepend(notice);const r=await api('/api/services/ready',{id});for(let i=0;i<320;i++){const state=await api('/api/chat/events',{id:r.id,cursor:0});if(state.status!=='running')break;await new Promise(resolve=>setTimeout(resolve,1000));}await refreshServices();if(S.view==='deploy')render();}
function deployView(){const d=panel('newDeploy');d.append(txt('deployJourney'));jobForm('deploy',d);deploymentProgress();const p=panel('services');p.append(txt('serviceNote'));actions(p,button('refresh',async()=>{await refreshServices();render();}));table(p,['name','url','status','lastChecked','actions'],S.services.map(s=>{const a=el('div',undefined,'actions');a.append(button('readyProbe',()=>verifyService(s.id)),button('unloadModel',async()=>{if(!confirm(t('unloadConfirm')))return;await api('/api/services/unload',{id:s.id,confirm:s.id});await refreshServices();render();}),button('check',async()=>{await api('/api/services/check',{id:s.id});await refreshServices();render();}),button('openChat',()=>{if(S.chatId&&S.forms.playground.service_id!==s.id)throw Error(t('stop'));if(S.forms.playground.service_id!==s.id){S.chat=[];S.chatMetrics={};}S.forms.playground.service_id=s.id;navigate('playground');}),button('test',()=>{S.forms.evaluate.service_id=s.id;navigate('evaluate');}));return [s.name,s.url+' / '+s.model,t(s.status),s.checked_at?new Date(s.checked_at*1000).toLocaleString(language==='zh'?'zh-CN':'en-US'):'—',a];}));const c=panel('connect'),g=fields(c);for(const [k,l]of [['name','serviceName'],['url','url'],['model','modelId']])field(g,'connect',k,l);field(c,'connect','capabilities','serviceCapabilities','text',[['text','capText'],['text-image','capImage']]);c.append(txt('capabilityWarning'));actions(c,button('register',async()=>{const r=await api('/api/services/register',{...S.forms.connect,supports_images:S.forms.connect.capabilities==='text-image'});if(!S.chatId){S.chat=[];S.chatMetrics={};S.forms.playground.service_id=r.id;}S.forms.evaluate.service_id=r.id;await refreshServices();render();},true));}
function imageComposer(parent){const service=S.services.find(x=>x.id===S.forms.playground.service_id);parent.append(txt(service?.supports_images?'imageHelp':'imageOnlyService'));const input=el('input');input.type='file';input.accept='image/png,image/jpeg';input.disabled=!service?.supports_images||!!S.chatId;input.setAttribute('aria-label',t('imageInput'));const preview=el('div');input.onchange=async()=>{S.chatImage=null;preview.replaceChildren();const f=input.files[0];if(!f)return;if(f.size>256*1024||!['image/png','image/jpeg'].includes(f.type)){fail(Error(t('imageInput')));input.value='';return;}const value=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(f);});S.chatImage=value;const img=el('img');img.src=value;img.style.maxWidth='240px';preview.append(img);};parent.append(input,preview);}
function messageContent(content){if(typeof content==='string')return document.createTextNode(content);const box=el('div');for(const p of content||[]){if(p.type==='text')box.append(el('p',p.text));if(p.type==='image_url'){const img=el('img');img.src=p.image_url.url;img.style.maxWidth='240px';img.alt=t('imageInput');box.append(img);}}return box;}
function playgroundView(){const p=panel('playground');p.append(el('p',t('chatNote'),'note'));const g=fields(p),sel=serviceSelect(g,'playground');sel.disabled=!!S.chatId;sel.addEventListener('change',()=>{S.chat=[];S.chatReasoning='';S.chatMetrics={};S.chatImage=null;render();});field(g,'playground','temperature','temperature','number');field(g,'playground','max_tokens','maxTokens','number');field(g,'playground','thinking','thinkingMode','text',[['auto','thinkingAuto'],['off','thinkingOff'],['on','thinkingOn']]);field(p,'playground','system','system','textarea');const log=el('div',undefined,'chat-log');log.id='chatLog';p.append(log);field(p,'playground','message','message','textarea');imageComposer(p);actions(p,button('send',sendChat,true),button('stop',async()=>{if(S.chatId)await api('/api/chat/stop',{id:S.chatId});}),button('clear',async()=>{if(S.chatId)await api('/api/chat/stop',{id:S.chatId});S.chatEpoch++;clearTimeout(S.chatTimer);S.chatId=null;S.chat=[];S.chatReasoning='';S.chatMetrics={};render();}),button('exportSample',()=>{const last=[...S.chat].reverse().find(m=>m.role==='user');if(last)S.forms.evaluate.prompt=last.content;S.forms.evaluate.service_id=S.forms.playground.service_id;navigate('evaluate');}));p.append(txt('playgroundHelp'));buildTracePanel();const metrics=panel('metricNote');metrics.id='chatMetrics';renderChat();}
function renderChat(){if(S.view!=='playground')return;drawTrace();const log=$('chatLog');log.replaceChildren();for(const m of S.chat){const d=el('div',undefined,'bubble '+m.role);d.append(el('strong',t(m.role==='user'?'you':'assistant')),messageContent(m.content||((m.role==='assistant'&&!m.incomplete)?t(S.chatReasoning?'noFinalAnswer':'emptyResponse'):'')));log.append(d);}if(S.chatReasoning){const details=el('details',undefined,'reasoning');details.append(el('summary',t('reasoning')),el('p',S.chatReasoning));log.append(details);}if(S.chatId)log.append(el('p',t('running'),'busy'));log.scrollTop=log.scrollHeight;const m=S.chatMetrics,c=$('chatMetrics');c.replaceChildren(txt('metricNote'));cards(c,{duration:fmt(m.duration_s),first:fmt(m.first_delta_s),firstVisible:fmt(m.first_visible_delta_s),inputTokens:m.input_tokens,outputTokens:m.output_tokens,rate:fmt(m.end_to_end_output_tokens_per_s)});$('playground_service_id').disabled=!!S.chatId;}
async function sendChat(){if(S.chatId)return;clearInterval(S.replayTimer);const f=S.forms.playground;if(!f.service_id)throw Error(t('noService'));if(!f.message.trim())return;const question=S.chatImage?[{type:'text',text:f.message},{type:'image_url',image_url:{url:S.chatImage}}]:f.message;const history=S.chat.filter(m=>!m.incomplete&&(m.role!=='assistant'||m.content)).map(m=>({role:m.role,content:m.content}));history.push({role:'user',content:question});const messages=[...(f.system?[{role:'system',content:f.system}]:[]),...history];const r=await api('/api/chat/start',{service_id:f.service_id,messages,temperature:f.temperature,max_tokens:f.max_tokens,thinking:f.thinking||'auto'});S.traceReplay=false;S.traceEvents=[];S.traceLog=r.log_path||'';S.chat.push({role:'user',content:question},{role:'assistant',content:'',incomplete:true});const answer=S.chat[S.chat.length-1];S.chatReasoning='';S.chatMetrics={};S.chatId=r.id;S.chatCursor=0;const epoch=++S.chatEpoch;S.chatImage=null;f.message='';if($('playground_message'))$('playground_message').value='';renderChat();async function poll(){if(epoch!==S.chatEpoch)return;try{const x=await api('/api/chat/events',{id:r.id,cursor:S.chatCursor});if(epoch!==S.chatEpoch)return;S.chatCursor=x.cursor;for(const event of x.events){S.traceEvents.push(event);if(event.type==='content')answer.content+=event.text;else if(event.type==='reasoning')S.chatReasoning+=event.text;}S.chatMetrics=x.metrics;if(x.status!=='running'&&!x.has_more){S.chatId=null;answer.incomplete=x.status!=='succeeded';if(x.error)fail(Error(x.error));renderChat();return;}renderChat();S.chatTimer=setTimeout(poll,250);}catch(e){S.chatId=null;fail(e);renderChat();}}poll();}
function evaluateView(){const p=panel('quickTest');p.append(el('p',t('benchNote'),'note'));const g=fields(p);serviceSelect(g,'evaluate');for(const [k,l]of [['requests','requests'],['concurrency','concurrency'],['max_tokens','maxTokens']])field(g,'evaluate',k,l,'number');field(g,'evaluate','label','label');field(p,'evaluate','prompt','prompt','textarea');actions(p,button('startBench',async()=>{if(S.benchId)return;const f=S.forms.evaluate;if(!f.service_id)throw Error(t('noService'));if(!Number.isInteger(f.requests)||f.requests<1||f.requests>200)throw Error('Request count must be 1–200');if(!confirm(t('benchConfirm')))return;const r=await api('/api/evaluations/start',{service_id:f.service_id,prompts:Array.from({length:f.requests},()=>({prompt:f.prompt})),concurrency:f.concurrency,max_tokens:f.max_tokens,label:f.label});S.benchId=r.id;render();pollBenchmark();},true));if(S.benchId)p.append(el('p',t('running')+' · '+S.benchId,'busy'));const reports=panel('reports');actions(reports,button('refresh',async()=>{S.reports=await api('/api/reports');render();}));table(reports,['label','success','throughput','p95','fingerprint','details'],S.reports.map(r=>[r.label,r.successes+'/'+r.requests,fmt(r.aggregate_output_tokens_per_s),fmt(r.p95_first_delta_s),r.workload_sha256?.slice(0,12),button('details',()=>raw(reports,r))]));}
async function pollBenchmark(){if(!S.benchId)return;try{const r=await api('/api/evaluations/status',{id:S.benchId});if(r.status==='running'){S.benchTimer=setTimeout(pollBenchmark,1000);return;}S.benchId=null;S.reports=await api('/api/reports');if(r.error)fail(Error(r.error));if(S.view==='evaluate')render();}catch(e){S.benchId=null;fail(e);}}
function runsView(){const p=panel('runs');p.append(txt('plannedHelp'));const list=el('div'),detail=el('div');p.append(list,detail);async function load(){const rs=await api('/api/jobs');list.replaceChildren();table(list,['name','task','status','actions'],rs.map(r=>{const a=el('div',undefined,'actions');a.append(button('details',async()=>{const job=await api('/api/job?id='+r.id),logs=await api('/api/logs?id='+r.id);detail.replaceChildren(el('h3',t('command')),el('pre',job.plan.argv.map(x=>JSON.stringify(x)).join(' ')),el('h3',t('output')),el('p',job.plan.output||''),el('p',t('outputNote')),el('pre',logs.text||t('empty')));if(job.plan.output)detail.append(button('useOutput',async()=>{await inspect(job.plan.output);navigate('models');}));raw(detail,job);raw(detail,logs.metrics);if(job.status==='planned'&&!job.plan.blockers.length)detail.append(button('run',async()=>{if(!confirm(t('confirm')))return;await api('/api/start',{id:job.id,confirm:job.id,native:S.info.native_terminal});await load();},true));}));if(['running','starting','stopping'].includes(r.status))a.append(button('stop',async()=>{if(!confirm(t('confirm')))return;await api('/api/stop',{id:r.id});await load();}));return [r.id.slice(0,8),r.task,t(r.status),a];}));}actions(p,button('refresh',load));load().catch(fail);}
$('language').onchange=()=>{language=$('language').value;localStorage.setItem('workbench-language',language);render();};
$('runsButton').onclick=()=>navigate('runs');$('traceRefresh').onclick=async()=>{try{$('traceOutput').textContent=JSON.stringify(await api('/api/audit'),null,2);}catch(e){fail(e);}};
window.addEventListener('hashchange',()=>{const params=new URLSearchParams(location.hash.slice(1));if(params.has('token')){session=params.get('token')||'';sessionStorage.setItem('workbench-session',session);history.replaceState(null,'',location.pathname+'#'+S.view);document.getElementById('session-recovery')?.close();initializeWorkspace().catch(fail);return;}const v=location.hash.slice(1);if(v!==S.view)navigate(v);});
async function initializeWorkspace(info){S.info=info||await api('/api/info');if(S.info.apple_silicon)S.forms.deploy.backend='mlx';for(const k of ['train','deploy','optimize']){if(!S.forms[k].backend_python)S.forms[k].backend_python=S.info.python;S.forms[k].execution=S.info.native_terminal?'native':'managed';}await refreshServices();S.reports=await api('/api/reports');$('alert').hidden=true;navigate(location.hash.slice(1)||'home');}
initializeWorkspace().catch(e=>{render();fail(e);});
render();
