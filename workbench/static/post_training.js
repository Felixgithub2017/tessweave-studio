'use strict';
function postTrainingJourney(parent){
  const box=el('section',undefined,'academy-detail');parent.append(box);
  box.append(el('h3','模型驱动的连续方案 · SFT → GRPO'),el('p','先解释为什么训练，再决定如何训练。此处读取当前主机模型的 config 与权重头，不加载权重、不启动训练。','note'));
  S.forms.course??={model_path:S.forms.train.model_path||'',mode:'lora',reward:'unknown',seq:2048,prompt_tokens:512,output_tokens:512,group:4,rank:8,samples:1000,mean_tokens:512,epochs:1,batch:1,grad_acc:8,capacity_gib:80,reserve_percent:15,activation_gib:0,workspace_gib:0};
  const g=fields(box);const models=el('select'),modelLabel=el('label','当前主机已扫描模型');models.setAttribute('aria-label','连续方案模型');modelLabel.append(models);g.append(modelLabel);
  const manual=field(g,'course','model_path','模型目录');
  const out=el('div');let revision=0;
  function invalidate(){revision++;out.replaceChildren(el('p','输入已改变，请重新生成解释方案。'));}
  api('/api/modelchoices',{}).then(rows=>{models.append(el('option','选择模型，或填写目录'));models.firstChild.value='';for(const m of rows.filter(x=>x.kind==='local')){const o=el('option',m.name+' · '+m.path);o.value=m.path;models.append(o);}models.value=S.forms.course.model_path;}).catch(fail);
  models.onchange=()=>{S.forms.course.model_path=models.value;manual.value=models.value;invalidate();};
  field(g,'course','mode','连续方案参数更新方式','text',[['lora','LoRA · 目标投影层'],['full','Full · 全参数']]);
  field(g,'course','reward','第二阶段为何使用 GRPO？','text',[['unknown','尚无可核验奖励'],['math','数学答案可核验（现有受限配方）'],['subjective','主观质量／人类偏好（先评估 DPO）']]);
  const inputs=el('section',undefined,'control-section');inputs.append(el('h3','规划输入 · 示例默认值，不是推荐超参数'));box.append(inputs);const ig=fields(inputs);
  for(const [k,label,value]of [['dp','ZeRO 对照 DP',8],['tp','状态分片假设 TP',1],['pp','状态分片假设 PP',1]]){S.forms.course[k]??=value;field(ig,'course',k,label,'number');}
  for(const [k,label]of [['seq','SFT 长度上限'],['samples','SFT 训练样本数（不含验证集）'],['mean_tokens','平均有效 token／样本'],['epochs','SFT 遍历次数'],['batch','每 GPU 微批'],['grad_acc','梯度累积'],['rank','LoRA rank'],['prompt_tokens','GRPO prompt 上限'],['output_tokens','GRPO 输出上限'],['group','每题采样 G'],['capacity_gib','单 GPU 可用容量 GiB（以实测为准）'],['reserve_percent','安全预留 %'],['activation_gib','阶段激活预算 GiB（0=未知，须实测）'],['workspace_gib','阶段工作区预算 GiB（0=未知，须实测）']])field(ig,'course',k,label,'number');
  g.addEventListener('input',invalidate);ig.addEventListener('input',invalidate);
  actions(box,button('生成模型专属解释方案（不执行）',async()=>{const rev=++revision;out.replaceChildren(el('p','正在读取模型结构与权重头…'));const r=await api('/api/post-training/design',{...S.forms.course});if(rev!==revision)return;out.replaceChildren();draw(r);},true,true));box.append(out);
  function section(title,open=false){const d=el('details');d.open=open;d.append(el('summary',title));out.append(d);return d;}
  function draw(r){
    const m=r.model,s=r.schedule;
    out.append(el('h3',m.name+' · '+m.type),el('p',r.execution_note,'note'));
    const correction=section('先纠正概念：这条训练路线是否真的适合你？',true);for(const text of r.corrections)correction.append(el('p',text));
    if(r.blockers.length){const warn=el('div',undefined,'note');warn.append(el('strong','未解决条件 · 这不是可直接启动的最终配方'));for(const x of r.blockers)warn.append(el('p',x));out.append(warn);}
    const arch=section('01 · 这个模型内部是什么？训练改变哪里？',true);arch.append(el('p',r.architecture_note));
    const flow=el('ol',undefined,'academy-steps');for(const x of ['输入／token','Embedding','Attention + FFN × '+(m.dims.num_hidden_layers||'?'),'LM head → 概率','Loss → 梯度 → 参数更新'])flow.append(el('li',x));arch.append(flow);
    table(arch,['结构事实','读取结果／依据'],[['模型类型',m.type],['模态',m.modality],['参数元素',m.parameters?.toLocaleString()||'未知'],['计数来源',m.parameter_basis||'未知'],['专家数',m.experts||'未声明'],...Object.entries(m.dims).map(([k,v])=>[k,v??'未知']),['可训练参数预算',r.update.trainable?.toLocaleString()||'未知'],['LoRA 目标矩阵数',r.update.targets]]);
    arch.append(el('p',r.update.scope),el('p','LoRA：W 保持冻结，训练 A、B，使 W′x = Wx + (α/r)BAx。优化器只更新适配器，但误差仍须穿过冻结层，激活成本不会消失。全参数则更新全部指定可训练权重。'));
    const sft=section('02 · SFT：从一条样本到一次更新',true);
    sft.append(el('p','SFT 是带目标答案的下一 token 学习。输入问题与正确回答，模型输出各位置的词表概率；用正确 token 的负对数概率形成 loss。teacher forcing 使用真实前缀，不是在训练时先自由回答再人工改分。'));
    sft.append(el('code','L_SFT = − Σ(mask_t × log pθ(y_t | x,y_<t)) / Σ mask_t','memory-formula'),el('p','本方案意图：只让 assistant 目标 token 参与损失，prompt／padding 的 mask 为 0；打包时保留样本边界，检查训练器实际 mask。损失经反向传播得到梯度，累积后优化器更新指定参数。'));
    sft.append(el('p','直觉例子：正确 token 的概率只有 0.1 时，单项 loss = −ln(0.1) ≈ 2.303；变成 0.8 时约 0.223。优化器依据梯度调整参数，让示范答案更可能出现；这不是把问答文本存进一个可直接查询的数据库。'));
    sft.append(el('pre',r.data.sft.jsonl),el('p',`预算：${s.samples} 样本 × ${s.mean_tokens} 平均 token × ${s.epochs} 遍 ≈ ${s.sft_tokens.toLocaleString()} 有效 token。微批 ${s.batch} × 累积 ${s.grad_acc} = ${s.effective_batch} 样本/更新；约 ${s.sft_updates} 次更新。`),el('p',s.assumptions));
    sft.append(el('p','先去重、去敏感信息与无授权内容，再按来源／题目族分开训练与验证，避免近似样本泄漏。长度取实际分布与任务需要，不因模型支持 128K 就把每条填满。学习率、rank、轮数需用小样本试验选择；本页不把示例值称为最佳参数。'));
    const bridge=section('03 · SFT 结束后，为什么不能直接点 GRPO？',true);
    bridge.append(el('p','在未见过的验证集上比较原模型与 SFT 模型：任务成功率、格式、基础能力与安全回归。选择验证合格的 checkpoint，而非最后一个或训练 loss 最低的。若没有改善，先查数据、mask、学习率与过拟合，不叠加 RL 掩盖问题。'),el('p','全参数 checkpoint 要包含可复载权重与 tokenizer。LoRA 产物需要底座路径/revision、适配器配置和训练模板；选择继续加载或验证合并。合并前后跑同一验证集，检查 dtype 与 logits。当前执行器未自动编排这些阶段，必须显式复载检查。'));
    const grpo=section('04 · GRPO：没有标准回答 token，如何得到梯度？',true);
    grpo.append(el('p',`策略模型对每个问题采样 ${s.group} 个回答。奖励函数先判分，再比较同组回答好坏。训练不是把 reward 数字反向传播穿过判题器，而是用相对优势加权回答的 log probability，提升高优势行为、抑制低优势行为。`));
    grpo.append(el('code','A_i = (r_i − mean(r_group)) / (std(r_group)+ε)\nρ = πθ(token|prefix) / πold(token|prefix)\n最大化 E[min(ρA, clip(ρ,1−ε,1+ε)A)] − β KL(πθ || πref)','memory-formula'),el('p','这是常见裁剪目标的教学形式，不同 GRPO 实现对归一化、长度权重和 KL 有不同选项；必须记录具体版本与配置。πold 是本轮采样策略，πref 是锚定参考，两者不是同一概念；β=0 时不需要 KL 参考前向。'));
    grpo.append(el('p','四个答案若得分 [1,0,1,0]，组均值 0.5、总体标准差 0.5，则忽略 ε 时优势为 [+1,−1,+1,−1]。若四个都得 0，相对优势全为 0，不能靠这组信号学到哪种回答更好；先检查任务难度与奖励，而不是盲目增加训练时间。'));
    grpo.append(el('pre',r.data.grpo.jsonl),el('p','本数学例的 solution 供判题器使用，不应泄漏到模型 prompt。先测试判题器能处理正确答案、错误答案、空输出、格式伪造；形式正确不等于答案正确。主观服务质量没有可靠判题规则时，应先收集偏好做 DPO 或建设经过校验的奖励模型。'));
    grpo.append(el('p',`一组最大新生成 ${s.group} × ${s.output_tokens} = ${s.grpo_generated_tokens_per_group} token；输入上限 ${s.prompt_tokens}，输出上限 ${s.output_tokens}。G 是采样组大小，不是 GPU 数，也不等同 SFT 梯度累积。`));
    const hardware=section('05 · 硬件为何这样选？两阶段显存逐项账本',true);
    if(r.sharding){const z=r.sharding;hardware.append(el('h4','面试速答 · ZeRO 0–3 到底分别省了什么？'),el('p',z.assumptions),el('p',`DP ${z.dp} × TP ${z.tp} × PP ${z.pp} = ${z.gpus} GPU（仅当前假设，不是建议租用数量）`));
      table(hardware,['ZeRO','权重 GiB/卡','梯度 GiB/卡','优化器 GiB/卡','状态合计 GiB/卡'],z.rows.map(x=>[x.zero,...x.parts.map(p=>p.gib==null?'未知':p.gib.toFixed(3)),x.state_gib==null?'未知':x.state_gib.toFixed(3)]));
      const formulas=el('details');formulas.append(el('summary','展开分片公式与通信代价'));hardware.append(formulas);table(formulas,['ZeRO','分项','代入公式'],z.rows.flatMap(x=>x.parts.map(p=>[x.zero,p.item,p.formula])));
      formulas.append(el('p','DP：复制计算不同样本并同步梯度。TP：层内矩阵切分，频繁集合通信，优先高带宽互联。PP：不同阶段驻留不同层，传递激活并产生流水线气泡。CP：切序列并交换注意力上下文，不默认切权重。EP：路由专家分布与 all-to-all，必须另审计负载均衡。'),el('p','ZeRO-1 分优化器，ZeRO-2 再分梯度，ZeRO-3 再分参数；更高阶段节省驻留状态，却增加参数聚合等通信与瞬时内存，不等于更快。'),el('p',z.caveat));}
    hardware.append(el('p','这是单 GPU、阶段不重叠的保守预算：SFT 和 GRPO 先后执行，不能把两阶段峰值相加。GRPO 按共享 actor/rollout 权重、独立 BF16 参考副本、数学规则奖励预算；独立推理引擎通常还需额外权重副本。激活／工作区需覆盖各自阶段，不能把 0 当免费。'));
    table(hardware,['阶段','状态','当前结论'],r.budgets.map(b=>[b.stage,b.status,b.next]));
    table(hardware,['阶段／分项','精度','元素数','代入公式','GiB'],r.rows.map(x=>[x.stage+' / '+x.item,x.precision,x.elements?.toLocaleString()??'未知',x.formula+(x.elements==null?'':` = ${x.elements} × ${x.bytes_per_element} / 1073741824`),x.gib==null?'未知':x.gib.toFixed(3)]));
    table(hardware,['阶段','已知分项','激活预算','工作区预算','预留','条件峰值'],r.budgets.map(b=>[b.stage,b.known_gib.toFixed(3),b.activation_budget_gib||'未知',b.workspace_budget_gib||'未知',b.reserve_gib.toFixed(3),b.peak_budget_gib?.toFixed(3)??'不完整']));
    hardware.append(el('p','预算放得下后，仍需核对 GPU 架构/BF16、训练后端、模型算子、磁盘 checkpoint 空间与 CPU RAM。CUDA GPU 优先走现有候选训练配方；Mac、NPU 等不能因内存够就套用它。选择更大显存可降低分片复杂度，但哪个最快需测 SFT token/s 与 GRPO rollout/update 分阶段耗时。'));
    const run=section('06 · 如何实际启动、监控和决定继续？',true);
    const steps=el('ol');for(const text of ['扫描当前主机环境并选择 Python；缺依赖先预览安装命令，经用户确认再配置。','导入 SFT 数据并验证字段、模板、mask、长度分布；在训练页预览命令，核对实际 target_modules 与本账本一致。','先做短程 smoke test：loss 有限、非零梯度、checkpoint 能保存和复载、显存峰值可接受；再扩大预算。','SFT 监控训练/验证 loss、任务指标、梯度范数、有效 token/s、OOM 与 checkpoint。验证退化则暂停，不能只看 loss。','验证 SFT checkpoint 并显式复载；审核 GRPO 奖励样例，核对 G、长度、batch 整除约束、参考策略与 KL。','GRPO 监控 reward 分布、组内方差、KL、clip ratio、截断率、rollout/update 用时和独立任务分数；全同分、NaN、奖励上升而能力下降都需要诊断。','保存模型与 tokenizer revision、数据 split/hash、奖励代码版本、seed、环境、命令、响应、退出码；评估通过后再部署。'])steps.append(el('li',text));run.append(steps);
actions(run,button('准备本模型 SFT 数据',()=>{Object.assign(S.forms.data,{model_path:m.path,task:'sft'});navigate('data');},false,true),button('准备 SFT 20 步冒烟配置（不执行）',()=>{Object.assign(S.forms.train,{model_path:m.path,dataset:'',steps:20,gpus:1,tp:1,backend:'swift',task:'sft',tuner:r.update.mode,rank:r.update.rank,seq:s.seq,batch:s.batch,grad_acc:s.grad_acc});delete S.plans.train;navigate('train');},true,true),button('infrastructure',()=>navigate('infrastructure')),button('runs',()=>navigate('runs')));
    run.append(el('p','GRPO 请在 SFT 验收后选择实际产物，再在数据／训练页使用 GRPO 配方；这里不自动把原模型接成第二阶段，也不宣称训练监控指标已全部接入界面。'));
    for(const url of r.sources){const a=el('a',url);a.href=url;a.target='_blank';a.rel='noopener noreferrer';out.append(el('p'),a);}
  }
}
