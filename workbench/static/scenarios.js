'use strict';
function scenarioView(parent){
  parent.append(el('h3','从使用场景出发 · 模型、数据、设备与验收'),el('p','首批候选路线，不是全行业最优榜单。训练主机与最终部署设备可以不同；端侧能运行不意味着适合在端侧训练。','note'));
  S.forms.scenario??={scene:'assistant',modality:'text',target:'cuda',memory_gib:0,budget_cny:0,hour_price_cny:0,fixed_cost_cny:0};const g=fields(parent);
  field(g,'scenario','scene','使用场景','text',[['assistant','文本助手／企业知识'],['documents','图文文档／视觉理解'],['cockpit','座舱助手'],['robot','机器人']]);
  field(g,'scenario','modality','实际输出能力','text',[['text','文本／工具调用'],['vision','看图回答'],['action','动作策略']]);
  field(g,'scenario','target','最终部署设备路线','text',[['cuda','NVIDIA GPU'],['mac','Apple Silicon'],['qualcomm','Qualcomm'],['jetson','Jetson'],['rockchip','Rockchip'],['ascend','Ascend'],['musa','MUSA']]);
  for(const [k,l]of [['memory_gib','模型可用内存 GiB（0 未知）'],['budget_cny','实验总预算／元（0 未知）'],['hour_price_cny','训练整套实例报价／元每小时（0 未知）'],['fixed_cost_cny','云盘／下载／其他费用预算／元']])field(g,'scenario',k,l,'number');
  const result=el('div');let rev=0;g.addEventListener('input',()=>{rev++;result.replaceChildren(el('p','条件已修改，请重新评估。'));});
  actions(parent,button('评估候选路线（不下载／不执行）',async()=>{const id=++rev;result.replaceChildren();const r=await api('/api/scenarios/plan',{...S.forms.scenario});if(id!==rev)return;
    result.append(el('p',r.note,'note'));for(const x of r.blockers)result.append(el('p','需处理：'+x));
    table(result,['模型／官方依据','为何进入候选','BF16 权重下界 GiB','容量结论'],r.candidates.map(c=>{const a=el('a',c.id);a.href=c.source;a.target='_blank';a.rel='noopener noreferrer';return [a,c.why,c.weight_only_bf16_gib?.toFixed(2)??'需读取具体权重',({unknown:'未知',weights_alone_exceed_budget:'仅权重已超预算',weights_only_not_total_fit:'仅权重可能装下，非总容量通过'})[c.capacity_status]];}));
    for(const [title,text]of [['训练路线',r.training],['数据如何准备',r.data],['部署与转换路线',r.deployment],['如何证明可用',r.evaluation]]){const d=el('section',undefined,'academy-detail');d.append(el('h4',title),el('p',text));result.append(d);}
    result.append(el('h4','预算不是训练耗时'),el('p',r.economics.formula),el('p','可负担机时：'+(r.economics.affordable_hours?.toFixed(2)??'未知')+'；是否能完成训练，仍需小规模吞吐与质量试验。'),el('h4','最终推荐前必须补齐'));for(const x of r.missing)result.append(el('p','• '+x));
    actions(result,button('discover',()=>navigate('discover')),button('resources',()=>navigate('resources')),button('infrastructure',()=>navigate('infrastructure')));
  },true,true));parent.append(result);
}
