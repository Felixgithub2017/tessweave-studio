'use strict';
// Full-origin navigation deliberately isolates API tokens, plans, streams and paths.
// A disconnected remote never falls back to local execution.
function safeWorkbenchOrigin(value) {
  try {
    const u = new URL(value);
    if(u.protocol!=='http:' || !['127.0.0.1','localhost','[::1]'].includes(u.hostname) || u.username || u.password || u.pathname!=='/' || u.search || u.hash)return null;
    return u.origin;
  } catch (_) { return null; }
}
if(typeof module!=='undefined')module.exports={safeWorkbenchOrigin};
let executionHostRenderEpoch=0;

async function renderExecutionHost() {
  const epoch=++executionHostRenderEpoch;
  const tr=(en,zh)=>language==='zh'?zh:en;
  let box=document.getElementById('execution-host-bar');
  if(!box){box=el('section',undefined,'panel execution-host-bar');box.id='execution-host-bar';$('alert').before(box);}
  const current=S.info.execution_target;
  const remembered=safeWorkbenchOrigin(sessionStorage.getItem('workbench-return-origin')||'');
  if(!current){box.textContent=tr('Execution host unavailable — reconnect before running tasks.','执行主机未知，请先恢复连接。');if(remembered&&remembered!==location.origin)box.append(button('return',()=>location.assign(remembered+'/#home'),false,true));return;}
  const remote=current.kind==='remote';
  box.replaceChildren(el('strong',(remote?tr('REMOTE GPU','远程 GPU'):tr('LOCAL','本机'))+' · '+current.hostname),
    el('small',tr(' Downloads · models · environments · training · deployment · logs all belong to this host.',' 下载、模型、环境、训练、部署和日志均属于此主机。')));
  const controls=el('div',undefined,'actions'),select=el('select');select.setAttribute('aria-label',tr('Execution host','执行主机'));
  const here=el('option',tr('Current: ','当前：')+current.hostname);here.value='current';select.append(here);
  if(remote&&remembered&&remembered!==location.origin){const o=el('option',tr('Return to local workspace','切回本机工作区'));o.value='return';select.append(o);}
  select.onchange=async()=>{
    const id=select.value;if(id==='current')return;
    if(!confirm(tr('Switch workspace? Unsaved forms are cleared. Existing jobs keep running on their original host.','切换工作区？未保存表单将清空，已有任务继续在原主机执行。'))){select.value='current';return;}
    select.disabled=true;
    try{
      if(id==='return'){location.assign(remembered+'/#home');return;}
      const x=await api('/api/cloud/access',{id});
      if(!Number.isInteger(x.port)||x.port<1024||x.port>65535||!x.token)throw Error('Invalid cloud workspace access');
      const origin=safeWorkbenchOrigin('http://127.0.0.1:'+x.port+'/');
      if(origin===location.origin)throw Error('Remote tunnel port conflicts with this workspace');
      // Token and return address are URL fragments, never query strings or referers.
      const f=new URLSearchParams({token:x.token,return_origin:remembered||location.origin});
      location.assign(origin+'/#'+f.toString());
    }catch(e){select.disabled=false;select.value='current';fail(e);}
  };
  const refresh=button('refresh',()=>renderExecutionHost());
  controls.append(select,refresh,button('infrastructure',()=>navigate('infrastructure')));box.append(controls);
  const data=await api('/api/execution-targets');
  if(!box.isConnected||epoch!==executionHostRenderEpoch)return;
  for(const host of data.targets){const o=el('option',`${host.alias} · ${host.root} · ${host.status}`);o.value=host.id;o.disabled=!host.selectable;select.append(o);}
}
