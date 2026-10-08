'use strict';
const q=new URLSearchParams(location.hash.slice(1)),token=q.get('token')||sessionStorage.getItem('workbench-session'),id=q.get('id')||sessionStorage.getItem('workbench-job');
if(token)sessionStorage.setItem('workbench-session',token);
if(id)sessionStorage.setItem('workbench-job',id);
history.replaceState(null,'',location.pathname);
async function req(path,body){const r=await fetch(path,{method:body?'POST':'GET',headers:{'X-Workbench-Token':token,...(body?{'Content-Type':'application/json'}:{})},body:body?JSON.stringify(body):undefined});const v=await r.json();if(!r.ok)throw Error(v.error);return v;}
let busy=false;
async function tick(){if(busy)return;busy=true;try{const j=await req('/api/job?id='+id);document.getElementById('title').textContent=id.slice(0,8)+' · '+j.status;const r=await req('/api/logs?id='+id);document.getElementById('output').textContent=r.text||'进程尚未输出日志';document.getElementById('stop').disabled=!['running','starting'].includes(j.status);}catch(e){document.getElementById('output').textContent=e.message;}finally{busy=false;}}
document.getElementById('stop').onclick=async()=>{if(confirm('停止此任务及其子进程？')){await req('/api/stop',{id});await tick();}};
tick();setInterval(tick,2000);
