// NODE_PATH must resolve Playwright. No Internet or model downloads.
const {chromium}=require('playwright');
const {spawn}=require('node:child_process');
const assert=require('node:assert/strict');
(async()=>{
 const fixture=spawn(process.env.PYTHON||'python3',['tests/host_switch_fixture.py'],{env:{...process.env,PYTHONPATH:'.'},stdio:['pipe','pipe','inherit']});
 let browser;
 try {
  const info=await new Promise((resolve,reject)=>{let s='';const timer=setTimeout(()=>reject(Error('Fixture timeout')),10000);fixture.once('exit',()=>reject(Error('Fixture exited')));fixture.stdout.on('data',d=>{s+=d;if(s.includes('\n')){clearTimeout(timer);resolve(JSON.parse(s.split('\n')[0]));}});});
  browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});const page=await browser.newPage();const errors=[];
  page.on('pageerror',e=>errors.push(String(e)));page.on('dialog',d=>d.accept());
  await page.goto(info.local_url);await page.locator('#execution-host-bar select option[value="'+info.job+'"]').waitFor({state:'attached'});
  assert.match(await page.locator('#execution-host-bar strong').innerText(),/LOCAL/);
  await page.locator('#execution-host-bar select').selectOption(info.job);
  await page.waitForURL('**:'+info.remote_port+'/**');
  await page.waitForFunction(()=>document.querySelector('#execution-host-bar strong')?.textContent.includes('REMOTE GPU'));
  assert.match(await page.locator('#execution-host-bar strong').innerText(),/remote-gpu-fixture/);
  const state=await page.evaluate(async()=>{const h={'X-Workbench-Token':sessionStorage.getItem('workbench-session')};return {info:await (await fetch('/api/info',{headers:h})).json(),jobs:await (await fetch('/api/jobs',{headers:h})).json()};});
  assert.equal(state.jobs.length,0);assert.ok(state.info.state.endsWith('/remote'));assert.ok(!page.url().includes('token='));
  const created=await page.evaluate(async parent=>{const r=await fetch('/api/folders/create',{method:'POST',headers:{'X-Workbench-Token':sessionStorage.getItem('workbench-session'),'Content-Type':'application/json'},body:JSON.stringify({parent,name:'download-destination'})});return {status:r.status,body:await r.json()};},state.info.state);
  assert.equal(created.status,200);
  await page.locator('#execution-host-bar select').selectOption('return');
  await page.waitForFunction(()=>document.querySelector('#execution-host-bar strong')?.textContent.includes('LOCAL'));
  assert.equal(new URL(page.url()).origin,new URL(info.local_url).origin);assert.deepEqual(errors,[]);
  const isolated=await page.evaluate(async path=>{const r=await fetch('/api/folders',{method:'POST',headers:{'X-Workbench-Token':sessionStorage.getItem('workbench-session'),'Content-Type':'application/json'},body:JSON.stringify({path})});return r.status;},state.info.state);
  assert.equal(isolated,400); // The local workspace cannot browse the remote fixture root.
  console.log('PASS: same-tab local → remote → local; distinct tokens/state/jobs; no browser errors. Loopback fixture, not SSH qualification.');
 } finally {if(browser)await browser.close();fixture.stdin.end('\n');fixture.kill();}
})().catch(e=>{console.error(e);process.exitCode=1;});
