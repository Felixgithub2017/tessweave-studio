// Offline, real local server. No installations, cloud calls or model execution.
const {chromium}=require('playwright');
const {spawn,execFileSync}=require('node:child_process');
const assert=require('node:assert/strict');
(async()=>{
  const fixture=spawn(process.env.PYTHON||'python3',['tests/host_switch_fixture.py'],{env:{...process.env,PYTHONPATH:'.'},stdio:['pipe','pipe','inherit']});
  let browser;
  try{
    const info=await new Promise((resolve,reject)=>{let s='';const timer=setTimeout(()=>reject(Error('fixture timeout')),10000);fixture.once('exit',()=>reject(Error('fixture exit')));fixture.stdout.on('data',d=>{s+=d;if(s.includes('\n')){clearTimeout(timer);resolve(JSON.parse(s.split('\n')[0]));}});});
    browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{})});
    const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(String(e)));
    await page.goto(info.local_url);await page.locator('#nav button').first().waitFor();
    await page.locator('#language').selectOption('zh');
    await page.locator('#nav').getByRole('button',{name:'硬件地图与训练向导',exact:true}).click();
    await page.getByRole('heading',{name:'H100 SXM',exact:true}).waitFor();
    await page.getByRole('searchbox').fill('RK3588');assert.equal(await page.locator('.academy-list button').count(),1);
    await page.getByLabel('设备类别',{exact:true}).selectOption('datacenter');assert.equal(await page.locator('.academy-list button').count(),0);
    await page.getByRole('searchbox').fill('');await page.getByLabel('设备类别',{exact:true}).selectOption('all');
    assert.equal(await page.locator('.academy-list button').count(),12);
    if(process.env.ACADEMY_SCREENSHOT)await page.screenshot({path:process.env.ACADEMY_SCREENSHOT,fullPage:true});
    await page.getByRole('button',{name:'场景选型',exact:true}).click();
    await page.locator('#scenario_scene').selectOption('robot');
    await page.locator('#scenario_modality').selectOption('action');
    await page.locator('#scenario_target').selectOption('qualcomm');
    await page.getByRole('button',{name:'评估候选路线（不下载／不执行）',exact:true}).click();
    await page.getByRole('link',{name:'lerobot/smolvla_base',exact:true}).waitFor();
    assert.ok((await page.locator('.academy').innerText()).includes('急停'));
    assert.ok((await page.locator('.academy').innerText()).includes('尚未认证'));
    await page.getByRole('button',{name:'后训练向导',exact:true}).click();
    const report=JSON.parse(execFileSync(process.env.PYTHON||'python3',['-c',"import json; from workbench.post_training import design_course; print(json.dumps(design_course(dict(path='/fixture',name='Course fixture',model_type='qwen2',modality='text',experts=0,parameter_estimate=7000000000,config=dict(num_hidden_layers=28,hidden_size=3584,num_attention_heads=28,num_key_value_heads=4),tensors=[]),dict(mode='full'))))"],{encoding:'utf8'}));
    await page.route('**/api/post-training/design',route=>route.fulfill({json:report}));
    await page.locator('#course_model_path').fill('/fixture');
    await page.getByRole('button',{name:'生成模型专属解释方案（不执行）',exact:true}).click();
    await page.getByRole('heading',{name:'Course fixture · qwen2'}).waitFor();
    assert.ok((await page.locator('.academy').innerText()).includes('未解决条件'));
    assert.ok((await page.locator('.academy').innerText()).includes('rollout KV'));
    await page.getByText('规划输入 · 示例默认值，不是推荐超参数',{exact:true}).click();
    await page.locator('#course_seq').fill('1024');
    assert.equal(await page.getByRole('heading',{name:'Course fixture · qwen2'}).count(),0);
    const apply=page.getByRole('button',{name:'填入训练配置（不执行）',exact:true});
    await page.getByLabel('参数更新方式',{exact:true}).selectOption('qlora');assert.ok(await apply.isDisabled());
    await page.getByLabel('参数更新方式',{exact:true}).selectOption('full');
    await page.getByLabel('训练目标',{exact:true}).selectOption('ppo');assert.ok(await apply.isDisabled());
    await page.getByLabel('训练目标',{exact:true}).selectOption('gpo');assert.ok(await apply.isDisabled());
    assert.ok(await page.getByRole('button',{name:'查看模型专用数据指南',exact:true}).isDisabled());
    await page.getByLabel('训练目标',{exact:true}).selectOption('dpo');await apply.click();
    assert.equal(await page.locator('#train_task').inputValue(),'dpo');assert.equal(await page.locator('#train_tuner').inputValue(),'full');
    const jobs=await page.evaluate(async()=>await (await fetch('/api/jobs',{headers:{'X-Workbench-Token':sessionStorage.getItem('workbench-session')}})).json());
    assert.equal(jobs.length,1); // pre-existing fake tunnel from the fixture, no training created
    await page.locator('#nav').getByRole('button',{name:'硬件地图与训练向导',exact:true}).click();
    await page.setViewportSize({width:390,height:844});
    assert.ok(await page.locator('.academy').evaluate(e=>e.scrollWidth<=e.clientWidth+2));
    assert.deepEqual(errors,[]);console.log('Academy browser: routes, filters, guards, transfer, responsive layout passed');
  }finally{if(browser)await browser.close();fixture.stdin.end('\n');}
})().catch(e=>{console.error(e);process.exitCode=1;});
