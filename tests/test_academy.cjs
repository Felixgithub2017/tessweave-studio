const assert = require('node:assert/strict');
const a = require('../workbench/static/academy.js');
assert.equal(new Set(a.devices.map(d=>d.id)).size,a.devices.length);
for(const d of a.devices){assert.ok(d.source.startsWith('https://'));assert.ok(['cuda','mlx','external'].includes(d.path));assert.ok(d.precision);assert.ok(d.arch);}
assert.equal(a.selectDevices('RK3588','edge')[0].id,'rockchip');
assert.equal(a.selectDevices('RK3588','datacenter').length,0);
assert.equal(a.selectDevices('no matching chip').length,0);
assert.deepEqual(a.handoff('sft','lora'),{ok:true,task:'sft',tuner:'lora'});
assert.ok(a.handoff('dpo','full').ok);
assert.ok(a.handoff('grpo','lora').ok);
for(const task of Object.keys(a.methods))assert.equal(a.handoff(task,'qlora').ok,false);
for(const task of ['gpo','rlhf','ppo','invented'])assert.equal(a.handoff(task,'full').ok,false);
assert.equal(a.handoff('sft','invented').ok,false);
assert.ok(a.devices.find(d=>d.id==='hopper').compute.includes('sparsity'));
console.log('Academy: catalogue and handoff guards passed');
