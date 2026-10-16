const assert=require('node:assert/strict');
const {safeWorkbenchOrigin}=require('../workbench/static/host_switch.js');
assert.equal(safeWorkbenchOrigin('http://127.0.0.1:8877/'),'http://127.0.0.1:8877');
for(const u of ['https://evil.com','javascript:alert(1)','http://user:pass@localhost:8877/','http://localhost:8877/a','http://localhost:8877/?token=x','http://localhost:8877/#token=x'])assert.equal(safeWorkbenchOrigin(u),null);
console.log('Host origin guards passed');
