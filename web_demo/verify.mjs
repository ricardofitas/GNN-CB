import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import crypto from 'node:crypto';
import {createRequire} from 'node:module';
import {prepareModel,predict,decodeObservations,observationAt} from './engine.mjs';

const root=path.resolve(new URL('..',import.meta.url).pathname), out=path.join(root,'results/diagnosis/cache/web_demo');
const manifest=JSON.parse(fs.readFileSync(path.join(root,'web_demo/data/manifest.json')));
const refs=JSON.parse(fs.readFileSync(path.join(out,'pytorch_reference.json')));
const verification={schema_version:1,checked_at_utc:new Date().toISOString(),source_commit:manifest.source_commit,
  reference_tolerance:1e-5,cases:[],coordinates_checked:0,max_abs_error:0,checks:[],browser:null};
for(const ref of refs){
  const bytes=fs.readFileSync(path.join(root,`web_demo/data/${ref.id}.json`));
  assert.equal(crypto.createHash('sha256').update(bytes).digest('hex'),manifest.cases.find(c=>c.id===ref.id).sha256);
  const d=JSON.parse(bytes),m=prepareModel(d.model),meta=d.checkpoint_metadata;
  assert(!meta.training_physical_specimens.includes(d.physical_id));assert(!meta.validation_physical_specimens.includes(d.physical_id));
  let error=0;
  for(let k=0;k<ref.phases.length;k++){const y=predict(m,d.graph,ref.phases[k]);for(let i=0;i<y.length;i++)for(let j=0;j<2;j++){error=Math.max(error,Math.abs(y[i][j]-ref.prediction[k][i][j]));verification.coordinates_checked++;}}
  assert(error<verification.reference_tolerance);
  const base=predict(m,d.graph,.3812345),poisoned={...d.graph,observations:[[NaN]],mask:false,images:null,y:Infinity};
  assert.deepEqual(predict(m,poisoned,.3812345),base);
  assert.throws(()=>predict(m,d.graph,-.01));assert.throws(()=>predict(m,d.graph,1.01));assert.throws(()=>predict(m,d.graph,NaN));
  assert.throws(()=>predict(m,{...d.graph,edge_index:[[0],[d.graph.x0.length]]},.5));
  const ob=decodeObservations(d.observations),n=d.graph.x0.length;
  for(let k=0;k<ob.frames.length;k++)assert.equal(observationAt(ob,k,n).filter(Boolean).length,ob.valid.slice(k*n,(k+1)*n).reduce((s,v)=>s+v,0));
  verification.cases.push({id:d.id,held_out:d.physical_id,archived_phases:ob.phases.length,additional_nonarchived_phases:3,max_abs_error:error});
  verification.max_abs_error=Math.max(verification.max_abs_error,error);
}
verification.checks.push('All five checkpoint test identities disjoint from fit and validation',
  'All archived compression phases plus three new phases per case agree with PyTorch',
  'Inference ignores added observation, image, mask and target properties',
  'Invalid phase and invalid edge rejected','Every frame preserves exactly its source observation mask','Manifest SHA256 matches all exported case bytes');

if(process.env.PLAYWRIGHT_PATH){
  const {chromium}=createRequire(import.meta.url)(process.env.PLAYWRIGHT_PATH);
  const server=http.createServer((req,res)=>{const p=path.join(root,decodeURIComponent(req.url).replace(/\?.*$/,''));const f=fs.existsSync(p)&&fs.statSync(p).isDirectory()?path.join(p,'index.html'):p;if(!f.startsWith(root+path.sep)||!fs.existsSync(f)){res.writeHead(404).end();return;}res.setHeader('Content-Type',({'.html':'text/html','.mjs':'text/javascript','.css':'text/css','.json':'application/json','.jpg':'image/jpeg'})[path.extname(f)]||'application/octet-stream');fs.createReadStream(f).pipe(res);});
  await new Promise(r=>server.listen(0,'127.0.0.1',r));
  const proxy=process.env.HTTPS_PROXY||process.env.HTTP_PROXY;
  const browser=await chromium.launch({headless:true,args:['--no-sandbox'],...(process.env.CHROMIUM_PATH?{executablePath:process.env.CHROMIUM_PATH}:{}),...(proxy?{proxy:{server:proxy,bypass:'127.0.0.1,localhost'}}:{})});
  try{
    const errors=[],page=await browser.newPage({viewport:{width:1440,height:1200},ignoreHTTPSErrors:process.env.QA_PROXY_TLS_EXCEPTION==='1',acceptDownloads:true});
    page.on('pageerror',e=>errors.push(e.message));
    await page.goto(`http://127.0.0.1:${server.address().port}/web_demo/`);
    await page.waitForFunction(()=>document.querySelector('#specimen').disabled===false);
    for(const c of manifest.cases){
      await page.selectOption('#specimen',c.id);await page.waitForFunction(id=>document.querySelector('#status').textContent.startsWith(id+' was excluded'),c.physical_id);
      await page.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth>0),null,{timeout:60000});
      await page.locator('#phase').focus();await page.keyboard.press('End');assert.equal(await page.locator('#phase-label').textContent(),'100.0%');
      await page.locator('#phase').focus();await page.keyboard.press('Home');assert.equal(await page.locator('#phase-label').textContent(),'0.0%');
      assert.equal(await page.locator('#initial-svg circle').count(),c.nodes);
    }
    await page.selectOption('#specimen','Probe1_22');await page.waitForFunction(()=>document.querySelector('#status').textContent.startsWith('Probe1_22 was excluded'));
    await page.locator('#snapshots button').nth(1).click();await page.waitForFunction(()=>!document.querySelector('#snapshot-panel').hidden);
    await page.waitForFunction(()=>[...document.images].every(i=>i.complete&&i.naturalWidth>0));
    await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
    const before=await page.locator('#prediction-svg').innerHTML();await page.fill('#free-phase','0.1234567');await page.locator('#free-phase').press('Tab');
    assert.notEqual(await page.locator('#prediction-svg').innerHTML(),before);assert.equal(await page.locator('#rmse').textContent(),'—');assert(await page.locator('#snapshot-panel').isHidden());
    const downloadPromise=page.waitForEvent('download');await page.click('#download');const download=await downloadPromise;const csv=await fs.promises.readFile(await download.path(),'utf8');assert.equal(csv.trim().split('\n').length,41);assert(csv.includes('0.1234567'));
    await page.click('#reset');assert.equal(await page.locator('#phase-label').textContent(),'0.0%');
    await page.setViewportSize({width:375,height:900});await page.locator('#snapshots button').nth(1).click();
    const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>window.innerWidth);assert.equal(overflow,false);
    await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
    const box=await page.locator('#prediction-svg').getAttribute('viewBox');await page.check('#zoom-node');assert.notEqual(await page.locator('#prediction-svg').getAttribute('viewBox'),box);await page.selectOption('#node','10');assert((await page.locator('#node-values').textContent()).includes('Predicted'));await page.uncheck('#zoom-node');
    assert.deepEqual(errors,[]);
    verification.browser={browser:await browser.version(),desktop:'1440x1200',mobile:'375x900',page_errors:errors,
      external_images:'Real pinned GitHub JPEGs loaded and decoded for all five cases',
      checks:['5 specimen selections','10 slider keyboard endpoints','snapshot selection and overlay','continuous phase recalculates graph; unmatched observations and RMSE hidden','CSV downloaded and inspected','reset','node selection and zoom','no horizontal overflow at 375px'],
      proxy_tls_exception:process.env.QA_PROXY_TLS_EXCEPTION==='1',
      limitation:process.env.QA_PROXY_TLS_EXCEPTION==='1'?'Browser trust exception used only for the environment proxy; this is not an independent public TLS audit.':'No test trust exception.'};
  }finally{await browser.close();await new Promise(r=>server.close(r));}
}
verification.source_sha256=Object.fromEntries(['web_demo/engine.mjs','web_demo/app.mjs','web_demo/index.html','web_demo/style.css','web_demo/verify.mjs','scripts/export_web_demo.py'].map(p=>[p,crypto.createHash('sha256').update(fs.readFileSync(path.join(root,p))).digest('hex')]));
verification.status='PASS';fs.writeFileSync(path.join(root,'web_demo/verification.json'),JSON.stringify(verification,null,2)+'\n');console.log(JSON.stringify(verification,null,2));
