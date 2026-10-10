#!/usr/bin/env python
"""Build an offline single-HTML copy of the verified real-inference demo.

Original runtime/module/data files stay unchanged. This derived bundle embeds
the original JPEG bytes and updates the embedded per-case integrity hashes.
"""
import base64
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
WEB=ROOT/'web_demo'


def main():
    manifest=json.loads((WEB/'data/manifest.json').read_text()); bundle={}
    for descriptor in manifest['cases']:
        case=json.loads((WEB/descriptor['path']).read_text())
        for image in case['images']:
            raw=(ROOT/image['source_path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest()==image['sha256']
            image['url']='data:image/jpeg;base64,'+base64.b64encode(raw).decode('ascii')
        text=json.dumps(case,separators=(',',':'))
        descriptor['sha256']=hashlib.sha256(text.encode()).hexdigest(); bundle[descriptor['path']]=text
    bundle['data/manifest.json']=json.dumps(manifest,separators=(',',':'))
    engine=re.sub(r'^export ','',(WEB/'engine.mjs').read_text(),flags=re.M)
    app=(WEB/'app.mjs').read_text().split('\n',1)[1]
    loader='const EMBEDDED='+json.dumps(bundle,separators=(',',':')).replace('</','<\\/')+';\n'
    loader+='const fetch=async path=>{if(!(path in EMBEDDED))throw new Error("Unknown embedded asset");return {ok:true,status:200,text:async()=>EMBEDDED[path],json:async()=>JSON.parse(EMBEDDED[path])};};\n'
    code=loader+engine+'\n'+app
    html=(WEB/'index.html').read_text().replace('<link rel="stylesheet" href="style.css">','<style>'+(WEB/'style.css').read_text()+'</style>')
    html=html.replace('<script type="module" src="app.mjs"></script>','<script type="module">'+code.replace('</script','<\\/script')+'</script>')
    dest=WEB/'standalone.html'; dest.write_text(html)
    print(json.dumps({'file':str(dest),'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'network_assets':0}))


if __name__=='__main__': main()
