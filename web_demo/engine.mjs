/** Dependency-free port of gnn_cb.model.DeformationModel (original GNN only).
 * Inputs are initial geometry, edges and phase; observations/images never enter.
 * IEEE double arithmetic is checked against the source float32 PyTorch network.
 */
export function decodeTensor(t) {
  const bytes = Uint8Array.from(atob(t.base64), c => c.charCodeAt(0));
  if (t.dtype === 'u1') return bytes;
  if (t.dtype !== '<f4' || bytes.length % 4) throw new Error('Unsupported tensor encoding');
  const view = new DataView(bytes.buffer);
  return Float64Array.from({length: bytes.length / 4}, (_, i) => view.getFloat32(i * 4, true));
}

export function prepareModel(exported) {
  if (exported.config.kind !== 'gnn') throw new Error('Only the verified GNN is supported');
  const weights = Object.fromEntries(Object.entries(exported.weights).map(([k,v]) => [k, decodeTensor(v)]));
  return {config: exported.config, weights};
}

function linear(input, weight, bias) {
  const out = new Float64Array(bias.length), cols = input.length;
  for (let j=0; j<out.length; j++) {
    let v=bias[j];
    for (let k=0; k<cols; k++) v += weight[j*cols+k]*input[k];
    out[j]=v;
  }
  return out;
}
function dense(input, name, w) {
  const h=linear(input,w[name+'.0.weight'],w[name+'.0.bias']);
  for (let j=0;j<h.length;j++) h[j] /= 1+Math.exp(-h[j]);
  return linear(h,w[name+'.2.weight'],w[name+'.2.bias']);
}
function concat(...parts) {
  const a=new Float64Array(parts.reduce((s,p)=>s+p.length,0)); let offset=0;
  for (const p of parts) {a.set(p,offset);offset+=p.length;} return a;
}

export function predict(model, graph, phase) {
  if (!Number.isFinite(phase) || phase<0 || phase>1) throw new Error('Phase must be in [0,1]');
  const x=graph.x0, [src,dst]=graph.edge_index, n=x.length, width=model.config.hidden_dim;
  if (!n || x.some(p=>p.length!==2 || p.some(v=>!Number.isFinite(v)))) throw new Error('Invalid initial geometry');
  if (src.length!==dst.length || [...src,...dst].some(i=>!Number.isInteger(i)||i<0||i>=n)) throw new Error('Invalid edges');
  const mean=[x.reduce((s,p)=>s+p[0],0)/n,x.reduce((s,p)=>s+p[1],0)/n], w=model.weights;
  const centered=x.map(p=>[p[0]-mean[0],p[1]-mean[1]]);
  let h=centered.map(p=>dense([...p,phase,phase*phase],'encoder',w));
  const rel=src.map((s,i)=>{const t=dst[i],dx=x[s][0]-x[t][0],dy=x[s][1]-x[t][1];return [dx,dy,Math.hypot(dx,dy)];});
  const degree=new Int32Array(n); for(const t of dst) degree[t]++;
  for(let block=0;block<model.config.message_steps;block++) {
    const name=`blocks.${block}`, agg=Array.from({length:n},()=>new Float64Array(width));
    for(let e=0;e<src.length;e++) {
      const message=dense(concat(h[src[e]],h[dst[e]],rel[e]),name+'.message',w);
      for(let j=0;j<width;j++) agg[dst[e]][j]+=message[j];
    }
    h=h.map((row,i)=>{
      for(let j=0;j<width;j++) agg[i][j]/=Math.max(1,degree[i]);
      const u=dense(concat(row,agg[i]),name+'.update',w);
      for(let j=0;j<width;j++) u[j]+=row[j];
      const avg=u.reduce((s,v)=>s+v,0)/width;
      const variance=u.reduce((s,v)=>s+(v-avg)**2,0)/width;
      const denom=Math.sqrt(variance+1e-5);
      for(let j=0;j<width;j++) u[j]=(u[j]-avg)/denom*w[name+'.norm.weight'][j]+w[name+'.norm.bias'][j];
      return u;
    });
  }
  return h.map((row,i)=>{
    const residual=dense(row,'decoder',w), [cx,cy]=centered[i];
    const feature=[1,cx,cy,phase,phase*cx,phase*cy,phase*phase,phase*phase*cx,phase*phase*cy];
    return [0,1].map(d=>feature.reduce((s,v,j)=>s+v*w.affine_coefficients[2*j+d],0)+w.residual_scale[0]*residual[d]);
  });
}

export function decodeObservations(data) {
  return {...data, displacement: decodeTensor(data.displacements), valid: decodeTensor(data.mask)};
}

export function observationAt(data, frameIndex, nodes) {
  if (!Number.isInteger(frameIndex) || frameIndex<0 || frameIndex>=data.frames.length) throw new Error('Invalid observation frame');
  return Array.from({length:nodes},(_,i)=>data.valid[frameIndex*nodes+i] ?
    [data.displacement[(frameIndex*nodes+i)*2],data.displacement[(frameIndex*nodes+i)*2+1]] : null);
}
