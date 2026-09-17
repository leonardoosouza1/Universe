const test = require('node:test');
const assert = require('node:assert/strict');

global.window = {};
require('./native_brain.js');

const layout = window.NativeBrain3D.layout();
const offsets = layout.offsets;

function topology() {
  const brain = Object.create(window.NativeBrain3D.prototype);
  brain.offsets=offsets;brain.nodes=layout.nodes;brain.positions=layout.positions;brain.xyz=new Float32Array(layout.positions);brain.n=layout.n;brain.yaw=-.12;brain.pitch=-.12;brain.zoom=1;brain.pan=[0,0];
  brain.data={traceBeing:0,fast:{l1Weights:Array(192*96).fill(0),l2Weights:Array(96*48).fill(0),tau:Array(96).fill(.5),sensoryKnots:Array(16).fill(1),l1Alpha:Array(96).fill(0),l1Period:Array(96).fill(1),l1Threshold:Array(96).fill(0),l1Activation:Array(96).fill(0),l2Alpha:Array(48).fill(0),l2Activation:Array(48).fill(0),outputKnots:Array.from({length:3},()=>Array(12).fill(0))},slow:{gatherMap:Array(48).fill(-1),d1Weights:Array(64*16).fill(0),d1Bias:Array(16).fill(0),d2Weights:Array(16*3).fill(0),d2Bias:Array(3).fill(0)}};
  brain.data.slow.gatherMap[4]=33;
  return brain;
}

test('fast graph keeps sensory KAN, sensory CfC and L1 as distinct stages',()=>{
  const brain=topology();
  assert.deepEqual(brain.incident(offsets['sensory KAN']),[
    {a:offsets.wave,b:offsets['sensory KAN'],weight:null},
    {a:offsets['sensory KAN'],b:offsets['input CfC'],weight:null}
  ]);
  assert.equal(brain.incident(offsets['sensory KAN']+100).length,97);
  assert.equal(brain.incident(offsets['input CfC']).length,97);
  assert.equal(brain.incident(offsets.L1).length,240);
});

test('slow gather preserves zero-filled and executed channel mappings',()=>{
  const brain=topology();
  assert.equal(brain.incident(offsets['slow input']).length,1);
  assert.deepEqual(brain.incident(offsets['slow input']+4)[0],{
    a:33,b:offsets['slow input']+4,weight:null
  });
});

test('measurement exposes only fields that were actually sampled',()=>{
  const brain=topology();brain.selectedBeing=0;brain.sample={being:0,wave:[64],fast_state:[.25],body:{motor_trace:{raw_intent:[.1,.2,.3],after_head:[.4,.5,.6],slow_output:[.7,.8,.9]}}};
  assert.match(brain.measurement(offsets.wave),/64 i8 code/);
  assert.match(brain.measurement(offsets['input CfC']),/sensory memory/);
  assert.equal(brain.measurement(offsets.L1),null);
  assert.match(brain.measurement(offsets['motor head']),/after unidentified motor head/);
  assert.match(brain.measurement(offsets['slow output']),/aggression signal/);
  brain.selectedBeing=1;
  assert.equal(brain.measurement(offsets['input CfC']),null);
  assert.equal(brain.measurement(offsets['motor head']),null);
});

test('focused edges omit zero weights and preserve matrix orientation and signs',()=>{
  const brain=topology(), selected=offsets.L1+7;
  brain.data.fast.l1Weights[7*192+3]=-.25;
  brain.data.fast.l1Weights[7*192+100]=.75;
  brain.data.fast.l2Weights[2*96+7]=.5;
  brain.data.fast.l1Weights[3*192+7]=10; // A transposed lookup must not pick this up.
  assert.deepEqual(brain.visibleConnections(selected),[
    {a:offsets['input CfC']+3,b:selected,weight:-.25},
    {a:offsets['sensory KAN']+100,b:selected,weight:.75},
    {a:selected,b:offsets.L2+2,weight:.5}
  ]);
  assert.deepEqual(brain.connectionSummary(selected),{
    slots:240,shown:3,incoming:2,outgoing:1,zero:237,invalid:0,positive:2,negative:1,fixed:0
  });
  const cfc=offsets['input CfC']+95;
  brain.data.fast.l1Weights[95]=NaN;
  assert.deepEqual(brain.visibleConnections(cfc),[
    {a:offsets['sensory KAN']+95,b:cfc,weight:null}
  ]);
  assert.equal(brain.connectionSummary(cfc).invalid,1);
  assert.equal(brain.connectionSummary(cfc).fixed,1);
});

test('schema-2 stages light up L1/L2; schema-1 samples keep L1 grey',()=>{
  const brain=topology();
  const l1=Array.from({length:96},(_,i)=>i-48), l2=Array.from({length:48},(_,i)=>i-24);
  const schema2={being:0,wave:[64],fast_state:[.1],stages:{l1_effective:l1,l2_adapted:l2,l1_to_l2:l1},
    body:{motor_trace:{raw_intent:[.1,.2,.3]}}};
  const lit=brain.computeActivity(schema2,0);
  assert.ok(lit[offsets.L1]>-1.5,'L1 must be non-grey for a schema-2 sample');
  assert.ok(lit[offsets.L2]>-1.5,'L2 must be non-grey for a schema-2 sample');
  assert.ok(lit[offsets.L1+95]>-1.5&&lit[offsets.L1+95]<=1,'L1 activity stays within the colour range');

  const schema1={being:0,wave:[64],fast_state:[.1],body:{motor_trace:{raw_intent:[.1,.2,.3]}}};
  const grey=brain.computeActivity(schema1,0);
  assert.equal(grey[offsets.L1],-2,'L1 must stay grey (unavailable) without a recorded stages block');
  assert.ok(grey[offsets.L2]>-1.5,'the three L2 motor rows still populate from the legacy motor trace in schema 1');
  assert.equal(grey[offsets.L2+10],-2,'non-motor L2 rows stay grey in schema 1, exactly as before');
});

test('L1<->L2 edge contribution is the recorded activation times the checkpoint weight',()=>{
  const brain=topology();brain.selectedBeing=0;
  brain.data.fast.l2Weights[2*96+5]=3; // L2 node 2 <- L1 node 5
  brain.sample={being:0,stages:{l1_to_l2:Array(96).fill(0).map((_,i)=>i===5?10:0)}};
  const fromL1=brain.incident(offsets.L1+5).find(e=>e.b===offsets.L2+2);
  assert.equal(fromL1.contribution,30);
  const fromL2=brain.incident(offsets.L2+2).find(e=>e.a===offsets.L1+5);
  assert.equal(fromL2.contribution,30);
  brain.sample=null;
  const noSample=brain.incident(offsets.L1+5).find(e=>e.b===offsets.L2+2);
  assert.equal(noSample.contribution,undefined,'without a valid sample, edges keep the old {a,b,weight} shape');
});

test('reset fitting keeps every node inside desktop and narrow canvases',()=>{
  const brain=topology();
  for(const [width,height] of [[900,500],[320,640],[484,180],[320,160]]){
    brain.fitCamera(width,height);
    for(let i=0;i<brain.n;i++){
      const p=brain.projectPoint(i,width,height);
      assert.ok(Math.abs(p.x)<=.781,`node ${i} x=${p.x} at ${width}x${height}`);
      assert.ok(Math.abs(p.y)<=.781,`node ${i} y=${p.y} at ${width}x${height}`);
    }
  }
});
