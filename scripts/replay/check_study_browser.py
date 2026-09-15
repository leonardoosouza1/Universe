#!/usr/bin/env python3
"""Exercise the integrated study UI in real Chrome/WebGL and capture evidence."""
import argparse, hashlib, json, shutil, subprocess, tempfile, time, urllib.request
from pathlib import Path
import websocket
from browser_check import Browser


def run(html, output, expect_brain, expect_vision):
    output.mkdir(parents=True, exist_ok=True)
    (output/'study-validation.json').unlink(missing_ok=True)
    html_digest=hashlib.sha256(html.read_bytes()).hexdigest()
    profile=Path(tempfile.mkdtemp(prefix='study-chrome-',dir=output))
    process=subprocess.Popen(['/usr/bin/google-chrome','--headless=new','--use-gl=angle','--use-angle=swiftshader',
        '--enable-unsafe-swiftshader','--remote-allow-origins=*','--remote-debugging-port=0',
        f'--user-data-dir={profile}','about:blank'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    socket=None
    try:
        deadline=time.monotonic()+20
        while not (profile/'DevToolsActivePort').exists():
            if time.monotonic()>deadline or process.poll() is not None:raise RuntimeError('Chrome did not start')
            time.sleep(.05)
        port=(profile/'DevToolsActivePort').read_text().splitlines()[0]
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/json') as response: tabs=json.load(response)
        socket=websocket.create_connection(next(t['webSocketDebuggerUrl'] for t in tabs if t['type']=='page'),timeout=60)
        b=Browser(socket);b.call('Runtime.enable');b.call('Page.enable')
        b.call('Emulation.setDeviceMetricsOverride',{'width':1800,'height':1100,'deviceScaleFactor':1,'mobile':False})
        b.call('Page.navigate',{'url':html.resolve().as_uri()})
        deadline=time.monotonic()+60
        while not b.evaluate('Boolean(window.__UNIVERSE_REPLAY__)'):
            if time.monotonic()>deadline: raise RuntimeError('Study did not initialize: '+str(b.errors))
            time.sleep(.1)
        b.evaluate('if(__UNIVERSE_REPLAY__.state.playing)document.getElementById("play").click()')
        def seek(tick):
            b.evaluate(f'__UNIVERSE_REPLAY__.state.index=__UNIVERSE_REPLAY__.data.ticks.findIndex(t=>t>={tick});__UNIVERSE_REPLAY__.render()')
        def check_vision_layout():
            layout=b.evaluate('''(() => {
                const box=selector=>document.querySelector(selector).getBoundingClientRect();
                const dock=box('#dock-vision'), image=box('#retina-full'), legend=box('.retina-legend'), footer=box('.transport');
                const hover=box('#retina-hover'), explanation=box('.sensor-explanation');
                return {imageHeight:image.height, legendBottom:legend.bottom, dockBottom:dock.bottom,
                    hoverInside:!explanation.height || (hover.top>=explanation.top && hover.bottom<=explanation.bottom),
                    footerTop:footer.top, legendInside:legend.top>=image.bottom && legend.bottom<=Math.min(dock.bottom,footer.top)};
            })()''')
            assert layout['legendInside'] and layout['hoverInside'] and layout['imageHeight']>=42,layout
            return layout
        seek(128)
        result={'html':str(html),'html_sha256':html_digest,'brain':expect_brain,'vision':expect_vision}
        if expect_brain:
            while not b.evaluate('Boolean(__UNIVERSE_REPLAY__.study.brain)'):
                error=b.evaluate('__UNIVERSE_REPLAY__.study.error')
                if error:raise RuntimeError(error)
                if time.monotonic()>deadline:raise RuntimeError('Brain3D did not initialize')
                time.sleep(.1)
            seek(128)
            result['brain_counts']=b.evaluate('({nodes:REPLAY_BRAIN_3D.n,located:__UNIVERSE_REPLAY__.study.brain.visibleCount,edges:REPLAY_BRAIN_3D.m,drawnEdges:__UNIVERSE_REPLAY__.study.brain.drawnEdges})')
            assert result['brain_counts']['located']>130000
            assert result['brain_counts']['drawnEdges']>0
            result['pixels']=b.evaluate('''(()=>{let b=__UNIVERSE_REPLAY__.study.brain;b.draw();let g=b.gl,p=new Uint8Array(b.canvas.width*b.canvas.height*4);g.readPixels(0,0,b.canvas.width,b.canvas.height,g.RGBA,g.UNSIGNED_BYTE,p);let lit=0,colors=new Set();for(let i=0;i<p.length;i+=4){if(p[i]+p[i+1]+p[i+2]>100){lit++;colors.add(p[i]+(p[i+1]<<8)+(p[i+2]<<16));}}return {lit,colors:colors.size,glError:g.getError()}})()''')
            assert result['pixels']['lit']>10000, result['pixels']
            assert result['pixels']['colors']>20, result['pixels']
            assert result['pixels']['glError']==0
            before=b.evaluate('Array.from(__UNIVERSE_REPLAY__.study.brain.activity)')
            seek(448)
            after=b.evaluate('Array.from(__UNIVERSE_REPLAY__.study.brain.activity)')
            result['changing_neurons']=sum(a!=c for a,c in zip(before,after));assert result['changing_neurons']>100
            b.evaluate('__UNIVERSE_REPLAY__.selectBeing(1)')
            assert b.evaluate('__UNIVERSE_REPLAY__.study.brain.frame')==-1,'borrowed activity from another being'
            b.evaluate('__UNIVERSE_REPLAY__.selectBeing(0)');seek(4)
            assert b.evaluate('__UNIVERSE_REPLAY__.study.brain.frame')==-1,'future sample leaked backwards'
            seek(256)
            box=b.evaluate('(()=>{const r=document.getElementById("brain3d").getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+r.height/2}})()')
            initial=b.evaluate('__UNIVERSE_REPLAY__.study.brain.yaw')
            b.call('Input.dispatchMouseEvent',{'type':'mousePressed','button':'left','clickCount':1,**box})
            b.call('Input.dispatchMouseEvent',{'type':'mouseMoved','buttons':1,'x':box['x']+80,'y':box['y']+20})
            b.call('Input.dispatchMouseEvent',{'type':'mouseReleased','button':'left','clickCount':1,'x':box['x']+80,'y':box['y']+20})
            assert b.evaluate('__UNIVERSE_REPLAY__.study.brain.yaw')!=initial
            result['orbit_verified']=True
            old_pan=b.evaluate('__UNIVERSE_REPLAY__.study.brain.pan.slice()')
            b.call('Input.dispatchMouseEvent',{'type':'mousePressed','button':'left','clickCount':1,'modifiers':8,**box})
            b.call('Input.dispatchMouseEvent',{'type':'mouseMoved','buttons':1,'modifiers':8,'x':box['x']+30,'y':box['y']+15})
            b.call('Input.dispatchMouseEvent',{'type':'mouseReleased','button':'left','clickCount':1,'modifiers':8,'x':box['x']+30,'y':box['y']+15})
            assert b.evaluate('__UNIVERSE_REPLAY__.study.brain.pan')!=old_pan
            old_zoom=b.evaluate('__UNIVERSE_REPLAY__.study.brain.zoom')
            b.call('Input.dispatchMouseEvent',{'type':'mouseWheel','deltaX':0,'deltaY':-120,**box});time.sleep(.15)
            assert b.evaluate('__UNIVERSE_REPLAY__.study.brain.zoom')>old_zoom
            result['pan_zoom_verified']=True
            b.evaluate('document.getElementById("neuron-query").value="T5c";document.getElementById("find-neuron").click()')
            assert b.evaluate('__UNIVERSE_REPLAY__.study.matches.length')>1
            first=b.evaluate('__UNIVERSE_REPLAY__.study.brain.selection')
            b.evaluate('document.getElementById("next-neuron").click()')
            assert b.evaluate('__UNIVERSE_REPLAY__.study.brain.selection')!=first
            result['cell_type_search_verified']=True
            b.evaluate('document.getElementById("reset-brain").click()')
            b.screenshot(output/'study-brain.png')
        if expect_vision:
            seek(256)
            result['vision']=b.evaluate('({length:__UNIVERSE_REPLAY__.study.sample?.retina.length,w:document.getElementById("retina-full").width,h:document.getElementById("retina-full").height,tick:__UNIVERSE_REPLAY__.study.sample?.tick})')
            assert result['vision']['length']==1764 and result['vision']['w']==126 and result['vision']['h']==14
            if b.evaluate('Number.isFinite(__UNIVERSE_REPLAY__.study.sample?.sensor_pose?.gaze_yaw)'):
                projection = b.evaluate("""(() => {
                    const study=__UNIVERSE_REPLAY__.study, select=document.getElementById('vision-projection');
                    const before=JSON.stringify(study.sample.retina), canvas=document.getElementById('retina-full');
                    const set=mode=>{ select.value=mode;select.dispatchEvent(new Event('change'));return [...canvas.getContext('2d').getImageData(0,0,126,14).data]; };
                    const raw=set('raw'), aligned=set('angular');
                    return {rawUnchanged:before===JSON.stringify(study.sample.retina),
                        topChanged:raw.slice(0,126*4).some((v,i)=>v!==aligned[i]),
                        otherRowsUnchanged:raw.slice(126*4).every((v,i)=>v===aligned[126*4+i])};
                })()""")
                assert projection['rawUnchanged'] and projection['otherRowsUnchanged']
                assert projection['topChanged'], 'foveal display did not reproject'
                result['angular_projection']=projection
                hole=b.evaluate("""(() => {
                    const rows=__UNIVERSE_REPLAY__.study.rows.get('0')||[];
                    for(const row of rows) {const i=row.retina.findIndex((p,i)=>i>=126&&(p&255)===3);if(i>=0)return {tick:row.tick,row:Math.floor(i/126),col:i%126};}
                    return null;
                })()""")
                if hole and b.evaluate('__UNIVERSE_REPLAY__.data.meta.terrain!=null'):
                    seek(hole['tick'])
                    box=b.evaluate('(()=>{const r=document.getElementById("retina-full").getBoundingClientRect();return {x:r.x,y:r.y,w:r.width,h:r.height}})()')
                    b.call('Input.dispatchMouseEvent',{'type':'mouseMoved','x':box['x']+(hole['col']+.5)/126*box['w'],'y':box['y']+(hole['row']+.5)/14*box['h']})
                    result['hole_hover']=b.evaluate('document.getElementById("retina-hover").textContent')
                    assert 'recorded hole' in result['hole_hover'] and 'not a recorded hole' not in result['hole_hover'],result['hole_hover']
                    b.screenshot(output/'vision-hole-linked.png')
                seek(256)
                assert b.evaluate('__UNIVERSE_REPLAY__.study.hoverRay===null'), 'hover retained a different sensory frame'
                assert b.evaluate('document.getElementById("retina-hover").textContent.startsWith("Hover a pixel")'), 'stale pixel measurement after seeking'
                result['hover_clock_verified']=True
            b.screenshot(output/'study-vision.png')
            b.evaluate('document.querySelector("[data-tab=behavior]").click()')
            b.screenshot(output/'study-behavior.png')
            b.evaluate('document.querySelector("[data-tab=vision]").click()')
            result['vision_layout']=[{'width':1800, **check_vision_layout()}]
        assert b.evaluate('document.documentElement.scrollWidth<=innerWidth')
        for width in [1280,768,390]:
            b.call('Emulation.setDeviceMetricsOverride',{'width':width,'height':900,'deviceScaleFactor':1,'mobile':False});time.sleep(.15)
            if width<900:
                b.evaluate('if(!document.getElementById("inspector").classList.contains("closed")) document.getElementById("toggle-inspector").click()')
                time.sleep(.2)
            if not b.evaluate('document.documentElement.scrollWidth<=innerWidth'):
                b.screenshot(output/f'overflow-{width}.png')
                overflow=b.evaluate('''[...document.querySelectorAll('body *')].map(e=>({element:e.id||e.className,
                    right:e.getBoundingClientRect().right})).filter(e=>e.right>innerWidth+1).slice(0,20)''')
                raise AssertionError(f'overflow at {width}: {overflow}')
            if expect_vision:
                result['vision_layout'].append({'width':width, **check_vision_layout()})
                b.screenshot(output/f'study-vision-{width}.png')
        assert not b.errors,b.errors
        result['runtime_errors']=b.errors;result['passed']=True
        (output/'study-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    finally:
        if socket:socket.close()
        process.terminate()
        try:process.wait(timeout=10)
        except subprocess.TimeoutExpired:process.kill();process.wait()
        shutil.rmtree(profile)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('html',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--brain',action='store_true');p.add_argument('--vision',action='store_true');a=p.parse_args();run(a.html,a.output,a.brain,a.vision)
