#!/usr/bin/env python3
import argparse
import base64
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common.config import load_yaml, resolve_path
from src.datasets.frame_reader import preprocess_video, read_rgb_frames
from src.datasets.manifest import read_manifest
from src.datasets.task_roi import ROI_FIELDS
from src.experiments.runner import PROJECT_ROOT


EDITOR = r'''<!doctype html><meta charset="utf-8"><title>Task ROI annotation</title>
<style>body{font:16px sans-serif;max-width:1000px;margin:24px auto}canvas{border:1px solid #aaa;cursor:crosshair;width:672px;height:672px}button,input{margin:6px;padding:8px}#status{white-space:pre-wrap}</style>
<h1>逐帧任务目标 / 眼动 ROI 标注</h1>
<p>在画面拖动矩形标注任务区域；Shift+点击记录注视点。标注关键帧后可插值，必须逐帧检查。ROI 坐标和帧号对应预处理视频。当前编辑器每帧一个框；CSV支持多个ROI。</p>
<select id="sequence"></select><input type="range" id="frame" min="0" value="0"><span id="position"></span><br>
<canvas id="canvas"></canvas><br>
<button id="previous">上一帧</button><button id="next">下一帧</button><button id="clear">清除当前帧</button><button id="interpolate">关键帧间插值</button><button id="export">导出 CSV</button>
<p>标注会自动保存在当前浏览器的本地存储中；仍请及时导出 CSV 备份。</p>
<p>插值不提供目标跟踪保证。每个序列首帧和末帧必须先标注；移动、遮挡或任务切换处增加关键帧。没有实际眼动数据时，请勿把手工点击标记当作实测眼动。</p><div id="status"></div>
<script>
const sequences=__DATA__, annotations={};
const select=document.querySelector('#sequence'), slider=document.querySelector('#frame'), canvas=document.querySelector('#canvas'), ctx=canvas.getContext('2d');
for(const sequence of sequences){let option=document.createElement('option');option.value=sequence.id;option.textContent=sequence.id;select.append(option);annotations[sequence.id]={};}
const storageKey='bistar_task_roi:'+location.pathname;
try{const saved=JSON.parse(localStorage.getItem(storageKey)||'null');if(saved){for(const sequence of sequences){if(saved[sequence.id])annotations[sequence.id]=saved[sequence.id];}}}catch(error){}
const current=()=>sequences.find(s=>s.id===select.value);
let drag=null, image=new Image();
function render(){try{localStorage.setItem(storageKey,JSON.stringify(annotations));}catch(error){}let s=current(),f=+slider.value,a=annotations[s.id][f];canvas.width=s.width;canvas.height=s.height;slider.max=s.images.length-1;document.querySelector('#position').textContent=` frame ${f}/${s.images.length-1} | ${(f/s.fps).toFixed(3)}s`;const token=s.id+':'+f;const nextImage=new Image();nextImage.onload=()=>{if(token!==current().id+':'+slider.value)return;ctx.drawImage(nextImage,0,0);draw();};nextImage.src=s.images[f];image=nextImage;document.querySelector('#status').textContent=`已标注 ${Object.keys(annotations[s.id]).length}/${s.images.length} 帧\n当前来源: ${(a&&a.provenance)||'未标注'}`;}
function draw(){let a=annotations[current().id][+slider.value];if(a&&a.box){let b=a.box;ctx.strokeStyle='#00ff6a';ctx.lineWidth=2;ctx.strokeRect(b[0]*canvas.width,b[1]*canvas.height,(b[2]-b[0])*canvas.width,(b[3]-b[1])*canvas.height);}if(a&&a.gaze){ctx.fillStyle='#ff2266';ctx.beginPath();ctx.arc(a.gaze[0]*canvas.width,a.gaze[1]*canvas.height,3,0,2*Math.PI);ctx.fill();}}
function point(e){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),Math.max(0,Math.min(1,(e.clientY-r.top)/r.height))];}
canvas.onpointerdown=e=>{let s=current(),f=+slider.value;if(e.shiftKey){annotations[s.id][f]=annotations[s.id][f]||{provenance:'manual_boxes'};annotations[s.id][f].gaze=point(e);render();return;}drag=point(e);canvas.setPointerCapture(e.pointerId);};
canvas.onpointerup=e=>{if(!drag)return;let p=point(e),box=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),Math.max(drag[0],p[0]),Math.max(drag[1],p[1])];drag=null;if(box[2]-box[0]<.002||box[3]-box[1]<.002)return;let s=current(),f=+slider.value;annotations[s.id][f]={box,gaze:annotations[s.id][f]&&annotations[s.id][f].gaze,provenance:'manual_boxes'};render();};
slider.oninput=render;select.onchange=()=>{slider.value=0;render();};
document.querySelector('#previous').onclick=()=>{slider.value=Math.max(0,+slider.value-1);render();};document.querySelector('#next').onclick=()=>{slider.value=Math.min(current().images.length-1,+slider.value+1);render();};
document.querySelector('#clear').onclick=()=>{delete annotations[current().id][+slider.value];render();};
document.querySelector('#interpolate').onclick=()=>{let a=annotations[current().id],keys=Object.keys(a).map(Number).filter(k=>a[k].box&&a[k].provenance==='manual_boxes').sort((x,y)=>x-y);for(let n=1;n<keys.length;n++){let l=keys[n-1],r=keys[n];for(let f=l+1;f<r;f++){if(a[f]&&a[f].provenance==='manual_boxes')continue;let u=(f-l)/(r-l);a[f]={box:a[l].box.map((v,j)=>v*(1-u)+a[r].box[j]*u),gaze:undefined,provenance:'manual_interpolated'};}}render();};
document.querySelector('#export').onclick=()=>{let lines=['sequence_id,frame_id,roi_id,x0,y0,x1,y1,gaze_x,gaze_y,provenance'];for(let s of sequences){for(let f=0;f<s.images.length;f++){let a=annotations[s.id][f];if(!a||!a.box){alert(`${s.id} frame ${f} 尚未标注`);return;}lines.push([s.id,f,'task',...a.box.map(v=>v.toFixed(6)),...(a.gaze||['','']),a.provenance].join(','));}}let link=document.createElement('a');link.href=URL.createObjectURL(new Blob([lines.join('\n')+'\n'],{type:'text/csv'}));link.download='task_roi.csv';link.click();URL.revokeObjectURL(link.href);};render();
</script>'''


def prepare(dataset_path, split, output_path, example_center=False):
    dataset = load_yaml(dataset_path)["dataset"]
    manifest = read_manifest(resolve_path(dataset["manifest"], PROJECT_ROOT), split)
    if not manifest:
        raise ValueError("Dataset split has no sequences")
    output_path = Path(output_path)
    if output_path.exists():
        raise ValueError("Annotation output already exists; choose a new --output to preserve labels")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows, editor_data = [], []
    for sequence in manifest:
        folder = output_path.parent / "annotation_sources" / sequence["sequence_id"]
        common = preprocess_video(resolve_path(sequence["video_path"], PROJECT_ROOT), folder / "source.mkv",
                                  dataset["preprocessing"])
        frames = read_rgb_frames(Path(common["path"]))
        images = []
        for index, frame in enumerate(frames):
            ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                raise RuntimeError("Annotation frame export failed")
            images.append("data:image/jpeg;base64," + base64.b64encode(jpeg).decode())
            box = [0.25, 0.25, 0.75, 0.75] if example_center else ["", "", "", ""]
            rows.append(dict(zip(ROI_FIELDS, [sequence["sequence_id"], index, "task", *box,
                                             "", "", "example_center_proxy" if example_center else "unannotated"])))
        editor_data.append({"id": sequence["sequence_id"], "width": common["width"], "height": common["height"],
                            "fps": common["fps"], "images": images})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROI_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    editor_path = output_path.with_suffix(".html")
    safe_data = json.dumps(editor_data).replace("</", "<\\/")
    editor_path.write_text(EDITOR.replace("__DATA__", safe_data), encoding="utf-8")
    print(json.dumps({"csv": str(output_path), "editor": str(editor_path), "frames": len(rows),
                      "annotations_complete": bool(example_center), "proxy_only": bool(example_center)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare aligned per-frame ROI CSV and standalone browser annotation editor")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--example-center", action="store_true", help="Generate explicitly labelled proxy boxes for pipeline checks only")
    args = parser.parse_args()
    prepare(args.dataset, args.split, args.output, args.example_center)
