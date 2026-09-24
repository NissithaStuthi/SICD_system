import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import streamlit as st
from src.model import MultiClassSiameseUNet

st.set_page_config(page_title='GeoVision AI Platform', page_icon='🛰️', layout='wide')

# IMPORTANT: these are visual land-cover estimates applied ONLY to pixels
# first identified as changed. The existing LEVIR-CD+ model is used as a
# change detector; its output channels are NOT interpreted as water/road/etc.
CLASS_NAMES = {0:'No Change', 1:'Building', 2:'Vegetation', 3:'Road', 4:'Water Body'}
COLOR_MAP = {0:[0,0,0], 1:[239,68,68], 2:[34,197,94], 3:[234,179,8], 4:[59,130,246]}

st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif}.stApp{background:#f8fafc}
.block-container{padding:1.2rem 2rem 2rem;max-width:1500px}.main-title{background:linear-gradient(135deg,#f97316,#c2410c);padding:22px 32px;border-radius:16px;color:#fff;margin-bottom:22px;box-shadow:0 10px 25px -5px rgba(234,88,12,.35)}
.main-title h1{font-size:30px;font-weight:800;margin:0}.section{font-size:18px;font-weight:800;color:#0f172a;text-transform:uppercase;letter-spacing:.5px;margin:8px 0 15px}
.card{background:#fff;border:1px solid #e2e8f0;border-left:6px solid #ea580c;border-radius:16px;padding:20px;box-shadow:0 8px 20px rgba(0,0,0,.04)}
.metric{background:#fff;border:1px solid #e2e8f0;border-left:6px solid #ea580c;border-radius:14px;padding:18px;text-align:center}.num{color:#ea580c;font-size:28px;font-weight:800}.lbl{color:#64748b;font-size:13px;font-weight:600}
.status{background:#fff;border:1px solid #e2e8f0;border-left:5px solid #ea580c;border-radius:10px;padding:12px 16px;margin-top:15px}.footer{text-align:center;color:#94a3b8;font-size:12px;padding-top:30px}
</style>''', unsafe_allow_html=True)

for key, default in [('page','Home'),('results',None),('before_image',None),('after_image',None),('status','Ready for analysis.')]:
    if key not in st.session_state: st.session_state[key]=default

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiClassSiameseUNet(in_channels=3, num_classes=1).to(device)

    path = "best_siamese_model.pth"

    state = torch.load(path, map_location=device)
    model.load_state_dict(state)

    del state
    model.eval()

    return model, device
model,device=load_model()

def prepare_image(f):
    if f is None:return None
    data=np.asarray(bytearray(f.read()),dtype=np.uint8)
    img=cv2.imdecode(data,cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def cva_difference(a,b):
    d=np.linalg.norm(a.astype(np.float32)-b.astype(np.float32),axis=2)
    scale=max(float(np.percentile(d,99)),1.0)
    d=np.clip(d/scale,0,1)
    return cv2.GaussianBlur(d,(5,5),0)

def clean_mask(mask,min_area=25):
    k=np.ones((3,3),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,k,iterations=1)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,k,iterations=2)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    out=np.zeros_like(mask)
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]>=min_area: out[labels==i]=255
    return out

def features(img,mask):
    px=img[mask>0]
    if len(px)<5:return dict(brightness=0,saturation=0,green=0,blue=0,edge=0)
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV); hp=hsv[mask>0]
    gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY); edges=cv2.Canny(gray,60,140)[mask>0]
    r=px[:,0].astype(float);g=px[:,1].astype(float);b=px[:,2].astype(float);den=r+g+b+1
    return dict(brightness=float(hp[:,2].mean()/255),saturation=float(hp[:,1].mean()/255),
                green=float(np.mean((g-(r+b)/2)/den)),blue=float(np.mean((b-(r+g)/2)/den)),edge=float(np.mean(edges>0)))

def classify_region(img,mask):
    f=features(img,mask); g,b,s,v,e=f['green'],f['blue'],f['saturation'],f['brightness'],f['edge']
    # Heuristic semantic estimate. Water requires blue dominance and low texture,
    # preventing a grey highway from automatically becoming water.
    vegetation=.65*np.clip((g+.04)/.12,0,1)+.20*np.clip(s/.65,0,1)+.15*np.clip(e/.30,0,1)
    water=.60*np.clip((b+.025)/.10,0,1)+.20*np.clip(s/.55,0,1)+.20*(1-np.clip(e/.20,0,1))
    road=.45*(1-np.clip(s/.45,0,1))+.30*np.clip(e/.25,0,1)+.15*np.clip(v/.75,0,1)+.10*(1-np.clip(abs(g)/.08,0,1))
    neutral=1-np.clip(max(abs(g),abs(b))/.10,0,1)
    building=.35*np.clip(v/.80,0,1)+.35*np.clip(e/.30,0,1)+.30*neutral
    if b<.015 or s<.12: water*=.55
    if g<.015: vegetation*=.55
    scores={1:building,2:vegetation,3:road,4:water}
    cid=max(scores,key=scores.get); total=sum(scores.values())+1e-8
    return cid,float(scores[cid]/total),scores

def overlay(img,color_map,alpha=.55):
    out=img.astype(np.float32).copy(); active=np.any(color_map!=0,axis=2)
    out[active]=(1-alpha)*out[active]+alpha*color_map[active]
    return np.clip(out,0,255).astype(np.uint8)

def pipeline(img_before,img_after):
    before=cv2.resize(img_before,(256,256),interpolation=cv2.INTER_AREA)
    after=cv2.resize(img_after,(256,256),interpolation=cv2.INTER_AREA)
    x1=torch.from_numpy(before.astype(np.float32)/255).permute(2,0,1).unsqueeze(0).to(device)
    x2=torch.from_numpy(after.astype(np.float32)/255).permute(2,0,1).unsqueeze(0).to(device)
    with torch.inference_mode():
        out=model(x1,x2); probs=torch.softmax(out,dim=1).squeeze(0).cpu().numpy()
    model_prob=np.max(probs[1:],axis=0) if probs.shape[0]>1 else np.zeros((256,256),np.float32)
    diff=cva_difference(before,after)
    du=(diff*255).astype(np.uint8)
    _,cva=cv2.threshold(du,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cva=clean_mask(cva)
    model_mask=clean_mask((model_prob>=.50).astype(np.uint8)*255)
    strong=clean_mask((diff>=.38).astype(np.uint8)*255)
    binary=clean_mask(((model_mask>0)|(strong>0)).astype(np.uint8)*255)

    pred=np.zeros((256,256),np.uint8); n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8); confs=[]
    for i in range(1,n):
        if stats[i,cv2.CC_STAT_AREA]<25:continue
        rm=(labels==i).astype(np.uint8)*255
        cid,conf,_=classify_region(after,rm); pred[labels==i]=cid; confs.append((cid,conf,stats[i,cv2.CC_STAT_AREA]))
    cmap=np.zeros((256,256,3),np.uint8)
    for cid,col in COLOR_MAP.items():cmap[pred==cid]=col
    ov=overlay(after,cmap,.55); outline=ov.copy(); contours,_=cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);cv2.drawContours(outline,contours,-1,(255,255,255),1)
    heat=np.clip(.6*model_prob+.4*diff,0,1); hu=(heat*255).astype(np.uint8);hu=cv2.GaussianBlur(hu,(9,9),0);att=cv2.cvtColor(cv2.applyColorMap(hu,cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB);base=cv2.cvtColor(cv2.applyColorMap(du,cv2.COLORMAP_JET),cv2.COLOR_BGR2RGB)
    h,w=img_after.shape[:2]
    resize=lambda x,inter=cv2.INTER_LINEAR:cv2.resize(x,(w,h),interpolation=inter)
    pred_big=resize(pred,cv2.INTER_NEAREST);cmap_big=resize(cmap,cv2.INTER_NEAREST);ov_big=resize(ov);out_big=resize(outline);mask_big=resize(binary,cv2.INTER_NEAREST);att_big=resize(att);base_big=resize(base)
    total=pred.size; changed=int((pred>0).sum()); area=changed*2.25
    pixels={}; percentages={}; surface={}
    for cid,name in CLASS_NAMES.items():
        count=int((pred==cid).sum());pixels[name]=count;percentages[name]=100*count/total;surface[name]=count*2.25
    return dict(before=img_before,after=img_after,mask=mask_big,class_map=cmap_big,overlay=ov_big,outline=out_big,attention=att_big,baseline=base_big,predictions=pred_big,difference=diff,change_percentage=100*changed/total,changed_pixels=changed,total_changed_area=area,class_pixel_counts=pixels,class_percentages=percentages,surface_area=surface,region_confidences=confs)

def png(img):
    if img is None:return None
    x=cv2.cvtColor(img,cv2.COLOR_RGB2BGR) if img.ndim==3 else img
    ok,e=cv2.imencode('.png',x);return e.tobytes() if ok else None

def chart(r):
    names=['Building','Vegetation','Road','Water Body']; vals=[r['surface_area'][x] for x in names]; colors=['#ef4444','#22c55e','#eab308','#3b82f6']
    fig,ax=plt.subplots(figsize=(9,4),dpi=120);fig.patch.set_facecolor('#0f172a');ax.set_facecolor('#0f172a');ax.bar(names,vals,color=colors,edgecolor='white');ax.set_title('Estimated Changed Surface Area',color='white',fontweight='bold');ax.set_ylabel('Area (m²)',color='#94a3b8');ax.tick_params(colors='#94a3b8');ax.grid(axis='y',alpha=.2);plt.tight_layout();return fig

st.markdown('<div class="main-title"><h1>Welcome to GeoVision AI Platform</h1></div>',unsafe_allow_html=True)
nav=st.columns(4)
for col,label,page in zip(nav,['Home','Upload Scenes','Detection Results','Analytics & Downloads'],['Home','Upload Scenes','Detection Results','Analytics & Downloads']):
    with col:
        if st.button(label,use_container_width=True):st.session_state.page=page;st.rerun()
st.markdown('---')

if st.session_state.page=='Home':
    st.markdown('<div class="section">Analytical Surface Summary</div>',unsafe_allow_html=True)
    a,b=st.columns([2,1])
    with a:
        st.markdown('<div class="card">',unsafe_allow_html=True)
        if st.session_state.results:
            f=chart(st.session_state.results);st.pyplot(f,use_container_width=True);plt.close(f)
        else:st.info('Ready to analyze satellite imagery.')
        st.markdown('</div>',unsafe_allow_html=True)
    with b:
        st.markdown('<div class="card"><h3>Operational Log</h3>',unsafe_allow_html=True)
        if st.session_state.results:
            r=st.session_state.results;st.write(f'{r["total_changed_area"]:.1f} m² total change detected');st.write(f'{r["change_percentage"]:.2f}% total change')
        else:st.write('Ready to analyze satellite imagery.')
        st.markdown('</div>',unsafe_allow_html=True)

elif st.session_state.page=='Upload Scenes':
    st.markdown('<div class="section">Upload Scenes</div>',unsafe_allow_html=True);c1,c2=st.columns(2)
    with c1:
        f=st.file_uploader('Before Satellite Image',type=['png','jpg','jpeg','tif','tiff'],key='before_file')
        if f is not None:
            im=prepare_image(f);st.session_state.before_image=im
            if im is not None:st.image(im,use_container_width=True)
    with c2:
        f=st.file_uploader('After Satellite Image',type=['png','jpg','jpeg','tif','tiff'],key='after_file')
        if f is not None:
            im=prepare_image(f);st.session_state.after_image=im
            if im is not None:st.image(im,use_container_width=True)
    if st.button('Analyze Changes',use_container_width=True):
        if st.session_state.before_image is None:st.error('Please upload the Before Satellite Image.')
        elif st.session_state.after_image is None:st.error('Please upload the After Satellite Image.')
        else:
            with st.spinner('Detecting changes and estimating land-cover type...'):
                try:st.session_state.results=pipeline(st.session_state.before_image,st.session_state.after_image);st.session_state.status='Analysis Complete';st.success('Analysis completed successfully.')
                except Exception as e:st.error(f'Analysis failed: {e}')
    st.markdown(f'<div class="status"><b>Status:</b> {st.session_state.status}</div>',unsafe_allow_html=True)

elif st.session_state.page=='Detection Results':
    r=st.session_state.results
    if r is None:st.warning('No analysis has been performed yet.')
    else:
        st.markdown('<div class="section">Detection Results</div>',unsafe_allow_html=True)
        m=st.columns(3)
        for col,num,label in zip(m,[f'{r["change_percentage"]:.2f}%',f'{r["changed_pixels"]:,}',f'{r["total_changed_area"]:.1f}'],['Surface Coverage Changed','Changed Pixels','Estimated Changed Area (m²)']):
            with col:st.markdown(f'<div class="metric"><div class="num">{num}</div><div class="lbl">{label}</div></div>',unsafe_allow_html=True)
        st.markdown('### Detected Change Visualization');a,b=st.columns(2)
        with a:st.markdown('**Land-Cover Detection Map**');st.image(r['class_map'],use_container_width=True)
        with b:st.markdown('**Detection Overlay on After Image**');st.image(r['overlay'],use_container_width=True)
        a,b=st.columns(2)
        with a:st.markdown('**Outlined Detection Result**');st.image(r['outline'],use_container_width=True)
        with b:st.markdown('**Change Probability Heatmap**');st.image(r['attention'],use_container_width=True)
        st.markdown('### Before / After Comparison');a,b=st.columns(2)
        with a:st.caption('Before');st.image(r['before'],use_container_width=True)
        with b:st.caption('After');st.image(r['after'],use_container_width=True)
        st.markdown('### Detection Legend');cols=st.columns(5)
        for col,(cid,name) in zip(cols,CLASS_NAMES.items()):
            color=COLOR_MAP[cid];rgb=f'rgb({color[0]},{color[1]},{color[2]})'
            with col:st.markdown(f'<div style="background:white;border:1px solid #e2e8f0;border-radius:10px;padding:10px;text-align:center"><span style="display:inline-block;width:18px;height:18px;background:{rgb};border-radius:4px;vertical-align:middle"></span> <b>{name}</b></div>',unsafe_allow_html=True)
        st.markdown('### Estimated Change Composition')
        for cid in [1,2,3,4]:
            name=CLASS_NAMES[cid];pct=r['class_percentages'][name];st.write(f'**{name}:** {pct:.2f}% • {r["surface_area"][name]:.2f} m²');st.progress(min(int(round(pct)),100))
        st.caption('Road / vegetation / water / building are visual estimates on detected changed regions. They are not semantic labels learned from the current LEVIR-CD+ binary masks.')

else:
    r=st.session_state.results
    if r is None:st.warning('Run an analysis first.')
    else:
        st.markdown('<div class="section">Analytics & Downloads</div>',unsafe_allow_html=True);a,b=st.columns([2,1])
        with a:
            f=chart(r);st.pyplot(f,use_container_width=True);plt.close(f)
        with b:
            st.markdown('<div class="card"><h3>Operational Log</h3>',unsafe_allow_html=True);st.write(f'{r["total_changed_area"]:.1f} m² ({r["change_percentage"]:.2f}% total change)');st.write('Change mask generated.');st.write('Land-cover estimation completed.');st.markdown('</div>',unsafe_allow_html=True)
        st.markdown('### Surface Area Metrics');a,b,c,d=st.columns(4)
        for col,name in zip([a,b,c,d],['Building','Vegetation','Road','Water Body']):
            with col:st.metric(name,f'{r["surface_area"][name]:.2f} m²')
        st.markdown('### Download Detection Images');buttons=[('Detection Map','class_map','land_cover_detection_map.png'),('Detection Overlay','overlay','detected_changes_overlay.png'),('Outlined Detection','outline','outlined_detection.png'),('Heatmap','attention','change_probability_heatmap.png'),('Difference Map','baseline','difference_map.png'),('Binary Change Mask','mask','change_mask.png')]
        cols=st.columns(3)
        for i,(label,key,fn) in enumerate(buttons):
            with cols[i%3]:
                data=png(r[key])
                if data:st.download_button(f'Download {label}',data,fn,'image/png',use_container_width=True,key=f'dl_{key}')
        f=chart(r);p='geovision_analytics_chart.png';f.savefig(p,dpi=150,bbox_inches='tight',facecolor=f.get_facecolor());plt.close(f)
        with open(p,'rb') as q:st.download_button('Download Analytics Chart',q.read(),p,'image/png',use_container_width=True)

st.markdown('<div class="footer">GeoVision AI Platform • Satellite Image Change Detection</div>',unsafe_allow_html=True)
