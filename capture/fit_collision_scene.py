"""Fit conservative collision boxes from registered desk image correspondences.

The polygons below are manually selected in 0004.jpg. They identify rigid
surfaces, not semantic labels inferred by a model. The table-frame transform
is the same one used by the trained Gaussian asset. Boxes enclose panel tilt
because the current scene contract supports yaw but not pitch/roll.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
POLYGONS = {
    'upper_screen': [[130,166],[267,145],[271,230],[144,250]],
    'lower_screen': [[148,254],[263,237],[270,302],[158,322]],
    'laptop_base': [[159,328],[274,307],[315,360],[183,398]],
    'left_screen': [[7,315],[109,274],[136,344],[41,385]],
    'left_base': [[45,372],[124,337],[164,393],[75,423]],
}


def fit_scene():
    recon = ROOT / 'runs/splat-local'
    frame = json.loads((ROOT/'runs/photoreal-20261025/table-frame.json').read_text())
    basis = np.array([frame['x'], frame['y'], frame['up']]).T
    origin, scale = np.array(frame['origin']), frame['scale']
    lines = [l for l in (recon/'colmap_txt/images.txt').read_text().splitlines() if not l.startswith('#')]
    row = lines[0].split()
    assert row[-1] == '0004.jpg'
    uv = np.array(lines[1].split(),float).reshape(-1,3)
    points = {}
    for line in (recon/'colmap_txt/points3D.txt').read_text().splitlines():
        if line and not line.startswith('#'):
            a=line.split();points[int(a[0])] = np.array(a[1:4],float)
    q=np.array(row[1:5],float)
    rotation=Rotation.from_quat([*q[1:],q[0]]).as_matrix()
    camera=-rotation.T@np.array(row[5:8],float)
    k=np.array([[293.78312433716656,0,178],[0,293.49636299156936,316.5],[0,0,1.]])
    body=json.loads((ROOT/'examples/desk_live_base.json').read_text())
    body['scene']['objects']=[body['scene']['objects'][0]]
    fits={}
    for name, polygon in POLYGONS.items():
        poly=np.array(polygon,float);a=np.roll(poly,-1,axis=0)-poly;b=uv[:,:2,None].transpose(0,2,1)-poly
        cross=a[:,0]*b[:,:,1]-a[:,1]*b[:,:,0]
        inside=(cross>=0).all(1)|(cross<=0).all(1)
        ids=np.unique(uv[inside & (uv[:,2]>=0),2].astype(int))
        p=np.array([points[i] for i in ids])
        if len(p)<4:raise ValueError(f'{name}: only {len(p)} tracked points')
        rng=np.random.default_rng(1);best=[]
        for _ in range(2000):
            a,b,c=p[rng.choice(len(p),3,replace=False)];n=np.cross(b-a,c-a)
            norm=np.linalg.norm(n)
            if norm<1e-8:continue
            n/=norm
            hit=np.flatnonzero(np.abs((p-a)@n)*scale < .008)
            if len(hit)>len(best):best=hit
        center=p[best].mean(0);_,_,v=np.linalg.svd(p[best]-center);normal=v[-1]
        rays=(np.c_[poly,np.ones(4)]@np.linalg.inv(k).T)@rotation
        corners=camera+rays*((center-camera)@normal/(rays@normal))[:,None]
        sim=(corners-origin)@basis*scale
        edge=(sim[1]-sim[0])[:2];yaw=np.arctan2(edge[1],edge[0]);co,si=np.cos(yaw),np.sin(yaw)
        r=np.array([[co,-si],[si,co]])
        xy=sim[:,:2]@r
        low=np.r_[xy.min(0),sim[:,2].min()];high=np.r_[xy.max(0),sim[:,2].max()]
        low-=.008;high+=.008
        if 'base' in name:low[2]=min(low[2],0.0)
        if high[1]-low[1]<.025:low[1]-=.008;high[1]+=.008
        size=high-low;center_local=(low+high)/2;center_xy=center_local[:2]@r.T
        obj={'name':name,'kind':'box','size_m':size.tolist(),'pos_m':[float(center_xy[0]),float(center_xy[1]),float(low[2])],'yaw_deg':float(np.rad2deg(yaw)),'mass_kg':.3,'graspable':False}
        body['scene']['objects'].append(obj)
        fits[name]={'tracked_points':len(p),'plane_inliers':len(best),'plane_rms_m':float(np.sqrt(np.mean(((p[best]-center)@normal*scale)**2))),'corners_m':sim.tolist(),'box':obj}
    body['scene']['left_arm_base_m']=[-.55,-.25,0]
    body['scene']['right_arm_base_m']=[.55,-.25,0]
    body['scene']['objects'][0]['pos_m']=[-.15,-.28,0]
    body['scene']['target_m']=[.24,-.23,0]
    body['plan']['handover_point_m']=[0,-.28,.25]
    body['scene']['evidence'] += ['Screen and laptop collision boxes fitted to COLMAP correspondences in 0004.jpg using the trained splat table frame. Manually selected panel corners, 8 mm inflation; boxes conservatively enclose tilt. Unobserved surfaces remain uncertain. Robot pedestals and the virtual task can use the clear foreground workspace; this is a new layout, not an old run with changed graphics.']
    return body,fits

if __name__=='__main__':
    out=ROOT/'runs/collision-surfaces';out.mkdir(exist_ok=True)
    scene,fits=fit_scene();(out/'scene.json').write_text(json.dumps(scene,indent=2));(out/'fits.json').write_text(json.dumps(fits,indent=2))
    for name,f in fits.items():print(name,f['tracked_points'],f['plane_inliers'],round(f['plane_rms_m'],4),f['box'])
