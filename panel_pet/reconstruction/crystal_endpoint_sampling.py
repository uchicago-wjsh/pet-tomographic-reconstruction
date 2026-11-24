"""Independent uniform endpoint sampling across a crystal face and depth."""
from pathlib import Path
import numpy as np
import scanner_lut as sl


class CrystalVolumeSampler:
    def __init__(self, macro: Path, lut: np.ndarray, seed: int, face_sampling=True):
        geom=sl.load_geometry(macro)
        centers=sl.build_scanner_lut(geom,depth_mode='center').astype(np.float64)
        if centers.shape!=lut.shape or not np.allclose(centers,lut,rtol=0,atol=1e-5):
            raise ValueError('Geometry-derived crystal centers differ from the event LUT')
        self.centers=centers
        self.face=sl.build_scanner_lut(geom,depth_mode='frontface').astype(np.float64)
        self.depth=float(geom.crystal_depth_mm)
        self.normal=(centers-self.face)/(self.depth/2)
        plane=[a for a in range(3) if a!=geom.depth_axis]
        self.axis1=np.repeat(geom.module_rotations@np.eye(3)[plane[0]],geom.n_crystals_per_module,axis=0)
        self.axis2=np.repeat(geom.module_rotations@np.eye(3)[plane[1]],geom.n_crystals_per_module,axis=0)
        self.half1=float(geom.crystal_size_mm[plane[0]])/2
        self.half2=float(geom.crystal_size_mm[plane[1]])/2
        basis=np.stack([self.axis1,self.axis2,self.normal],axis=2)
        if not np.allclose(np.einsum('dji,djk->dik',basis,basis),np.eye(3),atol=1e-6):
            raise ValueError('Crystal face/depth basis is not orthonormal')
        self.seed=int(seed)
        self.face_sampling=bool(face_sampling)
        self.rng=[np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
        self.stats=[dict(n=0,minimum=np.full(3,np.inf),maximum=np.full(3,-np.inf),
                         total=np.zeros(3),total_square=np.zeros(3)) for _ in range(2)]
        self.audit=[]

    def sample(self, ids: np.ndarray, arm: int):
        if self.face_sampling:
            draws=self.rng[arm].random((len(ids),3))
            local=draws*np.array([2*self.half1,2*self.half2,self.depth])-np.array([self.half1,self.half2,0.])
        else:
            local=np.zeros((len(ids),3),dtype=np.float64)
            local[:,2]=self.rng[arm].random(len(ids))*self.depth
        points=(self.face[ids]+self.axis1[ids]*local[:,0,None]
                +self.axis2[ids]*local[:,1,None]+self.normal[ids]*local[:,2,None])
        # Verify all endpoints in their own crystal's local box, not only a
        # scanner bounding box that would miss rotated or misindexed frames.
        delta=points-self.centers[ids]
        recovered=np.column_stack([np.einsum('ij,ij->i',delta,a[ids])
                                    for a in (self.axis1,self.axis2,self.normal)])
        half=np.array([self.half1,self.half2,self.depth/2])
        if not np.isfinite(points).all() or (np.abs(recovered)>half+1e-7).any():
            raise AssertionError('Sampled endpoint escaped its assigned crystal')
        stats=self.stats[arm];stats['n']+=len(ids)
        stats['minimum']=np.minimum(stats['minimum'],local.min(axis=0))
        stats['maximum']=np.maximum(stats['maximum'],local.max(axis=0))
        stats['total']+=local.sum(axis=0);stats['total_square']+=(local**2).sum(axis=0)
        if len(self.audit)<2:
            self.audit.append(dict(arm=arm,ids=ids[:2048].copy(),points=points[:2048].copy(),local=local[:2048].copy()))
        return points

    def save(self, out: Path):
        np.savez_compressed(out/'crystal_endpoint_basis.npz',centers=self.centers,
                            front_face_center=self.face,axis1=self.axis1,axis2=self.axis2,
                            depth_normal=self.normal,half1=self.half1,half2=self.half2,
                            depth_mm=self.depth)
        blocks={}
        for entry in self.audit:
            for key in ('ids','points','local'):blocks[f'arm{entry["arm"]}_{key}']=entry[key]
        np.savez_compressed(out/'endpoint_sample_audit.npz',**blocks)
        rows=[]
        for stats in self.stats:
            mean=stats['total']/stats['n']
            std=np.sqrt(stats['total_square']/stats['n']-mean**2)
            rows.append(dict(n=stats['n'],minimum_mm=stats['minimum'].tolist(),
                             maximum_mm=stats['maximum'].tolist(),mean_mm=mean.tolist(),
                             std_mm=std.tolist()))
        return {'model':'uniform crystal-face position plus uniform depth' if self.face_sampling else 'face center plus uniform depth only',
                'face_sampling':self.face_sampling,
                'face_size_mm':[2*self.half1,2*self.half2],'depth_mm':self.depth,
                'seed':self.seed,'samples_per_event':1,'independent_arms':True,
                'depth_distribution':'Uniform(0, depth_mm), measured from the entrance face',
                'rng':'numpy default_rng; separate SeedSequence child for each arm',
                'all_sampled_endpoints_checked_inside_assigned_crystal':True,
                'coordinates':'two in-face coordinates and outward depth from the entrance face',
                'arm_statistics':rows}
