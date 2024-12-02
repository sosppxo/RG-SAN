import glob, json, os
import math
import numpy as np
import os.path as osp
import pointgroup_ops
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import torch
import torch_scatter
from torch.utils.data import Dataset
from typing import Dict, Sequence, Tuple, Union
from tqdm import tqdm
from gorilla import is_main_process
import dgl
from scipy import sparse as sp
import sng_parser
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('./backbones/mpnet-base')



def laplacian_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    # A = g.adj_external(scipy_fmt='csr')
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['lap_pos_enc'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 
    
    return g

class ScanNetDataset_sample_graph_edge(Dataset):

    CLASSES = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
               'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture')
    NYU_ID = (3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39)

    def __init__(self,
                 data_root,
                 prefix,
                 suffix,
                 voxel_cfg=None,
                 training=True,
                 with_label=True,
                 mode=4,
                 with_elastic=True,
                 aug=False,
                 use_xyz=True,
                 logger=None,
                 max_des_len=78,
                 graph_pos_enc_dim=5,
                 bidirectional=False,
                 lang_num_max=16,
                 src_sample=-1,
                 scene_graph=False,
                 ):
        self.data_root = data_root
        self.prefix = prefix
        self.suffix = suffix
        self.voxel_cfg = voxel_cfg
        self.training = training
        self.with_label = with_label
        self.mode = mode
        self.with_elastic = with_elastic
        self.aug = aug
        self.use_xyz = use_xyz
        self.logger = logger
        self.max_des_len = max_des_len
        self.bidirectional = bidirectional
        self.scene_graph = scene_graph
        self.depend2id = torch.load(os.path.join(self.data_root, 'dependency_map.pth'))['depend2id']
        self.id2depend = torch.load(os.path.join(self.data_root, 'dependency_map.pth'))['id2depend']
        self.graph_pos_enc_dim = graph_pos_enc_dim
        self.filenames = self.get_graph_filenames()
        self.sp_filenames = self.get_sp_filenames()
        self.src_sample = src_sample
        
        np.random.seed(1999)
        
        # load scanrefer
        if self.prefix == 'train':
            self.scanrefer = json.load(open(os.path.join(self.data_root, 'ScanRefer', 'ScanRefer_filtered_train.json')))
            if is_main_process(): self.logger.info(f'Load {self.prefix} scanrefer: {len(self.scanrefer)} samples')
        elif self.prefix == 'val':
            self.scanrefer = json.load(open(os.path.join(self.data_root, 'ScanRefer', 'ScanRefer_filtered_val.json')))
            if is_main_process(): self.logger.info(f'Load {self.prefix} scanrefer: {len(self.scanrefer)} samples')
        else:
            raise ValueError('ScanRefer only support train and val split, not support %s' % self.prefix)
        
        self.scanrefer.sort(key=lambda x: x['scene_id'])
        scene_list = sorted(list(set([data["scene_id"] for data in self.scanrefer])))
        scanrefer_new = []
        scanrefer_new_scene = []
        scene_id = ""
        for data in self.scanrefer:
            if data["scene_id"] in scene_list:
                if scene_id != data["scene_id"]:
                    scene_id = data["scene_id"]
                    if len(scanrefer_new_scene) > 0:
                        scanrefer_new.append(scanrefer_new_scene)
                    scanrefer_new_scene = []
                if len(scanrefer_new_scene) >= lang_num_max:
                    scanrefer_new.append(scanrefer_new_scene)
                    scanrefer_new_scene = []
                scanrefer_new_scene.append(data)
        scanrefer_new.append(scanrefer_new_scene)
        self.scene_inputs = scanrefer_new
        
        # load lang
        self.load_lang()
        # main(instance seg task) with others
        self.type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
            'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
            'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}  
        self.class2type = {self.type2class[t]:t for t in self.type2class}
        self.nyu40ids = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
        self.nyu40id2class = self._get_nyu40id2class()
        self.sem2nyu = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]


    def _get_type2class_all(self):
        lines = [line.rstrip() for line in open(os.path.join(self.data_root, 'scannetv2-labels.combined.tsv'))]
        lines = lines[1:]
        type2class = {}
        for i in range(len(lines)):
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in self.nyu40ids:
                class_id = np.where(self.nyu40ids == nyu40_id)[0][0]
                type2class[nyu40_name] = class_id
        return type2class
    
    def _get_nyu40id2class(self):
        lines = [line.rstrip() for line in open(os.path.join(self.data_root, 'scannetv2-labels.combined.tsv'))]
        lines = lines[1:]
        nyu40ids2class = {}
        for i in range(len(lines)):
            label_classes_set = set(self.type2class.keys())
            elements = lines[i].split('\t')
            nyu40_id = int(elements[4])
            nyu40_name = elements[7]
            if nyu40_id in self.nyu40ids:
                if nyu40_name not in label_classes_set:
                    nyu40ids2class[nyu40_id] = self.type2class["others"]
                else:
                    nyu40ids2class[nyu40_id] = self.type2class[nyu40_name]
        return nyu40ids2class

    def load_lang(self):
        lang = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = int(data["object_id"])
            ann_id = int(data["ann_id"])

            if scene_id not in lang:
                lang[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
            # load tokens temporarily, and you can extract then load embeddings or features if needed in the future
            lang[scene_id][object_id][ann_id]["token"] = data["token"]
            lang[scene_id][object_id][ann_id]["description"] = data["description"]
        self.lang = lang

    def get_graph_filenames(self):
        if not os.path.exists(osp.join(self.data_root, 'features', self.prefix, 'graph')):
            os.makedirs(osp.join(self.data_root, 'features', self.prefix, 'graph'))
        filenames = glob.glob(osp.join(self.data_root, 'features', self.prefix, 'graph', '*' + str(self.max_des_len).zfill(3) + self.suffix))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)
 
        graphs = {}
        if is_main_process():
            print('loading ' + self.prefix + ' graphs...')
        for filename in tqdm(filenames):
            # source graph filename
            graph_fn = filename
            # dgl filename
            dgl_fn = filename.replace(self.suffix, '.dgl')
            if not osp.exists(graph_fn):
                raise ValueError('Graph file not found: ' + graph_fn)
            
            graph = torch.load(graph_fn)
            heads = graph['heads']
            assert heads[0] == 0, 'ROOT node must be at the beginning'
            tails = graph['tails']
            relations = graph['relations']
            words = graph['words']
            words = [words[i-1] for i in sorted(list(set(tails)))]

            # build dgl graph and save
            if not osp.exists(dgl_fn):
                g = dgl.graph((tails, heads))
                # ROOT node
                token_id = [tokenizer.vocab_size]
                # words without cls token and sep token
                token_id += tokenizer.encode(words, add_special_tokens=False)
                assert len(token_id)==g.num_nodes()
                g.ndata['feat'] = torch.tensor(token_id)
                # edge feat
                relation_id = [self.depend2id[relation] for relation in relations]
                assert len(relation_id)==g.num_edges()
                g.edata['feat'] = torch.tensor(relation_id)
                laplacian_positional_encoding(g, self.graph_pos_enc_dim)
                dgl.save_graphs(dgl_fn, g)

            if self.scene_graph:
                scene_graph = Scene_graph_parse(' '.join(words))
            else:
                scene_graph = None
            graphs.update({osp.basename(filename): 
                            {'graph_file': dgl_fn,
                            'tokens': words,
                            'scene_graph': scene_graph,
                            }
                        })
        self.graphs = graphs
        return filenames

    def get_sp_filenames(self):
        filenames = glob.glob(osp.join(self.data_root, 'scannetv2', self.prefix, '*' + '_refer.pth'))
        assert len(filenames) > 0, 'Empty dataset.'
        filenames = sorted(filenames)
        return filenames
        
    def load(self, filename):
        if self.with_label:
            return torch.load(filename)
        else:
            xyz, rgb, superpoint = torch.load(filename)
            dummy_sem_label = np.zeros(xyz.shape[0], dtype=np.float32)
            dummy_inst_label = np.zeros(xyz.shape[0], dtype=np.float32)
            return xyz, rgb, superpoint, dummy_sem_label, dummy_inst_label
        
    def transform_train(self, xyz, rgb, superpoint, semantic_label, instance_label):
        if self.aug:
            xyz_middle = self.data_aug(xyz, True, True, True)
        else:
            xyz_middle = xyz.copy()
        rgb += np.random.randn(3) * 0.1
        xyz = xyz_middle * self.voxel_cfg.scale
        if self.with_elastic:
            xyz = self.elastic(xyz, 6, 40.)
            xyz = self.elastic(xyz, 20, 160.)
        xyz = xyz - xyz.min(0)
        # xyz, valid_idxs = self.crop(xyz)
        # random sample instead of crop
        valid_idxs = self.sample_rdn(xyz)
        xyz_middle = xyz_middle[valid_idxs]
        xyz = xyz[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = instance_label[valid_idxs]
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def transform_test(self, xyz, rgb, superpoint, semantic_label, instance_label):
        xyz_middle = xyz
        xyz = xyz_middle * self.voxel_cfg.scale
        xyz -= xyz.min(0)
        if self.src_sample > 0:
            # np.random.seed(1184)
            valid_idxs = np.random.choice(
                xyz.shape[0],
                self.src_sample,
                replace=xyz.shape[0] < self.src_sample
            )
        else:
            valid_idxs = np.ones(xyz.shape[0], dtype=bool)
        xyz = xyz[valid_idxs]
        xyz_middle = xyz_middle[valid_idxs]
        rgb = rgb[valid_idxs]
        semantic_label = semantic_label[valid_idxs]
        superpoint = np.unique(superpoint[valid_idxs], return_inverse=True)[1]
        instance_label = instance_label[valid_idxs]
        return xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label

    def data_aug(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(
                m,
                [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)

    def sample_rdn(self, xyz: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        if xyz.shape[0] > self.voxel_cfg.max_npoint:
            valid_idxs = np.random.choice(
                xyz.shape[0],
                self.voxel_cfg.max_npoint,
                replace=xyz.shape[0] < self.voxel_cfg.max_npoint
            )
            return valid_idxs
        else:
            valid_idxs = np.ones(xyz.shape[0], dtype=bool)
            return valid_idxs
    
    def crop(self, xyz: np.ndarray) -> Union[np.ndarray, np.ndarray]:
        r"""
        crop the point cloud to reduce training complexity

        Args:
            xyz (np.ndarray, [N, 3]): input point cloud to be cropped

        Returns:
            Union[np.ndarray, np.ndarray]: processed point cloud and boolean valid indices
        """
        xyz_offset = xyz.copy()
        valid_idxs = xyz_offset.min(1) >= 0
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.voxel_cfg.spatial_shape[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while valid_idxs.sum() > self.voxel_cfg.max_npoint:
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs

    def elastic(self, xyz, gran, mag):
        """Elastic distortion (from point group)

        Args:
            xyz (np.ndarray): input point cloud
            gran (float): distortion param
            mag (float): distortion scalar

        Returns:
            xyz: point cloud with elastic distortion
        """
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(xyz).max(0).astype(np.int32) // gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
        interp = [interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

        def g(xyz_):
            return np.hstack([i(xyz_)[:, None] for i in interp])

        return xyz + g(xyz) * mag
    
    def get_ref_mask(self, coord_float, instance_label, superpoint, object_id):
        sp_coord_float = torch_scatter.scatter_mean(coord_float, superpoint, dim=0)
        ref_lbl = instance_label == object_id
        gt_spmask = torch_scatter.scatter_mean(ref_lbl.float(), superpoint, dim=-1)
        gt_spmask = (gt_spmask > 0.5).float()
        gt_center = torch_scatter.scatter_mean(sp_coord_float, gt_spmask.long(), dim=0)
        if gt_center.shape[0]==1:gt_center=torch.tensor([-1000,-1000,-1000])
        else: gt_center = gt_center[1]
        gt_pmask = ref_lbl.float()
        return gt_pmask, gt_spmask, gt_center
    
    def __len__(self):
        return len(self.scene_inputs)
    
    def __getitem__(self, index: int) -> Tuple:
        ann_ids, object_ids, gt_pmasks, gt_spmasks, sp_ref_masks, gs, lang_tokenss, gt_centers, scene_graphs = [],[],[],[],[],[],[],[],[]
        nsubj_inds, nsubj_names = [], []
        scene_input = self.scene_inputs[index]
        for i in range(len(scene_input)):
            data = scene_input[i]
            scan_id = data['scene_id']
            
            if i==0:
                for fn in self.sp_filenames:
                    if scan_id in fn:
                        sp_filename = fn
                        break
                scene = self.load(sp_filename)
                scene = self.transform_train(*scene) if self.training else self.transform_test(*scene)
                xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label = scene
                coord = torch.from_numpy(xyz).long()
                coord_float = torch.from_numpy(xyz_middle).float()
                feat = torch.from_numpy(rgb).float()
                superpoint = torch.from_numpy(superpoint)
                semantic_label = torch.from_numpy(semantic_label).long()
                instance_label = torch.from_numpy(instance_label).long()
            
            object_id = int(data['object_id'])
            ann_id = int(data['ann_id'])
            lang_tokens = data['token']
            
            context_label = set()
            for word in lang_tokens:
                if word in self.type2class.keys() and word != 'others':
                    context_label.add(self.type2class[word])
            point_context_mask = np.zeros(instance_label.shape[0]) - 1
            for i_instance in np.unique(instance_label):            
                # find all points belong to that instance
                ind = np.where(instance_label == i_instance)[0]
                # find the semantic label            
                if int(semantic_label[ind[0]])>=0:
                    nyu_id = int(self.sem2nyu[int(semantic_label[ind[0]])])
                    if nyu_id in self.nyu40ids and self.nyu40id2class[nyu_id] in context_label:
                        point_context_mask[ind] = 1
            point_ref_mask = np.zeros(instance_label.shape[0])
            # assert len(context_label)==0 or point_context_mask.max()>0, 'no context points'
            point_ref_mask[point_context_mask > 0] = 0.5
            point_ref_mask[instance_label == object_id] = 1
            
            gt_pmask, gt_spmask, gt_center = self.get_ref_mask(coord_float, instance_label, superpoint, object_id)
            point_ref_mask = torch.from_numpy(point_ref_mask).float()
            sp_ref_mask = torch_scatter.scatter_mean(point_ref_mask, superpoint, dim=-1)
            
            filename = os.path.join(self.data_root, 'features', self.prefix, 'graph', scan_id+'_'+str(object_id).zfill(3)+'_'+str(ann_id).zfill(3)+'_max_len_'+str(self.max_des_len).zfill(3)+'.pth')
            g = dgl.load_graphs(self.graphs[osp.basename(filename)]['graph_file'])[0][0]
            lang_tokens = self.graphs[osp.basename(filename)]['tokens']
            
            scene_graph = self.graphs[osp.basename(filename)]['scene_graph']
            
            # find the nsubj token
            relations = [self.id2depend[i.item()] for i in g.edata['feat']]
            sec_root = [j for j in range(len(relations)) if ('ROOT' in relations[j])]
            # nsubj_edge_ind = [j for j in range(len(relations)) if ('compound' in relations[j])or('nsubj' in relations[j])][0]
            try:
                nsubj_edge_ind = [j for j in range(len(relations)) if ('nsubj' in relations[j])][0]
            except: # if no nsubj, use root
                nsubj_edge_ind = sec_root[0]
            nsubj_ind = g.edges()[0][nsubj_edge_ind]
            nsubj_name = lang_tokens[nsubj_ind - 1]
            
            if (len(sec_root)>1 and nsubj_ind > sec_root[1]) or nsubj_name=='that' or nsubj_name=='which': 
                nsubj_ind = g.edges()[0][sec_root[0]]
            nsubj_name = lang_tokens[nsubj_ind - 1]
            # if nsubj is a daici, find the real nsubj
            daici = ['there', 'this', 'it', 'object']
            if nsubj_name in daici:
                nsubj_ind = g.edges()[1][nsubj_ind]
                nsubj_name = lang_tokens[nsubj_ind - 1]
            # if nsubj is a set, find the real nsubj
            if nsubj_name=='set' or nsubj_name=='sets' or nsubj_name=='color':
                try:
                    nsubj_edge_ind = [j for j in range(len(relations)) if ('compound' in relations[j])][0]
                    if 'nmod' in relations[nsubj_edge_ind+1]:
                        nsubj_edge_ind = nsubj_edge_ind+1
                except: # if no nsubj, use root
                    try:
                        nsubj_edge_ind = [j for j in range(len(relations)) if ('nmod' in relations[j])][0]
                    except:
                        try:
                            nsubj_edge_ind = [j for j in range(len(relations)) if ('dep' in relations[j])][0]
                        except:
                            nsubj_edge_ind = sec_root[0]
                nsubj_ind = g.edges()[0][nsubj_edge_ind]
                nsubj_name = lang_tokens[nsubj_ind - 1]
            
            nsubj_inds.append(nsubj_ind)
            nsubj_names.append(nsubj_name)
            ann_ids.append(ann_id)
            object_ids.append(object_id)
            gt_pmasks.append(gt_pmask)
            gt_spmasks.append(gt_spmask)
            sp_ref_masks.append(sp_ref_mask)
            gs.append(g)
            lang_tokenss.append(lang_tokens)
            gt_centers.append(gt_center)
            scene_graphs.append(scene_graph)

        return ann_ids, scan_id, coord, coord_float, feat, superpoint, object_ids, gt_pmasks, gt_spmasks, sp_ref_masks, gs, lang_tokenss, gt_centers, nsubj_inds, nsubj_names, scene_graphs
    
    def collate_fn(self, batch: Sequence[Sequence]) -> Dict:
        ann_ids, scan_ids, coords, coords_float, feats, superpoints, object_ids, gt_pmasks, gt_spmasks, sp_ref_masks, batched_graph, lang_tokenss, lang_masks, lang_words, gt_centers = [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        nsubj_inds, nsubj_names = [], []
        batch_offsets = [0]
        scenes_len = []
        superpoint_bias = 0

        for i, data in enumerate(batch):
            ann_id, scan_id, coord, coord_float, feat, src_superpoint, object_id, gt_pmask, gt_spmask, sp_ref_mask, g, lang_tokens, gt_center, nsubj_ind, nsubj_name, scene_graphs = data
            
            superpoint = src_superpoint + superpoint_bias
            superpoint_bias = superpoint.max().item() + 1
            scenes_len.append(len(ann_id))
            batch_offsets.append(superpoint_bias)

            nsubj_names.extend(nsubj_name)
            nsubj_inds.extend(nsubj_ind)

            ann_ids.extend(ann_id)
            scan_ids.append(scan_id)
            coords.append(torch.cat([torch.LongTensor(coord.shape[0], 1).fill_(i), coord], 1))
            coords_float.append(coord_float)
            feats.append(feat)
            superpoints.append(superpoint)
            
            object_ids.extend(object_id)
            
            gt_pmasks.extend(gt_pmask)
            gt_spmasks.extend(gt_spmask)
            sp_ref_masks.extend(sp_ref_mask)
            
            for txt_id, lang_token in enumerate(lang_tokens):
                if self.scene_graph:
                    scene_graph = scene_graphs[txt_id]
                    graph_node = scene_graph["graph_node"]
                    caption = scene_graph["caption"]
                
                # mpnet
                token_dict = tokenizer(lang_token, is_split_into_words=False, add_special_tokens=True, truncation=True, max_length=self.max_des_len+2, padding='max_length', return_attention_mask=True,return_tensors='pt',)
                token_dict['input_ids'][0][1:len(lang_token)+1] = token_dict['input_ids'][:,1]
                token_dict['input_ids'][0][len(lang_token)+1] = 2
                token_dict['attention_mask'][0][token_dict['input_ids'][0]!=1]=1
                token_dict['input_ids'] = token_dict['input_ids'][0].unsqueeze(0)
                token_dict['attention_mask'] = token_dict['attention_mask'][0].unsqueeze(0)
            
                lang_words.append(lang_token)
                lang_tokenss.append(token_dict['input_ids']) 
                lang_masks.append(token_dict['attention_mask'])

            batched_graph.extend(g)
            gt_centers.extend(gt_center)

        nsubj_inds = torch.stack(nsubj_inds)

        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int [B+1]
        scenes_len = torch.tensor(scenes_len, dtype=torch.int) #int [B]
        coords = torch.cat(coords, 0)  # long [B*N, 1 + 3], the batch item idx is put in b_xyz[:, 0]
        coords_float = torch.cat(coords_float, 0)  # float [B*N, 3]
        feats = torch.cat(feats, 0)  # float [B*N, 3]
        superpoints = torch.cat(superpoints, 0).long()  # long [B*N, ]
        if self.use_xyz:
            feats = torch.cat((feats, coords_float), dim=1)
        # voxelize
        spatial_shape = np.clip((coords.max(0)[0][1:] + 1).numpy(), self.voxel_cfg.spatial_shape[0], None)  # long [3]
        voxel_coords, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(coords, len(batch), self.mode)

        lang_tokenss = torch.cat(lang_tokenss, 0)
        lang_masks = torch.cat(lang_masks, 0).int()
        # merge all scan in batch
        batched_graph = dgl.batch(batched_graph)
        gt_centers = torch.stack(gt_centers, 0)

        return {
            'ann_ids': ann_ids,
            'scan_ids': scan_ids,
            'voxel_coords': voxel_coords,
            'p2v_map': p2v_map,
            'v2p_map': v2p_map,
            'spatial_shape': spatial_shape,
            'feats': feats,
            'superpoints': superpoints,
            'batch_offsets': batch_offsets,
            'object_ids': object_ids,
            'gt_pmasks': gt_pmasks,
            'gt_spmasks': gt_spmasks,
            'sp_ref_masks': sp_ref_masks,
            'batched_graph': batched_graph,
            'lang_tokenss': lang_tokenss,
            'lang_masks': lang_masks,
            'coords_float': coords_float,
            'gt_centers': gt_centers,
            'scenes_len': scenes_len,
            'nsubj_inds': nsubj_inds,
        }



#########################
# BRIEF Text decoupling #
#########################
def Scene_graph_parse(caption):
    caption = ' '.join(caption.replace(',', ' , ').split())

    # some error or typo in ScanRefer.
    caption = ' '.join(caption.replace("'m", "am").split())
    caption = ' '.join(caption.replace("'s", "is").split())
    caption = ' '.join(caption.replace("2-tiered", "2 - tiered").split())
    caption = ' '.join(caption.replace("4-drawers", "4 - drawers").split())
    caption = ' '.join(caption.replace("5-drawer", "5 - drawer").split())
    caption = ' '.join(caption.replace("8-hole", "8 - hole").split())
    caption = ' '.join(caption.replace("7-shaped", "7 - shaped").split())
    caption = ' '.join(caption.replace("2-door", "2 - door").split())
    caption = ' '.join(caption.replace("3-compartment", "3 - compartment").split())
    caption = ' '.join(caption.replace("computer/", "computer /").split())
    caption = ' '.join(caption.replace("3-tier", "3 - tier").split())
    caption = ' '.join(caption.replace("3-seater", "3 - seater").split())
    caption = ' '.join(caption.replace("4-seat", "4 - seat").split())
    caption = ' '.join(caption.replace("theses", "these").split())
    # text parsing
    graph_node, graph_edge = sng_parser.parse(caption)

    # # NOTE If no node is parsed, add "this is an object ." at the beginning of the sentence
    # if (len(graph_node) < 1) or \
    #     (len(graph_node) > 0 and graph_node[0]["node_id"] != 0):
    #     caption = "This is an object . " + caption
    #     # parse again
    #     graph_node, graph_edge = sng_parser.parse(caption)


    # auxi object
    auxi_entity = None
    for node in graph_node:
        if (node["node_id"] != 0) and (node["node_type"] == "Object"):
            auxi_entity = node
            break
    
    return {
        "graph_node": graph_node,
        "graph_edge": graph_edge,
        "auxi_entity": auxi_entity,
        "caption": caption
    }
    