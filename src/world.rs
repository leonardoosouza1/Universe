use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::mesh::{Indices, PrimitiveTopology};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum VoxelType { Empty, Wall, Bound }

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum ShapeType { Sphere, Cube, Shield }

#[derive(Resource, Default)]
pub struct VoxelWorldResource(pub VoxelWorld);

pub struct VoxelWorld {
    pub size: usize,
    pub target_shape: ShapeType,
    pub cell_size: f32,
    pub start_pos: Vec3,
}

impl VoxelWorld {
    pub fn new(_seed: u64) -> Self {
        Self { 
            size: 200, 
            target_shape: ShapeType::Shield,
            cell_size: 3.0,
            start_pos: Vec3::new(300.0, 0.0, 300.0),
        }
    }

    pub fn get_shape_dist(&self, pos: Vec3) -> (f32, Vec3) {
        let rel_pos = pos - self.start_pos;
        match self.target_shape {
            ShapeType::Sphere => {
                let r = 80.0;
                let d = rel_pos.length() - r;
                (d, -rel_pos.normalize_or_zero())
            },
            ShapeType::Cube => {
                let size = 60.0;
                let d = rel_pos.abs().max_element() - size;
                let mut dir = Vec3::ZERO;
                if rel_pos.x.abs() > size { dir.x = -rel_pos.x.signum(); }
                if rel_pos.y.abs() > size { dir.y = -rel_pos.y.signum(); }
                if rel_pos.z.abs() > size { dir.z = -rel_pos.z.signum(); }
                (d, dir.normalize_or_zero())
            },
            ShapeType::Shield => {
                let r = 100.0;
                let z_limit = 20.0;
                let dist_to_plane = rel_pos.z.abs() - z_limit;
                let dist_to_center = rel_pos.xy().length();
                let d = (dist_to_center - r).max(dist_to_plane);
                (d, -rel_pos.normalize_or_zero())
            }
        }
    }

    pub fn generate_mesh(&self) -> (Mesh, Mesh, Mesh) {
        let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
        let mut pos = Vec::new();
        let mut color = Vec::new();
        for i in 0..64 {
            let a1 = (i as f32 / 64.0) * 6.28;
            let a2 = ((i+1) as f32 / 64.0) * 6.28;
            let r = 100.0;
            pos.push([self.start_pos.x + a1.cos()*r, 0.0, self.start_pos.z + a1.sin()*r]);
            pos.push([self.start_pos.x + a2.cos()*r, 0.0, self.start_pos.z + a2.sin()*r]);
            color.push([0.0, 1.0, 1.0, 0.2]); color.push([0.0, 1.0, 1.0, 0.2]);
        }
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, color);
        (mesh.clone(), mesh.clone(), mesh)
    }

    pub fn raycast(&self, wx: f32, wz: f32, _angle: f32, _max_dist: f32) -> f32 {
        let dist = Vec2::new(wx - self.start_pos.x, wz - self.start_pos.z).length();
        if dist > 300.0 { return 0.0; }
        100.0
    }
    
    pub fn get_voxel_at(&self, wx: f32, wz: f32) -> VoxelType {
        let dist = Vec2::new(wx - self.start_pos.x, wz - self.start_pos.z).length();
        if dist > 350.0 { VoxelType::Wall } else { VoxelType::Empty }
    }
    
    pub fn get_flow_at(&self, _wx: f32, _wz: f32) -> Vec2 { Vec2::ZERO }
}

impl Default for VoxelWorld {
    fn default() -> Self {
        Self { size: 0, target_shape: ShapeType::Shield, cell_size: 0.0, start_pos: Vec3::ZERO }
    }
}
