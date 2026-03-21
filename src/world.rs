use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use crate::kan::KanAxiomFilter;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum VoxelType { Empty, Wall, EnergyField }

#[derive(Resource, Default)]
pub struct VoxelWorldResource(pub VoxelWorld);

pub struct VoxelWorld {
    pub size: usize,
    pub cell_size: f32,
    pub start_pos: Vec3,
    pub axiom_filter: KanAxiomFilter,
}

impl VoxelWorld {
    pub fn new(_seed: u64) -> Self {
        Self { 
            size: 400, 
            cell_size: 5.0,
            start_pos: Vec3::new(0.0, 0.0, 0.0),
            axiom_filter: KanAxiomFilter::new(),
        }
    }

    /// O Universo vibra em uma frequência B-Spline.
    /// O agente deve ressonar com esta frequência para "ver" o caminho.
    pub fn get_frequency_at(&self, pos: Vec3) -> f32 {
        let rel_x = (pos.x - self.start_pos.x) * 0.01;
        let rel_z = (pos.z - self.start_pos.z) * 0.01;
        
        // Usamos o filtro KAN para gerar a "assinatura de energia" do local
        let base_freq = (rel_x.sin() + rel_z.cos()) * 0.5;
        self.axiom_filter.filter_pulse(base_freq)
    }

    pub fn get_voxel_at(&self, pos: Vec3) -> VoxelType {
        let freq = self.get_frequency_at(pos);
        if freq.abs() > 0.8 {
            VoxelType::Wall // Paredes de Energia KAN
        } else if freq.abs() > 0.4 {
            VoxelType::EnergyField
        } else {
            VoxelType::Empty
        }
    }

    pub fn generate_mesh(&self) -> (Mesh, Mesh, Mesh) {
        let mut mesh = Mesh::new(PrimitiveTopology::LineList, RenderAssetUsages::default());
        let mut pos = Vec::new();
        let mut color = Vec::new();
        
        let samples = 1200;
        let phi = std::f32::consts::PI * (3.0 - (5.0f32).sqrt()); // Ângulo de Ouro
        let r_max = 500.0;

        for i in 0..samples {
            let n = i as f32;
            let radius = 20.0 * n.sqrt(); // Espiral de Fermat (Fibonacci)
            if radius > r_max { break; }
            let theta = n * phi;

            let px = self.start_pos.x + theta.cos() * radius;
            let pz = self.start_pos.z + theta.sin() * radius;
            
            // A altura Y é determinada pela frequência KAN do local
            let py = self.get_frequency_at(Vec3::new(px, 0.0, pz)) * 50.0;

            pos.push([px, py, pz]);
            pos.push([px, -py, pz]); // Simetria Metafísica
            
            let freq = self.get_frequency_at(Vec3::new(px, 0.0, pz));
            let intensity = (freq + 1.0) / 2.0;
            color.push([0.0, intensity, 1.0, 0.2]); 
            color.push([1.0, 0.0, intensity, 0.1]); 
        }
        
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, pos);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, color);
        (mesh.clone(), mesh.clone(), mesh)
    }

    pub fn raycast(&self, origin: Vec3, dir: Vec3, max_dist: f32) -> f32 {
        let mut d = 0.0;
        let step = 2.0;
        while d < max_dist {
            let p = origin + dir * d;
            if self.get_voxel_at(p) == VoxelType::Wall { return d; }
            d += step;
        }
        max_dist
    }
}

impl Default for VoxelWorld {
    fn default() -> Self {
        Self { 
            size: 0, 
            cell_size: 0.0, 
            start_pos: Vec3::ZERO,
            axiom_filter: KanAxiomFilter::new()
        }
    }
}
