use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::mesh::{Indices, PrimitiveTopology};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum VoxelType { Grass = 0, Road = 1, Wall = 2 }

#[derive(Resource, Default)]
pub struct VoxelWorldResource(pub VoxelWorld);

pub struct VoxelWorld {
    pub size: usize,
    pub grid: Vec<VoxelType>,
    pub flow: Vec<Vec2>,
    pub cell_size: f32,
    pub start_pos: Vec3,
    pub lap_distance: f32,
}

impl VoxelWorld {
    pub fn new(_seed: u64) -> Self {
        let size = 120;
        let cell_size = 2.5;
        let mut grid = vec![VoxelType::Grass; size * size];
        let mut flow = vec![Vec2::ZERO; size * size];
        
        // CIRCUITO FIXO E ESTÁVEL (O QUE FUNCIONAVA)
        let waypoints = vec![
            Vec2::new(10.0, 10.0), Vec2::new(80.0, 10.0),
            Vec2::new(90.0, 40.0), Vec2::new(60.0, 60.0),
            Vec2::new(20.0, 80.0), Vec2::new(10.0, 40.0),
            Vec2::new(10.0, 10.0),
        ];

        let track_width = 10.0;
        for y in 0..size {
            for x in 0..size {
                let pos = Vec2::new(x as f32, y as f32);
                let mut min_dist = f32::MAX;
                let mut best_dir = Vec2::ZERO;

                for i in 0..waypoints.len() - 1 {
                    let d = dist_to_segment(pos, waypoints[i], waypoints[i+1]);
                    if d < min_dist { 
                        min_dist = d; 
                        best_dir = (waypoints[i+1] - waypoints[i]).normalize_or_zero();
                    }
                }

                let idx = y * size + x;
                flow[idx] = best_dir;
                grid[idx] = if min_dist < track_width / 2.0 { VoxelType::Road }
                            else if min_dist < (track_width / 2.0) + 1.0 { VoxelType::Wall }
                            else { VoxelType::Grass };
            }
        }

        Self { 
            size, grid, flow, cell_size,
            start_pos: Vec3::new(10.0 * cell_size, 0.5, 10.0 * cell_size),
            lap_distance: 400.0,
        }
    }

    pub fn get_voxel_at(&self, wx: f32, wz: f32) -> VoxelType {
        let x = (wx / self.cell_size).round() as i32;
        let y = (wz / self.cell_size).round() as i32;
        if x >= 0 && x < self.size as i32 && y >= 0 && y < self.size as i32 {
            self.grid[y as usize * self.size + x as usize]
        } else { VoxelType::Wall }
    }

    pub fn get_flow_at(&self, wx: f32, wz: f32) -> Vec2 {
        let x = (wx / self.cell_size).round() as i32;
        let y = (wz / self.cell_size).round() as i32;
        if x >= 0 && x < self.size as i32 && y >= 0 && y < self.size as i32 {
            self.flow[y as usize * self.size + x as usize]
        } else { Vec2::ZERO }
    }

    pub fn generate_mesh(&self) -> (Mesh, Mesh) {
        let mut road_pos = Vec::new();
        let mut road_colors = Vec::new();
        let mut road_indices = Vec::new();
        let mut wall_pos = Vec::new();
        let mut wall_indices = Vec::new();

        for y in 0..self.size {
            for x in 0..self.size {
                let v = self.grid[y * self.size + x];
                let px = x as f32 * self.cell_size;
                let pz = y as f32 * self.cell_size;

                if v == VoxelType::Road || v == VoxelType::Grass {
                    // Estrada com efeito de "Profundidade Quântica"
                    let color = if v == VoxelType::Road { 
                        [0.02, 0.02, 0.05, 1.0] // Azul Profundo
                    } else { 
                        [0.01, 0.01, 0.02, 1.0] // Quase Preto
                    };
                    
                    let b = road_pos.len() as u32;
                    road_pos.extend_from_slice(&[
                        [px, 0.0, pz], 
                        [px+self.cell_size, 0.0, pz], 
                        [px, 0.0, pz+self.cell_size], 
                        [px+self.cell_size, 0.0, pz+self.cell_size]
                    ]);
                    road_colors.extend_from_slice(&[color; 4]);
                    road_indices.extend_from_slice(&[b, b+2, b+1, b+1, b+2, b+3]);

                    // Adicionar "Bordas de Energia" se for Road adjacente a Wall ou Grass
                    if v == VoxelType::Road {
                        // (Lógica simplificada para bordas neon)
                    }
                } else if v == VoxelType::Wall {
                    let b = wall_pos.len() as u32;
                    let h = 2.5; // Muros de energia mais baixos e elegantes
                    wall_pos.extend_from_slice(&[
                        [px, 0.0, pz], 
                        [px+self.cell_size, 0.0, pz], 
                        [px, h, pz], 
                        [px+self.cell_size, h, pz]
                    ]);
                    wall_indices.extend_from_slice(&[b, b+2, b+1, b+1, b+2, b+3]);
                }
            }
        }

        let mut r_mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        r_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, road_pos);
        r_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, road_colors);
        r_mesh.insert_indices(Indices::U32(road_indices));

        let mut w_mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        w_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, wall_pos);
        w_mesh.insert_indices(Indices::U32(wall_indices));

        (r_mesh, w_mesh)
    }

    pub fn raycast(&self, wx: f32, wz: f32, angle: f32, max_dist: f32) -> f32 {
        let mut d = 0.0;
        while d < max_dist {
            let px = wx + angle.cos() * d;
            let pz = wz + angle.sin() * d;
            if self.get_voxel_at(px, pz) == VoxelType::Wall { return d; }
            d += 1.5;
        }
        max_dist
    }
}

fn dist_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

impl Default for VoxelWorld {
    fn default() -> Self {
        Self { size: 0, grid: Vec::new(), flow: Vec::new(), cell_size: 0.0, start_pos: Vec3::ZERO, lap_distance: 0.0 }
    }
}
