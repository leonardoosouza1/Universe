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
    pub height_map: Vec<f32>,   // Elevação do terreno (metros)
    pub banking: Vec<f32>,      // Inclinação lateral (radianos)
    pub cell_size: f32,
    pub start_pos: Vec3,
    pub lap_distance: f32,
    pub waypoints_world: Vec<Vec2>,      // Waypoints já em coordenadas de mundo (x,z)
    pub waypoint_arcs: Vec<f32>,         // Arco cumulativo por waypoint (arcs[i] = comprimento de wp[0] até wp[i])
    pub start_progress: f32,             // Arc-length da start_pos projetada (cacheado em new() — constante)
}

impl VoxelWorld {
    pub fn new(_seed: u64, track_width: f32) -> Self {
        let size = 160; 
        let cell_size = 3.0;
        let mut grid = vec![VoxelType::Grass; size * size];
        let mut flow = vec![Vec2::ZERO; size * size];
        let mut height_map = vec![0.0f32; size * size];
        let mut banking = vec![0.0f32; size * size];
        
        // TODO: Usar a seed para randomizar os waypoints se complexity > 0
        let waypoints = vec![
            Vec2::new(20.0, 20.0), Vec2::new(140.0, 20.0), 
            Vec2::new(140.0, 60.0), Vec2::new(100.0, 100.0), 
            Vec2::new(100.0, 140.0), Vec2::new(60.0, 140.0), 
            Vec2::new(20.0, 100.0), Vec2::new(20.0, 20.0),
        ];

        let track_width = track_width; 
        for y in 0..size {
            for x in 0..size {
                let pos = Vec2::new(x as f32, y as f32);
                let mut min_dist = f32::MAX;
                let mut best_dir = Vec2::ZERO;
                let mut best_seg_idx = 0usize;

                for i in 0..waypoints.len() - 1 {
                    let d = dist_to_segment(pos, waypoints[i], waypoints[i+1]);
                    if d < min_dist { 
                        min_dist = d; 
                        best_dir = (waypoints[i+1] - waypoints[i]).normalize_or_zero();
                        best_seg_idx = i;
                    }
                }

                let idx = y * size + x;
                flow[idx] = best_dir;
                grid[idx] = if min_dist < track_width / 2.0 { VoxelType::Road }
                            else if min_dist < (track_width / 2.0) + 1.5 { VoxelType::Wall }
                            else { VoxelType::Grass };
                
                // Elevação: ondulações suaves baseadas na posição + segmento
                let elev_base = ((x as f32 * 0.03).sin() * 1.5 + (y as f32 * 0.02).cos() * 2.0) * 0.5;
                height_map[idx] = if grid[idx] == VoxelType::Road || grid[idx] == VoxelType::Wall {
                    elev_base
                } else {
                    elev_base - 0.1 // Grama ligeiramente mais baixa
                };
                
                // Banking: inclinação em curvas (segmentos 2,3,5,6 são curvas)
                if grid[idx] == VoxelType::Road && (best_seg_idx == 2 || best_seg_idx == 3 || best_seg_idx == 5 || best_seg_idx == 6) {
                    let norm_dist = min_dist / (track_width / 2.0); // 0=centro, 1=borda
                    banking[idx] = 0.12 * (1.0 - norm_dist); // ~7° de banking no centro das curvas
                }
            }
        }

        let waypoints_world: Vec<Vec2> = waypoints.iter().map(|p| *p * cell_size).collect();
        let mut waypoint_arcs = vec![0.0];
        let mut total_dist = 0.0;
        for i in 0..waypoints_world.len() - 1 {
            total_dist += waypoints_world[i].distance(waypoints_world[i+1]);
            waypoint_arcs.push(total_dist);
        }

        let start_pos = Vec3::new(120.0 * cell_size, 0.5, 20.0 * cell_size);

        // Pré-calcula start_progress uma vez — antes era refeito a cada get_track_progress
        // (500 carros × 60 Hz × 10× time_scale = 300k recálculos/s de uma constante).
        let start_pos_2d = Vec2::new(start_pos.x, start_pos.z);
        let mut start_progress = 0.0;
        let mut min_start_dist = f32::MAX;
        for i in 0..waypoints_world.len() - 1 {
            let p1 = waypoints_world[i];
            let p2 = waypoints_world[i+1];
            let seg_len = p1.distance(p2);
            let mut dir = p2 - p1;
            if seg_len > 0.0 { dir /= seg_len; }
            let v = start_pos_2d - p1;
            let t = v.dot(dir).clamp(0.0, seg_len);
            let proj = p1 + dir * t;
            let dist = start_pos_2d.distance(proj);
            if dist < min_start_dist { min_start_dist = dist; start_progress = waypoint_arcs[i] + t; }
        }

        Self {
            size, grid, flow, height_map, banking, cell_size,
            start_pos,
            lap_distance: total_dist,
            waypoints_world,
            waypoint_arcs,
            start_progress,
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

    /// Calcula a distância absoluta na pista a partir da linha de largada
    pub fn get_track_progress(&self, wx: f32, wz: f32) -> f32 {
        let pos = Vec2::new(wx, wz);
        let mut min_dist = f32::MAX;
        let mut best_progress = 0.0;
        
        for i in 0..self.waypoints_world.len() - 1 {
            let p1 = self.waypoints_world[i];
            let p2 = self.waypoints_world[i+1];
            let seg_len = p1.distance(p2);
            
            let mut dir = p2 - p1;
            if seg_len > 0.0 { dir /= seg_len; }
            
            let v = pos - p1;
            let t = v.dot(dir).clamp(0.0, seg_len);
            
            let proj = p1 + dir * t;
            let dist = pos.distance(proj);
            
            if dist < min_dist {
                min_dist = dist;
                best_progress = self.waypoint_arcs[i] + t;
            }
        }
        
        // Arc-length relativo à linha de largada. NÃO fazemos wrap aqui: contagem de voltas
        // é feita em simulation.rs (cruzamento de lap_line). Se o carro está atrás da largada
        // (spawn grid recuado, ou andou pra trás), retornamos 0 — jamais um valor enorme falso.
        // start_progress agora é campo cacheado (é constante pra um dado mundo).
        (best_progress - self.start_progress).max(0.0)
    }

    /// Retorna a altura do terreno em coordenadas mundo (com interpolação bilinear)
    pub fn get_height_at(&self, wx: f32, wz: f32) -> f32 {
        let fx = wx / self.cell_size;
        let fz = wz / self.cell_size;
        let x0 = fx.floor() as i32;
        let z0 = fz.floor() as i32;
        let tx = fx - x0 as f32;
        let tz = fz - z0 as f32;
        
        let h = |x: i32, z: i32| -> f32 {
            if x >= 0 && x < self.size as i32 && z >= 0 && z < self.size as i32 {
                self.height_map[z as usize * self.size + x as usize]
            } else { 0.0 }
        };
        
        let h00 = h(x0, z0); let h10 = h(x0 + 1, z0);
        let h01 = h(x0, z0 + 1); let h11 = h(x0 + 1, z0 + 1);
        
        let top = h00 * (1.0 - tx) + h10 * tx;
        let bot = h01 * (1.0 - tx) + h11 * tx;
        top * (1.0 - tz) + bot * tz
    }

    /// Retorna a normal da superfície (para banking/inclinação)
    pub fn get_surface_normal(&self, wx: f32, wz: f32) -> Vec3 {
        let step = self.cell_size;
        let hc = self.get_height_at(wx, wz);
        let hx = self.get_height_at(wx + step, wz);
        let hz = self.get_height_at(wx, wz + step);
        let dx = Vec3::new(step, hx - hc, 0.0);
        let dz = Vec3::new(0.0, hz - hc, step);
        dz.cross(dx).normalize_or_zero()
    }

    pub fn generate_mesh(&self) -> (Mesh, Mesh, Mesh) {
        let mut road_pos = Vec::new();
        let mut road_colors = Vec::new();
        let mut road_indices = Vec::new();
        let mut grass_pos = Vec::new();
        let mut grass_colors = Vec::new();
        let mut grass_indices = Vec::new();
        let mut wall_pos = Vec::new();
        let mut wall_indices = Vec::new();

        for y in 0..self.size {
            for x in 0..self.size {
                let v = self.grid[y * self.size + x];
                let px = x as f32 * self.cell_size;
                let pz = y as f32 * self.cell_size;
                let py = self.height_map[y * self.size + x];

                if v == VoxelType::Road {
                    let is_start_line = (px - self.start_pos.x).abs() < self.cell_size && (pz - self.start_pos.z).abs() < 25.0;
                    let b = road_pos.len() as u32;
                    road_pos.extend_from_slice(&[
                        [px, py + 0.01, pz], [px+self.cell_size, py + 0.01, pz], 
                        [px, py + 0.01, pz+self.cell_size], [px+self.cell_size, py + 0.01, pz+self.cell_size]
                    ]);
                    
                    let is_light = ((x / 2) + (y / 2)) % 2 == 0;
                    let color = if is_start_line { [1.0, 1.0, 1.0, 1.0] } 
                                else if is_light { [0.22, 0.22, 0.25, 1.0] } 
                                else { [0.15, 0.15, 0.18, 1.0] };
                    
                    road_colors.extend_from_slice(&[color; 4]);
                    road_indices.extend_from_slice(&[b, b+2, b+1, b+1, b+2, b+3]);
                } else if v == VoxelType::Grass {
                    let b = grass_pos.len() as u32;
                    grass_pos.extend_from_slice(&[
                        [px, py, pz], [px+self.cell_size, py, pz], 
                        [px, py, pz+self.cell_size], [px+self.cell_size, py, pz+self.cell_size]
                    ]);
                    
                    let is_light = ((x / 3) + (y / 3)) % 2 == 0;
                    let color = if is_light { [0.15, 0.35, 0.15, 1.0] } else { [0.10, 0.28, 0.10, 1.0] };
                    grass_colors.extend_from_slice(&[color; 4]);
                    
                    grass_indices.extend_from_slice(&[b, b+2, b+1, b+1, b+2, b+3]);
                } else if v == VoxelType::Wall {
                    let b = wall_pos.len() as u32;
                    let h = 1.0;
                    wall_pos.extend_from_slice(&[
                        [px + self.cell_size * 0.4, 0.0, pz + self.cell_size * 0.4], 
                        [px + self.cell_size * 0.6, 0.0, pz + self.cell_size * 0.6], 
                        [px + self.cell_size * 0.4, h, pz + self.cell_size * 0.4], 
                        [px + self.cell_size * 0.6, h, pz + self.cell_size * 0.6]
                    ]);
                    wall_indices.extend_from_slice(&[b, b+2, b+1, b+1, b+2, b+3]);
                }
            }
        }

        let mut r_mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        r_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, road_pos);
        r_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, road_colors);
        r_mesh.insert_indices(Indices::U32(road_indices));

        let mut g_mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        g_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, grass_pos);
        g_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, grass_colors);
        g_mesh.insert_indices(Indices::U32(grass_indices));

        let mut w_mesh = Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
        w_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, wall_pos);
        w_mesh.insert_indices(Indices::U32(wall_indices));

        (r_mesh, g_mesh, w_mesh)
    }

    /// VISÃO ULTRA-PRECISA (0.5m)
    pub fn raycast(&self, wx: f32, wz: f32, angle: f32, max_dist: f32) -> f32 {
        let mut d = 0.0;
        while d < max_dist {
            let px = wx + angle.cos() * d;
            let pz = wz + angle.sin() * d;
            let voxel = self.get_voxel_at(px, pz);
            if voxel == VoxelType::Wall || voxel == VoxelType::Grass { 
                return d; 
            }
            d += 0.5;
        }
        max_dist
    }
    /// LÍDAR ÓPTICO (Visão Cíclope Linear HD com Raycast DDA: 32x2 pixels abrangendo 100 Graus Uniformes)
    pub fn render_binocular_retina(&self, pos: Vec3, forward: Vec3, right: Vec3) -> [i8; 64] {
        let mut pixels = [-128; 64];
        let eye_pos = pos + Vec3::Y * 0.4;
        let cell = self.cell_size;

        // Retina dual-scale:
        //   Row 0 = FÓVEA ESTREITA (FOV 50°, pitch -0.04): detalhe alto do que está à frente.
        //   Row 1 = PERIFÉRICA LARGA (FOV 150°, pitch 0.0): vê curva chegando e paredes laterais.
        for row in 0..2 {
            let fov = if row == 0 { 50.0_f32.to_radians() } else { 150.0_f32.to_radians() };
            let pitch = if row == 0 { -0.04_f32 } else { 0.0_f32 };
            let dir_y = pitch.sin();
            let cos_pitch = pitch.cos();
            
            for i in 0..32 {
                let index = (row * 32) + i;
                let normalized_i = (i as f32 / 31.0) * 2.0 - 1.0;
                
                // Distribuição Mapeada de forma 100% Linear. Precisão impecável em curvas.
                let yaw_angle = normalized_i * (fov / 2.0);
                
                let dir_x = (forward.x * yaw_angle.cos() + right.x * yaw_angle.sin()) * cos_pitch;
                let dir_z = (forward.z * yaw_angle.cos() + right.z * yaw_angle.sin()) * cos_pitch;
                
                let mut hit_voxel = VoxelType::Road;
                let mut hit_dist = 150.0;
                
                let dist_to_ground = if dir_y < -0.001 { -eye_pos.y / dir_y } else { 1000.0 };
                
                // Implementação do Algoritmo DDA (Digital Differential Analyzer) - Precisão Metafísica O(1)
                let ray_start = Vec2::new(eye_pos.x / cell, eye_pos.z / cell);
                let ray_dir = Vec2::new(dir_x, dir_z).normalize_or_zero();
                
                let mut map_x = ray_start.x.floor() as i32;
                let mut map_y = ray_start.y.floor() as i32;
                
                let delta_dist_x = if ray_dir.x.abs() < 1e-5 { 1e30 } else { (1.0 / ray_dir.x).abs() };
                let delta_dist_y = if ray_dir.y.abs() < 1e-5 { 1e30 } else { (1.0 / ray_dir.y).abs() };
                
                let step_x: i32; let step_y: i32;
                let mut side_dist_x: f32; let mut side_dist_y: f32;
                
                if ray_dir.x < 0.0 {
                    step_x = -1; side_dist_x = (ray_start.x - map_x as f32) * delta_dist_x;
                } else {
                    step_x = 1; side_dist_x = ((map_x as f32 + 1.0) - ray_start.x) * delta_dist_x;
                }
                
                if ray_dir.y < 0.0 {
                    step_y = -1; side_dist_y = (ray_start.y - map_y as f32) * delta_dist_y;
                } else {
                    step_y = 1; side_dist_y = ((map_y as f32 + 1.0) - ray_start.y) * delta_dist_y;
                }
                
                let mut hit = false;
                let mut side = 0; // 0 = hit X wall, 1 = hit Z wall
                
                // Limite de alcance ~150 metros (75 blocos de 2m)
                for _ in 0..75 {
                    if side_dist_x < side_dist_y {
                        side_dist_x += delta_dist_x; map_x += step_x; side = 0;
                    } else {
                        side_dist_y += delta_dist_y; map_y += step_y; side = 1;
                    }
                    
                    let voxel = self.get_voxel_at((map_x as f32) * cell, (map_y as f32) * cell);
                    if voxel == VoxelType::Wall {
                        hit_voxel = voxel; hit = true; break;
                    }
                }
                
                if hit {
                    let current_dist = (if side == 0 { (map_x as f32 - ray_start.x + (1 - step_x) as f32 / 2.0) / ray_dir.x } 
                    else { (map_y as f32 - ray_start.y + (1 - step_y) as f32 / 2.0) / ray_dir.y }) * cell;
                    
                    if dir_y < -0.001 && current_dist > dist_to_ground {
                        // O raio bateu no chão ANTES de atingir a parede lateral DDA
                        hit_dist = dist_to_ground;
                        let hit_px = eye_pos.x + dir_x * hit_dist;
                        let hit_pz = eye_pos.z + dir_z * hit_dist;
                        hit_voxel = self.get_voxel_at(hit_px, hit_pz);
                    } else {
                        hit_dist = current_dist;
                    }
                } else if dir_y < -0.001 && dist_to_ground <= 150.0 {
                    // Sem paredes, mas bateu no chão
                    hit_dist = dist_to_ground;
                    let hit_px = eye_pos.x + dir_x * hit_dist;
                    let hit_pz = eye_pos.z + dir_z * hit_dist;
                    hit_voxel = self.get_voxel_at(hit_px, hit_pz);
                }
                
                let depth_fade = (hit_dist / 150.0).clamp(0.0, 1.0);
                
                // --- VISÃO ABSOLUTA ---
                // Para que o modelo NUNCA confunda uma parede longe com a pista perto.
                let pixel_val = match hit_voxel {
                    VoxelType::Wall => {
                        // Parede DEVE gritar "luz" o mais forte possível (perigo)
                        // Vai de 127 (perto) até 60 (longe) — Sempre Positivo.
                        127.0 - (depth_fade * 67.0)
                    },
                    VoxelType::Grass => {
                        // Grama é neutra/ruim
                        // Vai de 40 (perto) a 0 (longe)
                        40.0 - (depth_fade * 40.0)
                    },
                    VoxelType::Road => {
                        // Pista grita "escuro" (Segurança)
                        // Vai de -128 (perto) até -80 (longe).
                        -128.0 + (depth_fade * 48.0)
                    }
                };
                
                pixels[index] = pixel_val.clamp(-128.0, 127.0) as i8;
            }
        }
        pixels
    }
}

fn dist_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

impl Default for VoxelWorld {
    fn default() -> Self {
        Self { 
            size: 0, 
            grid: Vec::new(), 
            flow: Vec::new(),
            height_map: Vec::new(),
            banking: Vec::new(),
            cell_size: 0.0,
            start_pos: Vec3::ZERO,
            lap_distance: 0.0,
            waypoints_world: Vec::new(),
            waypoint_arcs: Vec::new(),
            start_progress: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ref_world() -> VoxelWorld {
        VoxelWorld::new(42, 10.0)
    }

    #[test]
    fn track_progress_never_negative() {
        let w = ref_world();
        // Progresso em coordenadas arbitrárias, incluindo pontos aleatórios, nunca deve ser < 0.
        // (Cláusula do bug de wrap: valores atrás da largada devem dar 0, não lap_distance).
        for &(x, z) in &[(0.0, 0.0), (-500.0, -500.0), (1000.0, 1000.0), (50.0, -100.0)] {
            assert!(w.get_track_progress(x, z) >= 0.0, "progress < 0 em ({}, {})", x, z);
        }
    }

    #[test]
    fn lap_distance_is_positive() {
        let w = ref_world();
        assert!(w.lap_distance > 0.0, "lap_distance deve ser positivo para pista real");
    }

    #[test]
    fn track_progress_near_start_is_small() {
        let w = ref_world();
        // Na própria largada, progresso deve ser ~0 (até 2m de tolerância pela quantização do grid).
        let p = w.get_track_progress(w.start_pos.x, w.start_pos.z);
        assert!(p < 2.0, "progresso na largada = {}, esperado ~0", p);
    }
}
