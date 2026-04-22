use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::mesh::{Indices, PrimitiveTopology};

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum VoxelType { Grass = 0, Road = 1, Wall = 2, Hole = 3 }

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
    /// Ideal racing line (outside-in-outside, precomputed once per track).
    /// Used to feed sensor channels 108/109/110 (lateral offset, derivative, lookahead).
    pub racing_line: crate::simulation::racing_line::RacingLine,
}

impl VoxelWorld {
    pub fn new(_seed: u64, track_width: f32) -> Self {
        let size = 160; 
        let cell_size = 3.0;
        let mut grid = vec![VoxelType::Grass; size * size];
        let mut flow = vec![Vec2::ZERO; size * size];
        let mut height_map = vec![0.0f32; size * size];
        let mut banking = vec![0.0f32; size * size];
        
        // FIX: waypoints únicos por seed — cada pista é literalmente diferente,
        // progressão easy→very hard com rampas e desafios geométricos.
        // Seeds batem com TrackProfile em config.rs.
        let waypoints = track_waypoints_for_seed(_seed);
        let (height_amp, height_freq) = track_elevation_for_seed(_seed);

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
                
                // Elevação: amplitude/frequência por pista — Nürburgring Hills tem rampas sérias.
                let elev_base = ((x as f32 * height_freq).sin() * height_amp
                    + (y as f32 * height_freq * 0.67).cos() * height_amp * 1.3) * 0.5;
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

        // Overlay HOLES (VoxelType::Hole) em cells específicas por seed. Só em pistas
        // HARD+ (256, 512, 768, 1024). Holes matam no contato — força rede a aprender
        // desvio ativo, não passivo. Só sobrescreve Road (nunca Wall/Grass).
        // Cada ponto vira bloco 2×2 (6m × 6m) — visível na retina e passível de colisão
        // pelo chassi (2m × 2.6m). Holes isolados de 1 cell eram skipáveis.
        for hole_cell in track_holes_for_seed(_seed) {
            let hx = hole_cell.x.round() as i32;
            let hy = hole_cell.y.round() as i32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let cx = hx + dx;
                    let cy = hy + dy;
                    if cx < 0 || cx >= size as i32 || cy < 0 || cy >= size as i32 { continue; }
                    let idx = cy as usize * size + cx as usize;
                    if grid[idx] == VoxelType::Road {
                        grid[idx] = VoxelType::Hole;
                    }
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

        // Start pos por pista — runway suficiente pra ler primeira curva a ~40m/s.
        let start_pos_cells = track_start_for_seed(_seed);
        let start_pos = Vec3::new(start_pos_cells.x * cell_size, 0.5, start_pos_cells.y * cell_size);

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

        // === DIAGNÓSTICO DE MAPA ===
        // Impresso 1× no boot. Ajuda identificar problemas de design (curvas cegas,
        // start mal colocado, waypoints com segmentos absurdos).
        let n_road = grid.iter().filter(|v| **v == VoxelType::Road).count();
        let n_wall = grid.iter().filter(|v| **v == VoxelType::Wall).count();
        let n_grass = grid.iter().filter(|v| **v == VoxelType::Grass).count();
        let n_hole = grid.iter().filter(|v| **v == VoxelType::Hole).count();
        println!("🗺  MAP DIAG track_width={:.1}m cell={:.1}m lap={:.1}m", track_width * cell_size * 0.0 + track_width, cell_size, total_dist);
        println!("   voxels road={} wall={} grass={} hole={} road_ratio={:.1}%", n_road, n_wall, n_grass, n_hole, 100.0 * n_road as f32 / (n_road + n_wall + n_grass + n_hole) as f32);
        println!("   start_pos=({:.1}, {:.1}) start_progress={:.1}m / lap {:.1}m ({:.1}%)",
            start_pos.x, start_pos.z, start_progress, total_dist, 100.0 * start_progress / total_dist);
        for i in 0..waypoints_world.len() {
            let (seg_len, turn_deg) = if i == 0 {
                (0.0, 0.0)
            } else {
                let prev = waypoints_world[i - 1];
                let cur = waypoints_world[i];
                let sl = prev.distance(cur);
                let turn = if i + 1 < waypoints_world.len() {
                    let next = waypoints_world[i + 1];
                    let a = (cur - prev).normalize_or_zero();
                    let b = (next - cur).normalize_or_zero();
                    let cos_t = a.dot(b).clamp(-1.0, 1.0);
                    cos_t.acos().to_degrees()
                } else { 0.0 };
                (sl, turn)
            };
            let tag = if turn_deg > 60.0 { "⚠ SHARP" } else if turn_deg > 30.0 { "CURVE" } else { "straight" };
            println!("   WP[{}] ({:.0},{:.0}) seg_len={:.1}m turn={:.1}° {}",
                i, waypoints_world[i].x, waypoints_world[i].y, seg_len, turn_deg, tag);
        }

        // Precompute ideal racing line (outside-in-outside). Built once per
        // track — this is the teacher signal for channels 108/109/110.
        let racing_line = crate::simulation::racing_line::RacingLine::from_centerline(
            &waypoints_world,
            track_width,
        );

        Self {
            size, grid, flow, height_map, banking, cell_size,
            start_pos,
            lap_distance: total_dist,
            waypoints_world,
            waypoint_arcs,
            start_progress,
            racing_line,
        }
    }

    /// Renderiza mapa ASCII top-down.
    /// - `.` grass, `=` road, `#` wall
    /// - `0..9` waypoints (overlay)
    /// - `X` death markers (overlay)
    /// - `o` finish positions de racers vivos (overlay)
    pub fn render_ascii_map(
        &self,
        width: usize,
        height: usize,
        deaths: &[Vec3],
        finish_positions: &[Vec3],
    ) -> String {
        let mut buf = vec![vec![b' '; width]; height];
        let world_size = self.size as f32 * self.cell_size;
        for r in 0..height {
            for c in 0..width {
                let wx = (c as f32 + 0.5) / width as f32 * world_size;
                let wz = (r as f32 + 0.5) / height as f32 * world_size;
                buf[r][c] = match self.get_voxel_at(wx, wz) {
                    VoxelType::Grass => b'.',
                    VoxelType::Road => b'=',
                    VoxelType::Wall => b'#',
                    VoxelType::Hole => b'O',
                };
            }
        }
        let to_rc = |wx: f32, wz: f32| -> (usize, usize) {
            let c = ((wx / world_size) * width as f32).clamp(0.0, width as f32 - 1.0) as usize;
            let r = ((wz / world_size) * height as f32).clamp(0.0, height as f32 - 1.0) as usize;
            (r, c)
        };
        for p in finish_positions.iter() {
            let (r, c) = to_rc(p.x, p.z);
            buf[r][c] = b'o';
        }
        for p in deaths.iter() {
            let (r, c) = to_rc(p.x, p.z);
            buf[r][c] = b'X';
        }
        for (i, wp) in self.waypoints_world.iter().take(10).enumerate() {
            let (r, c) = to_rc(wp.x, wp.y);
            buf[r][c] = b'0' + (i as u8 % 10);
        }
        let mut s = String::new();
        for row in buf.iter() {
            s.push_str(std::str::from_utf8(row).unwrap_or(""));
            s.push('\n');
        }
        s
    }

    /// Waypoint-index (0..N-1) mais próximo de ponto mundo. Agrupa mortes por segmento.
    pub fn waypoint_zone(&self, wx: f32, wz: f32) -> usize {
        let pos = Vec2::new(wx, wz);
        let mut min_dist = f32::MAX;
        let mut best_seg = 0;
        for i in 0..self.waypoints_world.len().saturating_sub(1) {
            let p1 = self.waypoints_world[i];
            let p2 = self.waypoints_world[i+1];
            let seg_len = p1.distance(p2).max(0.001);
            let dir = (p2 - p1) / seg_len;
            let t = (pos - p1).dot(dir).clamp(0.0, seg_len);
            let proj = p1 + dir * t;
            let d = pos.distance(proj);
            if d < min_dist { min_dist = d; best_seg = i; }
        }
        best_seg
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
                } else if v == VoxelType::Hole {
                    // Desenha Hole como quad PRETO 0.3m abaixo do road — visualmente
                    // óbvio pro espectador humano e ainda detectável pela retina (hit_dist
                    // volta ~90 no canal). Usa road_pos (mesmo material road) com cor preta.
                    let b = road_pos.len() as u32;
                    road_pos.extend_from_slice(&[
                        [px, py - 0.3, pz], [px+self.cell_size, py - 0.3, pz],
                        [px, py - 0.3, pz+self.cell_size], [px+self.cell_size, py - 0.3, pz+self.cell_size]
                    ]);
                    road_colors.extend_from_slice(&[[0.0, 0.0, 0.0, 1.0]; 4]);
                    road_indices.extend_from_slice(&[b, b+2, b+1, b+1, b+2, b+3]);
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
            if voxel == VoxelType::Wall || voxel == VoxelType::Grass || voxel == VoxelType::Hole {
                return d;
            }
            d += 0.5;
        }
        max_dist
    }
    /// LÍDAR ÓPTICO (Visão Cíclope Linear HD com Raycast DDA: 32x2 pixels abrangendo 100 Graus Uniformes)
    ///
    /// `gaze_yaw` (rad): offset angular da fóvea relativa ao forward. Simula saccade ocular —
    /// a fóvea vira pro setor mais escuro detectado pela periférica. Periférica ignora gaze
    /// (sempre centrada no forward — é o "campo panorâmico" fixo do motorista).
    pub fn render_binocular_retina(&self, pos: Vec3, forward: Vec3, right: Vec3, gaze_yaw: f32) -> [i8; 64] {
        let mut pixels = [-128; 64];
        let eye_pos = pos + Vec3::Y * 0.4;
        let cell = self.cell_size;

        // Retina unificada 180° dual-scale:
        //   Row 0 = FÓVEA TAN-WARP 180° (pitch -0.04, yaw = gaze_yaw): cobertura total 180°
        //           com distribuição NÃO-UNIFORME center-dense (tan warp). ~1.5°/pixel no
        //           centro (precisão pra curvas fechadas), ~15°/pixel nas bordas (cobertura
        //           lateral). Resolve o trade-off antigo: antes era 50° linear (cego lateral)
        //           ou 180° linear (5.6°/px — cego pra curva).
        //   Row 1 = PERIFÉRICA 180° LINEAR (pitch 0.0, yaw fixo): uniforme, serve de motion
        //           detector baseline — row 0 warpada comparada com row 1 linear revela
        //           velocidade angular lateral (útil pra prever curva).
        const TAN_K: f32 = 1.25_f32; // Warp strength: maior k → centro mais denso
        let tan_k_val = TAN_K.tan();
        for row in 0..2 {
            let fov_half = 90.0_f32.to_radians(); // Ambas rows agora 180°
            let pitch = if row == 0 { -0.04_f32 } else { 0.0_f32 };
            let gaze = if row == 0 { gaze_yaw } else { 0.0 };
            let dir_y = pitch.sin();
            let cos_pitch = pitch.cos();

            for i in 0..32 {
                let index = (row * 32) + i;
                let normalized_i = (i as f32 / 31.0) * 2.0 - 1.0;

                // Row 0: tan warp center-dense. Row 1: linear uniforme.
                let yaw_angle = if row == 0 {
                    let warped = (normalized_i * TAN_K).tan() / tan_k_val;
                    warped * fov_half + gaze
                } else {
                    normalized_i * fov_half
                };
                
                let dir_x = (forward.x * yaw_angle.cos() + right.x * yaw_angle.sin()) * cos_pitch;
                let dir_z = (forward.z * yaw_angle.cos() + right.z * yaw_angle.sin()) * cos_pitch;
                
                let mut hit_voxel = VoxelType::Road;
                let mut hit_dist = 300.0;
                
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
                
                // Alcance ~300m (100 blocos × 3m cell). Suficiente pro carro antecipar
                // curvas longas e não "descobrir" a parede só quando está em cima dela.
                for _ in 0..100 {
                    if side_dist_x < side_dist_y {
                        side_dist_x += delta_dist_x; map_x += step_x; side = 0;
                    } else {
                        side_dist_y += delta_dist_y; map_y += step_y; side = 1;
                    }
                    
                    let voxel = self.get_voxel_at((map_x as f32) * cell, (map_y as f32) * cell);
                    if voxel == VoxelType::Wall || voxel == VoxelType::Hole {
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
                } else if dir_y < -0.001 && dist_to_ground <= 300.0 {
                    // Sem paredes, mas bateu no chão
                    hit_dist = dist_to_ground;
                    let hit_px = eye_pos.x + dir_x * hit_dist;
                    let hit_pz = eye_pos.z + dir_z * hit_dist;
                    hit_voxel = self.get_voxel_at(hit_px, hit_pz);
                }
                
                let depth_fade = (hit_dist / 300.0).clamp(0.0, 1.0);

                // --- VISÃO ABSOLUTA ---
                // Para que o modelo NUNCA confunda uma parede longe com a pista perto.
                //
                // TIER 1.3 FIX — log-compress na parede:
                //   Saturação a 127 em qualquer hit < 300m cegava a rede (Retina_C=127 constante
                //   = brake perpétuo via A4). Substituímos linear por sqrt(depth_fade/0.5),
                //   efetivo: range dinâmico 127..40 em 0..150m, saturando em 37 além disso.
                //   Dá diferenciação de 90 entre "encostando (0m)" e "razoavelmente próxima (75m)".
                let pixel_val = match hit_voxel {
                    VoxelType::Wall => {
                        // Dual-scale depth: sqrt perto (0-30m dense resolution — crítico pra
                        // bater ou não) + linear longe (30-120m — modelo antecipa curvas
                        // distantes). Antes saturava 50m → cego pra setup de curva longa.
                        // Agora: 0-30m ocupa 127→55 (gradient ~72 pts), 30-120m ocupa 55→-20
                        // (gradient ~75 pts). Modelo distingue "curva chegando em 80m" de
                        // "curva em 30m" — dá tempo de pré-frear.
                        if hit_dist < 30.0 {
                            let near = (hit_dist / 30.0).clamp(0.0, 1.0);
                            127.0 - (near.sqrt() * 72.0)
                        } else {
                            let far = ((hit_dist - 30.0) / 90.0).clamp(0.0, 1.0);
                            55.0 - (far * 75.0)
                        }
                    },
                    VoxelType::Grass => {
                        40.0 - (depth_fade * 40.0)
                    },
                    VoxelType::Road => {
                        -128.0 + (depth_fade * 48.0)
                    },
                    VoxelType::Hole => {
                        // Hole = buraco no asfalto. Mata no contato → canal precisa ser
                        // BEM distinguível de Road/Grass/Wall. Range 90→60 em 0..50m,
                        // satura em 60 além. Sempre > Grass (40) → rede lê como hazard.
                        let near = (hit_dist / 50.0).clamp(0.0, 1.0);
                        90.0 - (near * 30.0)
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

// ============================================================================
// TRACK LIBRARY — waypoints por seed (progressão easy → very hard)
// ----------------------------------------------------------------------------
// Coordenadas em CELLS (multiplicadas por cell_size=3m no VoxelWorld::new).
// Grid 160×160 cells = 480×480m. Manter waypoints entre [10, 150] pra buffer.
// ============================================================================
pub fn track_waypoints_for_seed(seed: u64) -> Vec<Vec2> {
    match seed {
        // Dubai Oval — EASY: oval largo, curvas suaves <45°, runway longo.
        123 => vec![
            Vec2::new(20.0, 20.0),  Vec2::new(120.0, 20.0),
            Vec2::new(140.0, 40.0), Vec2::new(140.0, 100.0),
            Vec2::new(120.0, 120.0), Vec2::new(40.0, 120.0),
            Vec2::new(20.0, 100.0), Vec2::new(20.0, 20.0),
        ],
        // Interlagos Gateway — MEDIUM: current layout (90° + 2×45°).
        64 => vec![
            Vec2::new(20.0, 20.0),  Vec2::new(140.0, 20.0),
            Vec2::new(140.0, 60.0), Vec2::new(100.0, 100.0),
            Vec2::new(100.0, 140.0), Vec2::new(60.0, 140.0),
            Vec2::new(20.0, 100.0), Vec2::new(20.0, 20.0),
        ],
        // Monaco Tight — HARD: sequência de hairpins 90° sem reversos geométricos.
        // Antigo WP[3] fazia swing 135° (NW→E) — U-turn impossível pra rede shallow.
        // Agora monotonico em sentido horário: S→E→S→W→S→W→N→E.
        256 => vec![
            Vec2::new(20.0, 20.0),  Vec2::new(110.0, 20.0),
            Vec2::new(140.0, 50.0), Vec2::new(140.0, 90.0),
            Vec2::new(100.0, 90.0), Vec2::new(100.0, 130.0),
            Vec2::new(60.0, 130.0), Vec2::new(20.0, 90.0),
            Vec2::new(20.0, 20.0),
        ],
        // Silverstone Flow — VERY HARD: curvas rápidas sem zigzag reverso.
        // Antigo fazia (110,115)→(80,100)→(60,130) U-turn SW→NW→SW. Removido.
        512 => vec![
            Vec2::new(20.0, 20.0),  Vec2::new(100.0, 25.0),
            Vec2::new(140.0, 55.0), Vec2::new(140.0, 95.0),
            Vec2::new(115.0, 120.0), Vec2::new(75.0, 130.0),
            Vec2::new(35.0, 125.0), Vec2::new(15.0, 95.0),
            Vec2::new(30.0, 65.0), Vec2::new(55.0, 45.0),
            Vec2::new(20.0, 20.0),
        ],
        // Nürburgring Hills — EXTREME: elevação forte, curvas longas, sem U-turns.
        // Antigo (95,110)→(130,140) fazia reverso SW→SE. Removido.
        768 => vec![
            Vec2::new(20.0, 30.0),  Vec2::new(90.0, 20.0),
            Vec2::new(140.0, 45.0), Vec2::new(140.0, 100.0),
            Vec2::new(110.0, 130.0), Vec2::new(70.0, 140.0),
            Vec2::new(30.0, 120.0), Vec2::new(20.0, 70.0),
            Vec2::new(20.0, 30.0),
        ],
        // Suzuka Crossing — INSANE: hairpins sucessivos SEM reversos. Antigo (90,50)→(70,70)
        // fazia SE→SW (U-turn) e (20,90)→(40,130) fazia SW→SE (outro U). Redesigned para
        // espiral horária com 90°s legítimos mas trajetória sempre forward.
        1024 => vec![
            Vec2::new(20.0, 30.0),  Vec2::new(80.0, 30.0),
            Vec2::new(115.0, 55.0), Vec2::new(140.0, 90.0),
            Vec2::new(140.0, 130.0), Vec2::new(100.0, 145.0),
            Vec2::new(55.0, 140.0), Vec2::new(20.0, 115.0),
            Vec2::new(25.0, 75.0), Vec2::new(45.0, 55.0),
            Vec2::new(20.0, 30.0),
        ],
        _ => vec![
            Vec2::new(20.0, 20.0),  Vec2::new(140.0, 20.0),
            Vec2::new(140.0, 60.0), Vec2::new(100.0, 100.0),
            Vec2::new(100.0, 140.0), Vec2::new(60.0, 140.0),
            Vec2::new(20.0, 100.0), Vec2::new(20.0, 20.0),
        ],
    }
}

/// (amp, freq) — amplitude em metros, frequência em radianos/cell.
pub fn track_elevation_for_seed(seed: u64) -> (f32, f32) {
    match seed {
        123  => (0.8, 0.02),   // Dubai: quase flat
        64   => (1.5, 0.03),   // Interlagos: ondulação suave
        256  => (2.0, 0.04),   // Monaco: street-level bumps
        512  => (2.5, 0.035),  // Silverstone: ondulações
        768  => (6.0, 0.025),  // Nürburgring Hills: RAMPAS de verdade
        1024 => (3.5, 0.05),   // Suzuka: relevo moderado
        _    => (1.5, 0.03),
    }
}

/// Cells (x,y) onde holes (buracos kill-on-touch) ficam sobrepostos ao Road.
/// Só pistas HARD+ recebem. Layout escolhido pra punir trajetória preguiçosa sem
/// bloquear a racing line ideal — buraco sempre fora do centro da pista ou em
/// trechos onde a IA deve já ter aprendido curvar.
pub fn track_holes_for_seed(seed: u64) -> Vec<Vec2> {
    match seed {
        // Dubai / Interlagos — sem buracos (fase de aprendizado).
        123 | 64 => vec![],

        // Monaco Tight — 2 buracos em apex de 90° interno.
        256 => vec![
            Vec2::new(130.0, 55.0),   // Apex curva 2 (E→S)
            Vec2::new(98.0, 128.0),   // Apex curva 5 (S→W)
        ],

        // Silverstone Flow — 3 buracos em trechos rápidos.
        512 => vec![
            Vec2::new(110.0, 27.0),   // Reta 1
            Vec2::new(138.0, 85.0),   // Curva longa
            Vec2::new(32.0, 123.0),   // Saída W→N
        ],

        // Nürburgring Hills — 3 buracos em descidas (rampas amplificam).
        768 => vec![
            Vec2::new(130.0, 40.0),   // Descida 1
            Vec2::new(130.0, 115.0),  // Vale
            Vec2::new(40.0, 125.0),   // Contra-subida
        ],

        // Suzuka Crossing — 4 buracos (insano: trajetória quase forçada).
        1024 => vec![
            Vec2::new(75.0, 30.0),    // Reta inicial
            Vec2::new(135.0, 80.0),   // Curva descendente
            Vec2::new(115.0, 138.0),  // Hairpin S
            Vec2::new(25.0, 95.0),    // Subida W
        ],

        _ => vec![],
    }
}

pub fn track_start_for_seed(seed: u64) -> Vec2 {
    match seed {
        123  => Vec2::new(55.0, 20.0),   // Dubai: runway largo
        64   => Vec2::new(50.0, 20.0),   // Interlagos
        256  => Vec2::new(40.0, 20.0),   // Monaco: start curto (pista técnica)
        512  => Vec2::new(55.0, 22.0),   // Silverstone
        768  => Vec2::new(55.0, 28.0),   // Nürburgring
        1024 => Vec2::new(40.0, 30.0),   // Suzuka
        _    => Vec2::new(50.0, 20.0),
    }
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
            racing_line: crate::simulation::racing_line::RacingLine::default(),
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
