use crate::config::iara::vision::*;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use std::sync::atomic::{AtomicBool, Ordering};

/// Difficulty flags — main.rs sets via CLI. Default OFF (full difficulty).
pub static NO_HOLES: AtomicBool = AtomicBool::new(false);
pub static EASY_TRACK: AtomicBool = AtomicBool::new(false);
/// Silencia o despejo de 129 waypoints que `VoxelWorld::new` faz no stdout.
///
/// O mapa impresso é útil UMA vez, para conferir o traçado. Em ferramentas que constroem
/// o mundo só para consultar geometria (o diagnosticador de replay, por exemplo) ele afoga
/// o relatório de verdade em 130 linhas.
pub static QUIET_MAP: AtomicBool = AtomicBool::new(false);

/// Liga o silêncio enquanto o valor devolvido estiver vivo.
pub fn silence_map_diag() -> impl Drop {
    struct Guard;
    impl Drop for Guard {
        fn drop(&mut self) {
            QUIET_MAP.store(false, Ordering::Relaxed);
        }
    }
    QUIET_MAP.store(true, Ordering::Relaxed);
    Guard
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum VoxelType {
    Grass = 0,
    Road = 1,
    Wall = 2,
    Hole = 3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectProposal {
    /// Angle from origin to cluster center (radians)
    pub angle: f32,
    /// Distance from origin to cluster center (meters)
    pub distance: f32,
    /// Width of cluster perpendicular to radial direction (cells)
    pub width: f32,
    /// Local curvature of centerline (average banking value)
    pub curvature: f32,
}

/// Fonte única do tamanho do grid — `VoxelWorld::new` (cells, ver `size` ali) e
/// `perfil_para_cells` usavam números divergentes (320 real vs. 160 num comentário
/// morto) porque cada um tinha sua própria constante mágica. Debug de unidades
/// 2026-08-15: `size` mudou de 160→320 num commit anterior e ninguém atualizou os
/// dois comentários que citavam "grid 160×160" nem o centro (80,80) que dependia
/// dele — a pista de perfil acabava recentrada 80 células (240 m) fora do centro
/// real do mundo. CENTRO é o geométrico do grid (`GRID_CELLS/2`); as pistas
/// legadas abaixo (Dubai/Monaco/etc. e `pista_gen::tracado`, centro (120,120))
/// são de um layout AINDA mais antigo e só continuam válidas por sorte de span
/// pequeno — não mexidas aqui, fora do escopo deste conserto.
pub const GRID_CELLS: f32 = 320.0; // == `size` em VoxelWorld::new; 320×320×3m = 960×960m

#[derive(Resource, Default)]
pub struct VoxelWorldResource(pub VoxelWorld);

pub struct VoxelWorld {
    pub size: usize,
    pub grid: Vec<VoxelType>,
    pub flow: Vec<Vec2>,
    pub height_map: Vec<f32>, // Elevação do terreno (metros)
    pub banking: Vec<f32>,    // Inclinação lateral (radianos)
    /// Min distance (cells) from each cell to nearest road waypoint segment.
    /// Used by generate_mesh to cull grass beyond extent threshold (sea visible).
    pub min_dist_to_road: Vec<f32>,
    pub cell_size: f32,
    pub start_pos: Vec3,
    pub lap_distance: f32,
    pub waypoints_world: Vec<Vec2>, // Waypoints já em coordenadas de mundo (x,z)
    pub waypoint_arcs: Vec<f32>, // Arco cumulativo por waypoint (arcs[i] = comprimento de wp[0] até wp[i])
    pub start_progress: f32, // Arc-length da start_pos projetada (cacheado em new() — constante)
    /// Ideal racing line (outside-in-outside, precomputed once per track).
    /// Used to feed sensor channels 108/109/110 (lateral offset, derivative, lookahead).
    pub racing_line: crate::simulation::racing_line::RacingLine,
    /// Detected wall clusters as object proposals (precomputed @ init)
    pub proposals: Vec<ObjectProposal>,
    /// Flag: proposals need recomputation if true
    pub proposal_valid: bool,
}

impl VoxelWorld {
    pub fn new(_seed: u64, track_width: f32) -> Self {
        let size = GRID_CELLS as usize; // 320×320 cells × 3m = 960×960m world (mapa maior)
        let cell_size = 3.0;
        let mut grid = vec![VoxelType::Grass; size * size];
        let mut flow = vec![Vec2::ZERO; size * size];
        let mut height_map = vec![0.0f32; size * size];
        let mut banking = vec![0.0f32; size * size];
        let mut min_dist_to_road = vec![0.0f32; size * size];

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
                    let d = dist_to_segment(pos, waypoints[i], waypoints[i + 1]);
                    if d < min_dist {
                        min_dist = d;
                        best_dir = (waypoints[i + 1] - waypoints[i]).normalize_or_zero();
                        best_seg_idx = i;
                    }
                }

                let idx = y * size + x;
                flow[idx] = best_dir;
                min_dist_to_road[idx] = min_dist;
                // ERRO DE UNIDADE, corrigido: `min_dist` sai de `pos = (x, y)` da GRADE,
                // logo está em CÉLULAS; `track_width` vem do chamador em METROS. Comparar
                // os dois direto multiplicava a pista por `cell_size` (3,0): uma pista
                // declarada de 28 m virava 84 m reais, medido com `pista_check` (42 m de
                // asfalto livre para cada lado do eixo, em todo o traçado).
                //
                // A consequência não foi cosmética. Com 84 m de largura e curva de raio
                // 51 m, a pista é mais larga do que a curva é fechada — não existe curva a
                // negociar. E TODOS os sensores de parede saturam: `fwd_dist` cravado em
                // 30,0, raios laterais no teto, retina com o mesmo valor em toda direção.
                // O ser não tinha o que ver nem por que virar. Foi isso, e não o cérebro,
                // que segurou a evolução em 3/24 setores por trinta gerações.
                let dist_m = min_dist * cell_size;
                grid[idx] = if dist_m < track_width / 2.0 {
                    VoxelType::Road
                } else if dist_m < (track_width / 2.0) + 4.5 {
                    VoxelType::Wall
                } else {
                    VoxelType::Grass
                };

                // Elevação: amplitude/frequência por pista — Nürburgring Hills tem rampas sérias.
                let elev_base = ((x as f32 * height_freq).sin() * height_amp
                    + (y as f32 * height_freq * 0.67).cos() * height_amp * 1.3)
                    * 0.5;
                height_map[idx] = if grid[idx] == VoxelType::Road || grid[idx] == VoxelType::Wall {
                    elev_base
                } else {
                    elev_base - 0.1 // Grama ligeiramente mais baixa
                };

                // Banking: inclinação em curvas (segmentos 2,3,5,6 são curvas)
                if grid[idx] == VoxelType::Road
                    && (best_seg_idx == 2
                        || best_seg_idx == 3
                        || best_seg_idx == 5
                        || best_seg_idx == 6)
                {
                    // Mesma correção de unidade: metros contra metros.
                    let norm_dist = (min_dist * cell_size) / (track_width / 2.0); // 0=centro, 1=borda
                    banking[idx] = 0.12 * (1.0 - norm_dist); // ~7° de banking no centro das curvas
                }
            }
        }

        // Overlay HOLES (VoxelType::Hole) em cells específicas por seed. Só em pistas
        // HARD+ (256, 512, 768, 1024). Holes matam no contato — força rede a aprender
        // desvio ativo, não passivo. Só sobrescreve Road (nunca Wall/Grass).
        // Cada ponto vira bloco 2×2 (6m × 6m) — visível na retina e passível de colisão
        // pelo chassi (2m × 2.6m). Holes isolados de 1 cell eram skipáveis.
        // P1.8: passa o traçado REAL (`waypoints`, cell-space) e a largura real da
        // pista — achado "buracos do currículo calculados sobre traçado FANTASMA"
        // (procedural_holes rodava seu próprio `procedural_waypoints(seed)`
        // independente, geometria sem relação com o que `do_curriculo` gerou para
        // seed≥1_000_000) e achado "buracos nascem em cima do eixo" (offset
        // lateral zero, sem guard de largura mínima).
        let holes_iter = if NO_HOLES.load(Ordering::Relaxed) {
            Vec::new()
        } else {
            track_holes_for_seed(_seed, &waypoints, track_width / cell_size)
        };
        for hole_cell in holes_iter {
            let hx = hole_cell.x.round() as i32;
            let hy = hole_cell.y.round() as i32;
            for dy in 0..2 {
                for dx in 0..2 {
                    let cx = hx + dx;
                    let cy = hy + dy;
                    if cx < 0 || cx >= size as i32 || cy < 0 || cy >= size as i32 {
                        continue;
                    }
                    let idx = cy as usize * size + cx as usize;
                    // Bloco tem ~6-8.5m de lado; só perfura pista larga o bastante pra
                    // sobrar corredor pro chassi (2m×2.6m) — achado "overlay incondicional
                    // sem consultar largura restante".
                    if grid[idx] == VoxelType::Road && track_width >= 12.0 {
                        grid[idx] = VoxelType::Hole;
                    }
                }
            }
        }

        let waypoints_world: Vec<Vec2> = waypoints.iter().map(|p| *p * cell_size).collect();
        let mut waypoint_arcs = vec![0.0];
        let mut total_dist = 0.0;
        for i in 0..waypoints_world.len() - 1 {
            total_dist += waypoints_world[i].distance(waypoints_world[i + 1]);
            waypoint_arcs.push(total_dist);
        }

        // Start pos por pista — runway suficiente pra ler primeira curva a ~40m/s.
        let start_pos_cells = track_start_for_seed(_seed);
        let start_pos = Vec3::new(
            start_pos_cells.x * cell_size,
            0.5,
            start_pos_cells.y * cell_size,
        );

        // Pré-calcula start_progress uma vez — antes era refeito a cada get_track_progress
        // (500 carros × 60 Hz × 10× time_scale = 300k recálculos/s de uma constante).
        let start_pos_2d = Vec2::new(start_pos.x, start_pos.z);
        let mut start_progress = 0.0;
        let mut min_start_dist = f32::MAX;
        for i in 0..waypoints_world.len() - 1 {
            let p1 = waypoints_world[i];
            let p2 = waypoints_world[i + 1];
            let seg_len = p1.distance(p2);
            let mut dir = p2 - p1;
            if seg_len > 0.0 {
                dir /= seg_len;
            }
            let v = start_pos_2d - p1;
            let t = v.dot(dir).clamp(0.0, seg_len);
            let proj = p1 + dir * t;
            let dist = start_pos_2d.distance(proj);
            if dist < min_start_dist {
                min_start_dist = dist;
                start_progress = waypoint_arcs[i] + t;
            }
        }

        // === DIAGNÓSTICO DE MAPA ===
        // Impresso 1× no boot. Ajuda identificar problemas de design (curvas cegas,
        // start mal colocado, waypoints com segmentos absurdos).
        let n_road = grid.iter().filter(|v| **v == VoxelType::Road).count();
        let n_wall = grid.iter().filter(|v| **v == VoxelType::Wall).count();
        let n_grass = grid.iter().filter(|v| **v == VoxelType::Grass).count();
        let n_hole = grid.iter().filter(|v| **v == VoxelType::Hole).count();
        if !QUIET_MAP.load(Ordering::Relaxed) {
            println!(
                "🗺  MAP DIAG track_width={:.1}m cell={:.1}m lap={:.1}m",
                track_width * cell_size * 0.0 + track_width,
                cell_size,
                total_dist
            );
            println!(
                "   voxels road={} wall={} grass={} hole={} road_ratio={:.1}%",
                n_road,
                n_wall,
                n_grass,
                n_hole,
                100.0 * n_road as f32 / (n_road + n_wall + n_grass + n_hole) as f32
            );
            println!(
                "   start_pos=({:.1}, {:.1}) start_progress={:.1}m / lap {:.1}m ({:.1}%)",
                start_pos.x,
                start_pos.z,
                start_progress,
                total_dist,
                100.0 * start_progress / total_dist
            );
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
                    } else {
                        0.0
                    };
                    (sl, turn)
                };
                let tag = if turn_deg > 60.0 {
                    "⚠ SHARP"
                } else if turn_deg > 30.0 {
                    "CURVE"
                } else {
                    "straight"
                };
                println!(
                    "   WP[{}] ({:.0},{:.0}) seg_len={:.1}m turn={:.1}° {}",
                    i, waypoints_world[i].x, waypoints_world[i].y, seg_len, turn_deg, tag
                );
            }
        }

        // Precompute ideal racing line (outside-in-outside). Built once per
        // track — this is the teacher signal for channels 108/109/110.
        let racing_line = crate::simulation::racing_line::RacingLine::from_centerline(
            &waypoints_world,
            track_width,
        );

        let mut world = Self {
            size,
            grid,
            flow,
            height_map,
            banking,
            min_dist_to_road,
            cell_size,
            start_pos,
            lap_distance: total_dist,
            waypoints_world,
            waypoint_arcs,
            start_progress,
            racing_line,
            proposals: Vec::new(),
            proposal_valid: true,
        };

        // Precompute wall cluster proposals (O(N) once at init, O(1) per query)
        world.proposals = world.detect_wall_clusters();
        world
    }

    /// O(1) material lookup: returns voxel type at world coordinates
    /// Returns Wall if outside grid bounds
    pub fn material_at(&self, pos: Vec2) -> VoxelType {
        let x = (pos.x / self.cell_size).round() as i32;
        let z = (pos.y / self.cell_size).round() as i32;
        if x >= 0 && x < self.size as i32 && z >= 0 && z < self.size as i32 {
            self.grid[z as usize * self.size + x as usize]
        } else {
            VoxelType::Wall
        }
    }

    /// Caminha o eixo da pista e reporta defeitos geométricos: eixo fora do
    /// asfalto (curva cortando parede/buraco) ou estrangulamento (largura livre
    /// somada dos dois lados < metade da largura declarada). Extraído de
    /// `src/bin/pista_check.rs` (P1.8, achado "pista_check nunca é chamado por
    /// gpu_evolve — não existe gate contra pista com defeito") para servir de
    /// GATE antes de treinar: uma pista com defeito não é vencível por NENHUM
    /// controlador (quatro controladores independentes pararam no mesmo ponto —
    /// não era coincidência de controle, era o mapa).
    ///
    /// `width`: largura declarada em metros (mesma usada em `VoxelWorld::new`).
    /// `ate`: até quantos metros de eixo caminhar (tipicamente `self.lap_distance`).
    /// Devolve uma linha de texto por defeito encontrado; vazio = pista dirigível
    /// em todo o trecho examinado.
    pub fn validar_pista(&self, width: f32, ate: f32) -> Vec<String> {
        let wps = &self.waypoints_world;
        let n = wps.len();
        let mut defeitos = Vec::new();
        if n < 2 {
            defeitos.push("pista sem waypoints suficientes (n < 2)".to_string());
            return defeitos;
        }
        let mut arco = 0.0f32;
        let mut wp = 0usize;
        let mut t = 0.0f32;
        while arco < ate && wp < n - 1 {
            let a = wps[wp];
            let b = wps[wp + 1];
            let seg = (b - a).length().max(0.001);
            let p = a + (b - a) * (t / seg);
            let dir_seg = (b - a) / seg;
            let nrm = Vec2::new(-dir_seg.y, dir_seg.x);

            let mat = self.material_at(p);
            if !matches!(mat, VoxelType::Road) {
                defeitos.push(format!("{arco:.0}m: eixo fora da pista ({mat:?})"));
            } else {
                let livre = |sinal: f32| -> f32 {
                    let mut d = 0.0f32;
                    while d < 300.0 {
                        let q = p + nrm * (d * sinal);
                        if !matches!(self.material_at(q), VoxelType::Road) {
                            return d;
                        }
                        d += 0.5;
                    }
                    300.0
                };
                let esq = livre(1.0);
                let dir = livre(-1.0);
                if esq + dir < width * 0.5 {
                    defeitos.push(format!(
                        "{arco:.0}m: estrangulamento (livre {:.1}m de {width:.0}m declarados)",
                        esq + dir
                    ));
                }
            }

            arco += 1.0;
            t += 1.0;
            if t >= seg {
                t -= seg;
                wp += 1;
            }
        }
        defeitos
    }

    /// O(1) elevation lookup: returns height at grid cell nearest to world position
    /// Returns 0.0 if outside grid bounds
    pub fn elevation_at(&self, pos: Vec2) -> f32 {
        let x = (pos.x / self.cell_size).round() as i32;
        let z = (pos.y / self.cell_size).round() as i32;
        if x >= 0 && x < self.size as i32 && z >= 0 && z < self.size as i32 {
            self.height_map[z as usize * self.size + x as usize]
        } else {
            0.0
        }
    }

    /// O(1) curvature (banking) lookup: returns banking angle at grid cell
    /// Returns 0.0 if outside grid bounds
    pub fn curvature_at(&self, pos: Vec2) -> f32 {
        let x = (pos.x / self.cell_size).round() as i32;
        let z = (pos.y / self.cell_size).round() as i32;
        if x >= 0 && x < self.size as i32 && z >= 0 && z < self.size as i32 {
            self.banking[z as usize * self.size + x as usize]
        } else {
            0.0
        }
    }

    /// Detects wall clusters using connected-components (flood-fill on Wall cells)
    /// Returns vector of ObjectProposals, one per cluster
    pub fn detect_wall_clusters(&self) -> Vec<ObjectProposal> {
        let mut visited = vec![false; self.size * self.size];
        let mut proposals = Vec::new();

        for y in 0..self.size {
            for x in 0..self.size {
                let idx = y * self.size + x;
                if !visited[idx] && self.grid[idx] == VoxelType::Wall {
                    let cluster = self.flood_fill_cluster(x, y, &mut visited);
                    if let Some(proposal) = self.compute_proposal_from_cluster(&cluster) {
                        proposals.push(proposal);
                    }
                }
            }
        }

        proposals
    }

    /// Flood-fill: collects all contiguous Wall cells starting from (x, y)
    fn flood_fill_cluster(
        &self,
        start_x: usize,
        start_y: usize,
        visited: &mut [bool],
    ) -> Vec<(usize, usize)> {
        let mut cluster = Vec::new();
        let mut stack = vec![(start_x as i32, start_y as i32)];

        while let Some((x, y)) = stack.pop() {
            if x < 0 || x >= self.size as i32 || y < 0 || y >= self.size as i32 {
                continue;
            }

            let idx = y as usize * self.size + x as usize;
            if visited[idx] {
                continue;
            }

            if self.grid[idx] != VoxelType::Wall {
                continue;
            }

            visited[idx] = true;
            cluster.push((x as usize, y as usize));

            stack.push((x + 1, y));
            stack.push((x - 1, y));
            stack.push((x, y + 1));
            stack.push((x, y - 1));
        }

        cluster
    }

    /// Computes ObjectProposal from cluster of Wall cells
    fn compute_proposal_from_cluster(&self, cluster: &[(usize, usize)]) -> Option<ObjectProposal> {
        if cluster.is_empty() {
            return None;
        }

        let mut cx = 0.0_f32;
        let mut cz = 0.0_f32;
        for &(x, z) in cluster {
            cx += (x as f32 + 0.5) * self.cell_size;
            cz += (z as f32 + 0.5) * self.cell_size;
        }
        cx /= cluster.len() as f32;
        cz /= cluster.len() as f32;

        let angle = cz.atan2(cx);
        let distance = (cx * cx + cz * cz).sqrt();

        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let mut min_perp = f32::MAX;
        let mut max_perp = f32::MIN;
        for &(x, z) in cluster {
            let wx = (x as f32 + 0.5) * self.cell_size;
            let wz = (z as f32 + 0.5) * self.cell_size;
            let perp = -wx * sin_a + wz * cos_a;
            min_perp = min_perp.min(perp);
            max_perp = max_perp.max(perp);
        }
        let width = (max_perp - min_perp) / self.cell_size;

        let mut total_curvature = 0.0_f32;
        for &(x, z) in cluster {
            total_curvature += self.banking[z * self.size + x];
        }
        let curvature = total_curvature / cluster.len() as f32;

        Some(ObjectProposal {
            angle,
            distance,
            width,
            curvature,
        })
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
            let p2 = self.waypoints_world[i + 1];
            let seg_len = p1.distance(p2).max(0.001);
            let dir = (p2 - p1) / seg_len;
            let t = (pos - p1).dot(dir).clamp(0.0, seg_len);
            let proj = p1 + dir * t;
            let d = pos.distance(proj);
            if d < min_dist {
                min_dist = d;
                best_seg = i;
            }
        }
        best_seg
    }

    pub fn get_voxel_at(&self, wx: f32, wz: f32) -> VoxelType {
        let x = (wx / self.cell_size).round() as i32;
        let y = (wz / self.cell_size).round() as i32;
        if x >= 0 && x < self.size as i32 && y >= 0 && y < self.size as i32 {
            self.grid[y as usize * self.size + x as usize]
        } else {
            VoxelType::Wall
        }
    }

    pub fn get_flow_at(&self, wx: f32, wz: f32) -> Vec2 {
        let x = (wx / self.cell_size).round() as i32;
        let y = (wz / self.cell_size).round() as i32;
        if x >= 0 && x < self.size as i32 && y >= 0 && y < self.size as i32 {
            self.flow[y as usize * self.size + x as usize]
        } else {
            Vec2::ZERO
        }
    }

    /// Calcula a distância absoluta na pista a partir da linha de largada
    pub fn get_track_progress(&self, wx: f32, wz: f32) -> f32 {
        let pos = Vec2::new(wx, wz);
        let mut min_dist = f32::MAX;
        let mut best_progress = 0.0;

        for i in 0..self.waypoints_world.len() - 1 {
            let p1 = self.waypoints_world[i];
            let p2 = self.waypoints_world[i + 1];
            let seg_len = p1.distance(p2);

            let mut dir = p2 - p1;
            if seg_len > 0.0 {
                dir /= seg_len;
            }

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
            } else {
                0.0
            }
        };

        let h00 = h(x0, z0);
        let h10 = h(x0 + 1, z0);
        let h01 = h(x0, z0 + 1);
        let h11 = h(x0 + 1, z0 + 1);

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

        // Overlap: expand grass/road quads by 0.05 cell (~15cm) to hide sub-pixel
        // seams at distance. Adjacent quads overlap; tearing/Z-fighting cracks vanish.
        let overlap = 0.05 * self.cell_size;

        for y in 0..self.size {
            for x in 0..self.size {
                let v = self.grid[y * self.size + x];
                let px = x as f32 * self.cell_size;
                let pz = y as f32 * self.cell_size;
                let py = self.height_map[y * self.size + x];

                if v == VoxelType::Road {
                    let is_start_line = (px - self.start_pos.x).abs() < self.cell_size
                        && (pz - self.start_pos.z).abs() < 25.0;
                    let b = road_pos.len() as u32;
                    let x0 = px - overlap;
                    let x1 = px + self.cell_size + overlap;
                    let z0 = pz - overlap;
                    let z1 = pz + self.cell_size + overlap;
                    road_pos.extend_from_slice(&[
                        [x0, py + 0.02, z0],
                        [x1, py + 0.02, z0],
                        [x0, py + 0.02, z1],
                        [x1, py + 0.02, z1],
                    ]);

                    // Asfalto coloring: faixa branca lateral + curb red/white na borda
                    // de curva forte (banking>0). Centro escuro xadrez sutil.
                    let dist_road = self.min_dist_to_road[y * self.size + x];
                    let banking_here = self.banking[y * self.size + x];
                    let is_lane_edge = dist_road > 5.5 && dist_road < 6.5; // ~6 cells = borda visual
                    let is_curb = banking_here > 0.05 && dist_road > 6.5; // só em curva
                    let dash_phase = ((x + y) / 4) % 2 == 0; // tracejado
                    let is_light = ((x / 2) + (y / 2)) % 2 == 0;
                    let color = if is_start_line {
                        [1.0, 1.0, 1.0, 1.0]
                    } else if is_curb {
                        // curb red/white alternado
                        if (x / 2) % 2 == 0 {
                            [0.85, 0.10, 0.10, 1.0]
                        } else {
                            [0.95, 0.95, 0.95, 1.0]
                        }
                    } else if is_lane_edge {
                        [0.85, 0.85, 0.85, 1.0] // faixa branca contínua
                    } else if dist_road < 0.8 && dash_phase {
                        [0.78, 0.78, 0.78, 1.0] // linha central tracejada
                    } else if is_light {
                        [0.22, 0.22, 0.25, 1.0]
                    } else {
                        [0.15, 0.15, 0.18, 1.0]
                    };

                    road_colors.extend_from_slice(&[color; 4]);
                    road_indices.extend_from_slice(&[b, b + 2, b + 1, b + 1, b + 2, b + 3]);
                } else if v == VoxelType::Grass {
                    // Grass cobre TODA grid (sem buracos no interior do circuito).
                    // Borda: cells nas extremidades do grid descem suavemente pra sea
                    // (Y=-2) criando ilha em vez de papel cortado. Centro grid = (cx,cy).
                    let half = self.size as f32 * 0.5;
                    let dx = x as f32 - half;
                    let dy = y as f32 - half;
                    let radial = (dx * dx + dy * dy).sqrt();
                    // Grid 320 cells. Plateau até radial 100 (300m centro). Slope 100..160 cells
                    // desce 4m → encontra sea Y=-2 antes do edge. Sem vácuo visível com céu.
                    let edge_t = ((radial - 100.0) / 60.0).clamp(0.0, 1.0);
                    let py_eff = py - edge_t * edge_t * 4.0; // até 4m abaixo (passa sea Y=-2)
                    let b = grass_pos.len() as u32;
                    let x0 = px - overlap;
                    let x1 = px + self.cell_size + overlap;
                    let z0 = pz - overlap;
                    let z1 = pz + self.cell_size + overlap;
                    grass_pos.extend_from_slice(&[
                        [x0, py_eff, z0],
                        [x1, py_eff, z0],
                        [x0, py_eff, z1],
                        [x1, py_eff, z1],
                    ]);

                    let is_light = ((x / 3) + (y / 3)) % 2 == 0;
                    let base = if is_light {
                        [0.18, 0.42, 0.18, 1.0]
                    } else {
                        [0.12, 0.32, 0.12, 1.0]
                    };
                    // Edge color: escurece + leve azulado próximo à sea pra blend.
                    let darken = 1.0 - 0.5 * edge_t;
                    let color = [
                        base[0] * darken,
                        base[1] * darken,
                        base[2] * darken + edge_t * 0.10,
                        1.0,
                    ];
                    grass_colors.extend_from_slice(&[color; 4]);
                    grass_indices.extend_from_slice(&[b, b + 2, b + 1, b + 1, b + 2, b + 3]);
                } else if v == VoxelType::Wall {
                    // Concrete barrier: 6 faces (caixa) ocupando cell quase inteira, altura 1.6m.
                    let b = wall_pos.len() as u32;
                    let h = 1.6_f32;
                    let inset = 0.05_f32;
                    let x0 = px + inset;
                    let x1 = px + self.cell_size - inset;
                    let z0 = pz + inset;
                    let z1 = pz + self.cell_size - inset;
                    let y0 = py;
                    let y1 = py + h;
                    // 8 vértices do cubo
                    wall_pos.extend_from_slice(&[
                        [x0, y0, z0],
                        [x1, y0, z0],
                        [x0, y0, z1],
                        [x1, y0, z1],
                        [x0, y1, z0],
                        [x1, y1, z0],
                        [x0, y1, z1],
                        [x1, y1, z1],
                    ]);
                    // 12 triangles (6 faces × 2 tris)
                    let f = |a, b, c| [a, b, c];
                    wall_indices.extend_from_slice(
                        &[
                            f(b + 0, b + 2, b + 1),
                            f(b + 1, b + 2, b + 3), // bottom
                            f(b + 4, b + 5, b + 6),
                            f(b + 5, b + 7, b + 6), // top
                            f(b + 0, b + 1, b + 4),
                            f(b + 1, b + 5, b + 4), // -z
                            f(b + 2, b + 6, b + 3),
                            f(b + 3, b + 6, b + 7), // +z
                            f(b + 0, b + 4, b + 2),
                            f(b + 2, b + 4, b + 6), // -x
                            f(b + 1, b + 3, b + 5),
                            f(b + 3, b + 7, b + 5), // +x
                        ]
                        .iter()
                        .flat_map(|tri| tri.iter().copied())
                        .collect::<Vec<u32>>(),
                    );
                } else if v == VoxelType::Hole {
                    // Hole 3D: caixa funda 2.5m com 4 paredes laterais + bottom preto.
                    // Antes apenas quad raso 0.3m — invisível no chase cam.
                    // Agora visualmente óbvio: caixa profunda preta com sombra de borda.
                    let depth = 2.5_f32;
                    let inset = 0.05_f32;
                    let x0 = px + inset;
                    let x1 = px + self.cell_size - inset;
                    let z0 = pz + inset;
                    let z1 = pz + self.cell_size - inset;
                    let y_top = py - 0.05; // borda superior (afundamento visível)
                    let y_bot = py - depth;
                    let b = road_pos.len() as u32;
                    // 8 vertices: 4 top + 4 bottom
                    road_pos.extend_from_slice(&[
                        [x0, y_top, z0],
                        [x1, y_top, z0],
                        [x0, y_top, z1],
                        [x1, y_top, z1],
                        [x0, y_bot, z0],
                        [x1, y_bot, z0],
                        [x0, y_bot, z1],
                        [x1, y_bot, z1],
                    ]);
                    // Top edges = cinza escuro (borda do buraco). Bottom = preto.
                    let edge = [0.10, 0.10, 0.12, 1.0];
                    let void = [0.0, 0.0, 0.0, 1.0];
                    road_colors
                        .extend_from_slice(&[edge, edge, edge, edge, void, void, void, void]);
                    // Bottom (visto de cima): triangles a partir dos bottom verts.
                    road_indices.extend_from_slice(&[
                        b + 4,
                        b + 5,
                        b + 6,
                        b + 5,
                        b + 7,
                        b + 6, // bottom face
                        b + 0,
                        b + 4,
                        b + 1,
                        b + 1,
                        b + 4,
                        b + 5, // -z wall
                        b + 2,
                        b + 3,
                        b + 6,
                        b + 3,
                        b + 7,
                        b + 6, // +z wall
                        b + 0,
                        b + 2,
                        b + 4,
                        b + 2,
                        b + 6,
                        b + 4, // -x wall
                        b + 1,
                        b + 5,
                        b + 3,
                        b + 3,
                        b + 5,
                        b + 7, // +x wall
                    ]);
                }
            }
        }

        let mut r_mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        r_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, road_pos);
        r_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, road_colors);
        r_mesh.insert_indices(Indices::U32(road_indices));

        let mut g_mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        g_mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, grass_pos);
        g_mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, grass_colors);
        g_mesh.insert_indices(Indices::U32(grass_indices));

        let mut w_mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
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
    ///
    /// `other_beings`: world (x, y, z) of every OTHER being — this being's own position
    /// excluded by the caller — for the occlusion test of `--retina-beings`
    /// (`docs/RETINA-BEINGS.md` §2). **An empty slice reproduces this function's
    /// behaviour exactly as it was before that flag existed** — the being loop below is
    /// skipped entirely, not merely a no-op pass over zero elements, so the retina of
    /// every caller that has not opted in stays bit-for-bit identical (see
    /// `retina_beings_off_is_bit_identical_to_before` in this file's tests). This is the
    /// CPU reference `gpu_validate --retina` compares the GPU kernel against
    /// (`src/gpu/retina.wgsl`); the two occlusion tests must stay the same algorithm.
    pub fn render_binocular_retina(
        &self,
        pos: Vec3,
        forward: Vec3,
        right: Vec3,
        gaze_yaw: f32,
        other_beings: &[Vec3],
    ) -> [i8; RETINA_TOTAL] {
        let mut pixels = [-128i8; RETINA_TOTAL];
        let eye_pos = pos + Vec3::Y * 0.4;
        let cell = self.cell_size;

        const TAN_K: f32 = 1.25_f32;
        let tan_k_val = TAN_K.tan();
        for row in 0..RETINA_ROWS {
            // ALCANCE e ÂNGULO vêm de `visao::` para poderem ser VARIADOS (ver `visao.rs`).
            // Sem parâmetro, devolvem exatamente as constantes originais — há teste travando
            // isso, senão a mudança alteraria o comportamento de todo experimento anterior.
            let fov_half = (crate::visao::fov_deg() / 2.0).to_radians();
            let pitch = crate::visao::pitch(row);
            let gaze = if row == 0 { gaze_yaw } else { 0.0 };
            let dir_y = pitch.sin();
            let cos_pitch = pitch.cos();

            for i in 0..RETINA_COLS {
                let index = row * RETINA_COLS + i;
                let normalized_i = (i as f32 / (RETINA_COLS as f32 - 1.0)) * 2.0 - 1.0;

                // Row 0: tan warp center-dense (fóvea). Demais: linear uniforme.
                let yaw_angle = if row == 0 {
                    let warped = (normalized_i * TAN_K).tan() / tan_k_val;
                    warped * fov_half + gaze
                } else {
                    normalized_i * fov_half
                };

                let dir_x = (forward.x * yaw_angle.cos() + right.x * yaw_angle.sin()) * cos_pitch;
                let dir_z = (forward.z * yaw_angle.cos() + right.z * yaw_angle.sin()) * cos_pitch;

                let mut hit_voxel = VoxelType::Road;
                let mut hit_dist = 1200.0;

                let dist_to_ground = if dir_y < -0.001 {
                    -eye_pos.y / dir_y
                } else {
                    1000.0
                };

                // Implementação do Algoritmo DDA (Digital Differential Analyzer) - Precisão Metafísica O(1)
                let ray_start = Vec2::new(eye_pos.x / cell, eye_pos.z / cell);
                let ray_dir = Vec2::new(dir_x, dir_z).normalize_or_zero();

                let mut map_x = ray_start.x.floor() as i32;
                let mut map_y = ray_start.y.floor() as i32;

                let delta_dist_x = if ray_dir.x.abs() < 1e-5 {
                    1e30
                } else {
                    (1.0 / ray_dir.x).abs()
                };
                let delta_dist_y = if ray_dir.y.abs() < 1e-5 {
                    1e30
                } else {
                    (1.0 / ray_dir.y).abs()
                };

                let step_x: i32;
                let step_y: i32;
                let mut side_dist_x: f32;
                let mut side_dist_y: f32;

                if ray_dir.x < 0.0 {
                    step_x = -1;
                    side_dist_x = (ray_start.x - map_x as f32) * delta_dist_x;
                } else {
                    step_x = 1;
                    side_dist_x = ((map_x as f32 + 1.0) - ray_start.x) * delta_dist_x;
                }

                if ray_dir.y < 0.0 {
                    step_y = -1;
                    side_dist_y = (ray_start.y - map_y as f32) * delta_dist_y;
                } else {
                    step_y = 1;
                    side_dist_y = ((map_y as f32 + 1.0) - ray_start.y) * delta_dist_y;
                }

                let mut hit = false;
                let mut side = 0; // 0 = hit X wall, 1 = hit Z wall

                // Alcance ~400m (134 blocos × 3m cell). Profundidade aumentada — modelo
                // antecipa curvas longas em retas largas a 80m/s antes (era 300m).
                for _ in 0..400 {
                    if side_dist_x < side_dist_y {
                        side_dist_x += delta_dist_x;
                        map_x += step_x;
                        side = 0;
                    } else {
                        side_dist_y += delta_dist_y;
                        map_y += step_y;
                        side = 1;
                    }

                    let voxel = self.get_voxel_at((map_x as f32) * cell, (map_y as f32) * cell);
                    if voxel == VoxelType::Wall || voxel == VoxelType::Hole {
                        hit_voxel = voxel;
                        hit = true;
                        break;
                    }
                }

                if hit {
                    let current_dist = (if side == 0 {
                        (map_x as f32 - ray_start.x + (1 - step_x) as f32 / 2.0) / ray_dir.x
                    } else {
                        (map_y as f32 - ray_start.y + (1 - step_y) as f32 / 2.0) / ray_dir.y
                    }) * cell;

                    if dir_y < -0.001 && current_dist > dist_to_ground {
                        // O raio bateu no chão ANTES de atingir a parede lateral DDA
                        hit_dist = dist_to_ground;
                        let hit_px = eye_pos.x + dir_x * hit_dist;
                        let hit_pz = eye_pos.z + dir_z * hit_dist;
                        hit_voxel = self.get_voxel_at(hit_px, hit_pz);
                    } else {
                        hit_dist = current_dist;
                    }
                } else if dir_y < -0.001 && dist_to_ground <= 1200.0 {
                    // Sem paredes, mas bateu no chão
                    hit_dist = dist_to_ground;
                    let hit_px = eye_pos.x + dir_x * hit_dist;
                    let hit_pz = eye_pos.z + dir_z * hit_dist;
                    hit_voxel = self.get_voxel_at(hit_px, hit_pz);
                }

                // ── Being occlusion (`--retina-beings`, docs/RETINA-BEINGS.md §2) ──────
                // `other_beings.is_empty()` short-circuits the whole block, which is what
                // makes an empty slice reproduce this function's pre-existing behaviour
                // exactly. Same algorithm as `src/gpu/retina.wgsl`'s occlusion test: a
                // vertical cylinder of radius `BEING_RADIUS_M` in the (x, z) plane, height
                // skipped in v1 (the eye is at 0.4 m and the bodies are low). `hit_voxel`
                // cannot represent Being — `VoxelType` (src/world.rs:30-35) stays
                // Grass/Road/Wall/Hole only — so a hit is tracked in `hit_being` and the
                // material is decided once, below, when the pixel is finally written.
                let mut hit_being = false;
                if !other_beings.is_empty() {
                    let a = dir_x * dir_x + dir_z * dir_z;
                    if a > 1e-9 {
                        let r = crate::config::iara::vision::BEING_RADIUS_M;
                        for &op in other_beings {
                            let lx = eye_pos.x - op.x;
                            let lz = eye_pos.z - op.z;
                            let b = 2.0 * (lx * dir_x + lz * dir_z);
                            let c = lx * lx + lz * lz - r * r;
                            let disc = b * b - 4.0 * a * c;
                            if disc < 0.0 {
                                continue;
                            }
                            let sq = disc.sqrt();
                            let mut t = (-b - sq) / (2.0 * a);
                            if t < 0.0 {
                                t = (-b + sq) / (2.0 * a);
                                if t < 0.0 {
                                    continue;
                                }
                                t = 0.0; // eye already inside the other being's cylinder
                            }
                            // Nearest wins.
                            if t < hit_dist {
                                hit_dist = t;
                                hit_being = true;
                            }
                        }
                    }
                }

                let depth_fade = (hit_dist / 1200.0).clamp(0.0, 1.0);

                let pixel_val = match hit_voxel {
                    VoxelType::Wall => {
                        // Quad-scale depth (alcance estendido 600m):
                        if hit_dist < 30.0 {
                            let near = (hit_dist / 30.0).clamp(0.0, 1.0);
                            127.0 - (near.sqrt() * 72.0)
                        } else if hit_dist < 120.0 {
                            let mid = ((hit_dist - 30.0) / 90.0).clamp(0.0, 1.0);
                            55.0 - (mid * 55.0)
                        } else if hit_dist < 1200.0 {
                            let far = ((hit_dist - 120.0) / 480.0).clamp(0.0, 1.0);
                            0.0 - (far * 25.0)
                        } else {
                            let xfar = ((hit_dist - 1200.0) / 1200.0).clamp(0.0, 1.0);
                            -25.0 - (xfar * 25.0)
                        }
                    }
                    VoxelType::Grass => 40.0 - (depth_fade * 40.0),
                    VoxelType::Road => {
                        // Expanded range: shows curves more clearly via finer gradation.
                        // Before: -128→-80 (range 48). Now: -128→-32 (range 96).
                        // Extra 48 values lets retina encode curve shape as smooth transitions.
                        -128.0 + (depth_fade * 96.0)
                    }
                    VoxelType::Hole => {
                        // Hole = buraco no asfalto. Mata no contato → canal precisa ser
                        // BEM distinguível de Road/Grass/Wall. Range 90→60 em 0..50m,
                        // satura em 60 além. Sempre > Grass (40) → rede lê como hazard.
                        let near = (hit_dist / 50.0).clamp(0.0, 1.0);
                        90.0 - (near * 30.0)
                    }
                };

                // A being nearer than the DDA hit wins over whatever `hit_voxel` was left
                // holding (`VoxelType` has no Being variant to match on above), so the
                // canonical encoding — `gpu::legacy_encoding`, the same function
                // `gpu/retina.wgsl` is proven against — decides the byte here instead of
                // a fourth copy of its Being formula.
                pixels[index] = if hit_being {
                    crate::gpu::legacy_encoding(crate::gpu::Pixel {
                        dist_m: hit_dist,
                        material: crate::gpu::BEING_MATERIAL,
                    })
                } else {
                    pixel_val.clamp(-128.0, 127.0) as i8
                };
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
// Grid real = GRID_CELLS×GRID_CELLS = 320×320 cells = 960×960m (`VoxelWorld::new`,
// size). As pistas escritas à mão logo abaixo (Dubai 123, Interlagos 64, Monaco
// 256, Silverstone 512, Nürburgring 768, Suzuka 1024) e `pista_gen::tracado()`
// foram desenhadas para um grid MENOR (comentários antigos citam 240×240,
// centro 120) que já não existe — sobrevivem sem tocar a borda só porque o span
// delas (raio até ~110 cells) cabe dentro de [0,320] mesmo fora do centro
// verdadeiro. Não mexidas aqui (fora do escopo do debug de unidades 2026-08-15).
// Pistas NOVAS (perfil / currículo) devem usar `GRID_CELLS` como fonte única,
// não reinventar a constante — foi exatamente a duplicação (160 vs 320) que
// causou o recentro errado corrigido abaixo.
// ============================================================================
/// Metros → cells (3 m) + recentro no CENTRO GEOMÉTRICO real do grid
/// (`GRID_CELLS/2`); encolhe com aviso se não couber.
///
/// `wps_m` tem de estar em METROS de verdade — é o contrato de saída de
/// `pista_gen::gerar_perfil` (que converte a saída CELLS de `tracado()` para
/// metros antes de densificar; ver comentário lá). Achado 2026-08-15: esta
/// função dividia por CELL uma entrada que JÁ estava em cells (o bug estava em
/// `gerar_perfil`, não aqui) — a volta saía 3× menor (607m → 202m, seed 8005).
/// Certo agora: gerar_perfil devolve metros de verdade, aqui dividimos 1× só.
fn perfil_para_cells(wps_m: &[Vec2]) -> Vec<Vec2> {
    const CELL: f32 = 3.0;
    let mut wps: Vec<Vec2> = wps_m.iter().map(|p| *p / CELL).collect();
    let (mut min, mut max) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for p in &wps {
        min = min.min(*p);
        max = max.max(*p);
    }
    let span = max - min;
    // Margem de 20 cells (60m) pra cada lado — sobra pro raio de detecção de
    // parede e pro deslocamento lateral do spawn. Antes: "cabe=130" e
    // "centro_grid=80" eram calibrados pra um grid de 160 cells que não existe
    // mais (real = 320, `GRID_CELLS`); o recentro caía 80 cells (240m) fora do
    // centro verdadeiro (160,160), num canto do mundo real.
    let margem = 20.0;
    let cabe = GRID_CELLS - 2.0 * margem;
    let escala = (cabe / span.x.max(span.y)).min(1.0);
    if escala < 1.0 {
        eprintln!(
            "[pista] perfil não coube no grid (span {:.0}x{:.0} cells, limite {cabe:.0}) — encolhido {:.0}%",
            span.x,
            span.y,
            escala * 100.0
        );
    }
    let centro_atual = (min + max) * 0.5;
    let centro_grid = Vec2::splat(GRID_CELLS / 2.0);
    for p in wps.iter_mut() {
        *p = centro_grid + (*p - centro_atual) * escala;
    }
    wps
}

pub fn track_waypoints_for_seed(seed: u64) -> Vec<Vec2> {
    match seed {
        // Dubai Oval — EASY: oval largo, curvas suaves <45°, runway longo.
        // Coords shifted +40 cells (x,y) para centrar em grid 240×240.
        123 => vec![
            Vec2::new(60.0, 60.0),
            Vec2::new(160.0, 60.0),
            Vec2::new(180.0, 80.0),
            Vec2::new(180.0, 140.0),
            Vec2::new(160.0, 160.0),
            Vec2::new(80.0, 160.0),
            Vec2::new(60.0, 140.0),
            Vec2::new(60.0, 60.0),
        ],
        // Interlagos Gateway — MEDIUM: current layout (90° + 2×45°).
        64 => vec![
            Vec2::new(60.0, 60.0),
            Vec2::new(180.0, 60.0),
            Vec2::new(180.0, 100.0),
            Vec2::new(140.0, 140.0),
            Vec2::new(140.0, 180.0),
            Vec2::new(100.0, 180.0),
            Vec2::new(60.0, 140.0),
            Vec2::new(60.0, 60.0),
        ],
        // Monaco Tight — HARD: sequência de hairpins 90° sem reversos geométricos.
        // Antigo WP[3] fazia swing 135° (NW→E) — U-turn impossível pra rede shallow.
        // Agora monotonico em sentido horário: S→E→S→W→S→W→N→E.
        256 => vec![
            Vec2::new(60.0, 60.0),
            Vec2::new(150.0, 60.0),
            Vec2::new(180.0, 90.0),
            Vec2::new(180.0, 130.0),
            Vec2::new(140.0, 130.0),
            Vec2::new(140.0, 170.0),
            Vec2::new(100.0, 170.0),
            Vec2::new(60.0, 130.0),
            Vec2::new(60.0, 60.0),
        ],
        // Silverstone Flow — VERY HARD: curvas rápidas sem zigzag reverso.
        // Antigo fazia (110,115)→(80,100)→(60,130) U-turn SW→NW→SW. Removido.
        512 => vec![
            Vec2::new(60.0, 60.0),
            Vec2::new(140.0, 65.0),
            Vec2::new(180.0, 95.0),
            Vec2::new(180.0, 135.0),
            Vec2::new(155.0, 160.0),
            Vec2::new(115.0, 170.0),
            Vec2::new(75.0, 165.0),
            Vec2::new(55.0, 135.0),
            Vec2::new(70.0, 105.0),
            Vec2::new(95.0, 85.0),
            Vec2::new(60.0, 60.0),
        ],
        // Nürburgring Hills — EXTREME: elevação forte, curvas longas, sem U-turns.
        // Antigo (95,110)→(130,140) fazia reverso SW→SE. Removido.
        768 => vec![
            Vec2::new(60.0, 70.0),
            Vec2::new(130.0, 60.0),
            Vec2::new(180.0, 85.0),
            Vec2::new(180.0, 140.0),
            Vec2::new(150.0, 170.0),
            Vec2::new(110.0, 180.0),
            Vec2::new(70.0, 160.0),
            Vec2::new(60.0, 110.0),
            Vec2::new(60.0, 70.0),
        ],
        // Suzuka Crossing — INSANE: hairpins sucessivos SEM reversos. Antigo (90,50)→(70,70)
        // fazia SE→SW (U-turn) e (20,90)→(40,130) fazia SW→SE (outro U). Redesigned para
        // espiral horária com 90°s legítimos mas trajetória sempre forward.
        1024 => vec![
            Vec2::new(60.0, 70.0),
            Vec2::new(120.0, 70.0),
            Vec2::new(155.0, 95.0),
            Vec2::new(180.0, 130.0),
            Vec2::new(180.0, 170.0),
            Vec2::new(140.0, 185.0),
            Vec2::new(95.0, 180.0),
            Vec2::new(60.0, 155.0),
            Vec2::new(65.0, 115.0),
            Vec2::new(85.0, 95.0),
            Vec2::new(60.0, 70.0),
        ],
        // CURRÍCULO POR DIFICULDADE MEDIDA (ver `pista_gen.rs`).
        // Faixa de sementes própria, ≥ 1_000_000, para não mudar o comportamento de nenhuma
        // semente existente — o campeão foi treinado na 42 e as pistas à mão têm sementes
        // baixas. Formato: 1_000_000 + degrau*1000 + variante.
        //   degrau 1 exige a sonda de 20 m · 2 a de 40 m · 3 a de 70 m · 4 a de 120 m
        // Ex.: `--seed 1003007` = degrau 3 (difícil), variante 7.
        s if s >= 1_000_000 => {
            let degrau = (((s - 1_000_000) / 1000) as usize).clamp(1, 4);
            let variante = (s - 1_000_000) % 1000;
            // se não achar pista naquela faixa, cai no procedural em vez de devolver uma
            // mais fácil fingindo ser a pedida
            crate::pista_gen::do_curriculo(degrau, variante)
                .unwrap_or_else(|| procedural_waypoints(seed))
        }
        // `gerar_perfil` devolve METROS de verdade (ver comentário lá); `perfil_para_cells`
        // converte pra CELLS e recentra no centro geométrico do grid real (`GRID_CELLS`,
        // 320×320 — ver header desta seção). Se mesmo recentrada não couber, encolhe
        // uniformemente e AVISA (pista menor que a pedida é informação, nunca silêncio).
        // Bandas de PERFIL estrutural (Leonardo 2026-08-15): 8000-8099 curta,
        // 8100-8199 só-curvas, 8200-8299 retão — roteiam pro gerador parametrizado
        // (pista_gen::gerar_perfil), com fallback pro procedural comum se a banda
        // não conseguir geometria válida (informação no log, nunca silêncio).
        8000..=8099 => crate::pista_gen::gerar_perfil(seed, "curta", 28.0)
            .map(|wps| perfil_para_cells(&wps))
            .unwrap_or_else(|| {
                eprintln!(
                    "[pista] perfil curta s={seed}: sem geometria válida, caindo no procedural"
                );
                procedural_waypoints(seed)
            }),
        8100..=8199 => crate::pista_gen::gerar_perfil(seed, "curvas", 28.0)
            .map(|wps| perfil_para_cells(&wps))
            .unwrap_or_else(|| {
                eprintln!(
                    "[pista] perfil curvas s={seed}: sem geometria válida, caindo no procedural"
                );
                procedural_waypoints(seed)
            }),
        8200..=8299 => crate::pista_gen::gerar_perfil(seed, "retao", 28.0)
            .map(|wps| perfil_para_cells(&wps))
            .unwrap_or_else(|| {
                eprintln!(
                    "[pista] perfil retao s={seed}: sem geometria válida, caindo no procedural"
                );
                procedural_waypoints(seed)
            }),
        _ => procedural_waypoints(seed),
    }
}

/// Procedural rich track: scattered anchors + Catmull-Rom + VALIDATION layer.
/// Generates múltiplas tentativas (até 32 sub-seeds via splitmix), valida cada
/// uma por critério "pista boa" (lap 2500-3500m, max turn 75°, mistura de
/// curvas), retorna a primeira que passa. Garante variedade entre seeds e
/// rejeita layouts sem sentido.
pub fn procedural_waypoints(seed: u64) -> Vec<Vec2> {
    // P1.8: orçamento subiu de 32→96 tentativas (achado "piso de raio piora o pior
    // caso" — adicionar uma restrição geométrica nova ao `validate_track` só é
    // seguro se o gerador continuar achando candidato válido; mais tentativas
    // absorve a rejeição extra sem precisar pular a validação).
    for attempt in 0..96u64 {
        let sub_seed = seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(attempt.wrapping_mul(0xC1D7BEEF));
        let candidate = generate_track_candidate(sub_seed);
        if validate_track(&candidate) {
            return candidate;
        }
    }
    // Fallback se 96 tentativas falharem: NÃO é silencioso (achado "fallback
    // contradiz a própria garantia que o módulo declara" — o comentário antigo
    // dizia "garantido válido" mas o candidato abaixo nunca passa por
    // `validate_track`). Loga alto o bastante pra aparecer em qualquer log de
    // bateria; quem estiver rodando `--pistas` com essa seed vai ver isto ANTES
    // do gate de passabilidade (`validar_pista`) travar o processo.
    eprintln!(
        "[AVISO] procedural_waypoints(seed={seed}): 96 tentativas de geração falharam a \
         validate_track — caindo no candidato de reserva SEM validação geométrica. Esta \
         pista pode ter curva impossível (piso de raio) ou comprimento fora da faixa. \
         Rode `pista_check --seed {seed}` antes de treinar nela."
    );
    generate_track_candidate(seed.wrapping_add(0xDEADBEEF))
}

/// Gera candidato de pista — anchors + Catmull-Rom. Sem validação.
fn generate_track_candidate(seed: u64) -> Vec<Vec2> {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    let mut next = || {
        s = s.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = s;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z = z ^ (z >> 31);
        ((z >> 11) as f64 / (1u64 << 53) as f64) as f32
    };

    // Grid 320 → centro (160, 160).
    let cx = 160.0_f32;
    let cy = 160.0_f32;
    let easy = EASY_TRACK.load(Ordering::Relaxed);
    // EASY = pista PEQUENA real (não oval): 10-14 anchors, raio 50-100, mais variação.
    // NORMAL = pista grande complexa: 16-22 anchors, raio 70-150.
    let (n_anchors, r_min, r_max, sp) = if easy {
        (10 + (next() * 5.0) as usize, 50.0, 100.0, 35.0)
    } else {
        (16 + (next() * 7.0) as usize, 70.0, 150.0, 42.0)
    };
    let mut anchors: Vec<Vec2> = Vec::with_capacity(n_anchors);
    let mut tries = 0;
    while anchors.len() < n_anchors && tries < 1000 {
        tries += 1;
        let r = r_min + next() * (r_max - r_min);
        let theta = next() * std::f32::consts::TAU;
        let p = Vec2::new(cx + r * theta.cos(), cy + r * theta.sin());
        if p.x < 35.0 || p.x > 285.0 || p.y < 35.0 || p.y > 285.0 {
            continue;
        }
        if anchors.iter().any(|a| a.distance(p) < sp) {
            continue;
        }
        anchors.push(p);
    }
    if anchors.len() < 8 {
        anchors.clear();
        for i in 0..10 {
            let theta = i as f32 / 10.0 * std::f32::consts::TAU;
            anchors.push(Vec2::new(
                cx + 110.0 * theta.cos(),
                cy + 110.0 * theta.sin(),
            ));
        }
    }

    let centroid = anchors.iter().copied().fold(Vec2::ZERO, |a, b| a + b) / anchors.len() as f32;
    anchors.sort_by(|a, b| {
        let aa = (a.y - centroid.y).atan2(a.x - centroid.x);
        let ab = (b.y - centroid.y).atan2(b.x - centroid.x);
        aa.partial_cmp(&ab).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Variable spline tension per anchor (mix de curvas tight + open).
    let n = anchors.len();
    let n_samples = 8;
    let mut waypoints: Vec<Vec2> = Vec::with_capacity(n * n_samples + 1);
    for i in 0..n {
        let p0 = anchors[(i + n - 1) % n];
        let p1 = anchors[i];
        let p2 = anchors[(i + 1) % n];
        let p3 = anchors[(i + 2) % n];
        for j in 0..n_samples {
            let t = j as f32 / n_samples as f32;
            let t2 = t * t;
            let t3 = t2 * t;
            let pt = 0.5
                * ((2.0 * p1)
                    + (-p0 + p2) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3);
            waypoints.push(Vec2::new(pt.x.clamp(30.0, 290.0), pt.y.clamp(30.0, 290.0)));
        }
    }
    waypoints.push(waypoints[0]);
    waypoints
}

/// Validation: pista é "boa" se:
/// - lap entre 2500 e 4500m
/// - max turn ≤ 80° (sem hairpins impossíveis)
/// - <= 1 sequência de 3+ curvas seguidas >50°
/// - razão curva/reta razoável (varia entre seeds)
fn validate_track(waypoints: &[Vec2]) -> bool {
    if waypoints.len() < 8 {
        return false;
    }
    // Lap length em cells × 3m.
    let mut lap_len_cells = 0.0_f32;
    for w in waypoints.windows(2) {
        lap_len_cells += w[0].distance(w[1]);
    }
    let lap_m = lap_len_cells * 3.0;
    // Lap range geral.
    if EASY_TRACK.load(Ordering::Relaxed) {
        // FIX: range estava invertido (11200..3500 = empty) — typo histórico
        // que rejeitava TODAS pistas em --easy mode → procedural caía em fallback
        // simples. UI vs headless: DNA treinado em fallback, UI gera pista
        // real diferente → carros pareciam ruins.
        if !(1200.0..=3500.0).contains(&lap_m) {
            return false;
        }
    } else {
        if !(2500.0..=7000.0).contains(&lap_m) {
            return false;
        }
    }

    // Turn angles entre segmentos + turn-rate per meter (minecraft-style validation).
    let n = waypoints.len();
    let mut max_turn: f32 = 0.0;
    let mut sharp_turns = 0usize;
    let mut consecutive_sharp = 0usize;
    let mut max_consecutive = 0usize;
    let mut total_curve_deg = 0.0_f32;
    let mut max_turn_per_meter: f32 = 0.0;
    let mut min_seg_len_at_sharp: f32 = f32::MAX;
    // P1.8, achado "gerador procedural sem piso de raio": `validate_track` media
    // ângulo de virada e comprimento de segmento, mas nunca o RAIO da curva — uma
    // pista podia ter curva de raio de poucos metros (fisicamente impossível de
    // negociar em velocidade de corrida) e passar validação assim mesmo.
    let mut min_radius_m: f32 = f32::MAX;
    for i in 1..n - 1 {
        let v_in_raw = waypoints[i] - waypoints[i - 1];
        let v_out_raw = waypoints[i + 1] - waypoints[i];
        let v_in = v_in_raw.normalize_or_zero();
        let v_out = v_out_raw.normalize_or_zero();
        let dot = v_in.dot(v_out).clamp(-1.0, 1.0);
        let turn_rad = dot.acos();
        let turn = turn_rad.to_degrees();
        if turn > max_turn {
            max_turn = turn;
        }
        total_curve_deg += turn;

        // Turn-rate-per-meter: curva apertada em segmento curto = perigo.
        // Cars só conseguem dirigir se gradient suficiente.
        let seg_in_m = v_in_raw.length() * 3.0; // cells × 3m
        let seg_out_m = v_out_raw.length() * 3.0;
        let avg_seg = (seg_in_m + seg_out_m) * 0.5;
        if avg_seg > 0.5 {
            let tpm = turn / avg_seg;
            if tpm > max_turn_per_meter {
                max_turn_per_meter = tpm;
            }
            // Raio local ≈ arco/ângulo (aproximação de círculo osculador — mesma
            // conta que `pista_check.rs` usa para reportar "raio" por waypoint).
            if turn_rad > 1e-4 {
                let raio_m = avg_seg / turn_rad;
                if raio_m < min_radius_m {
                    min_radius_m = raio_m;
                }
            }
        }
        if turn > 50.0 {
            sharp_turns += 1;
            consecutive_sharp += 1;
            if consecutive_sharp > max_consecutive {
                max_consecutive = consecutive_sharp;
            }
            if avg_seg < min_seg_len_at_sharp {
                min_seg_len_at_sharp = avg_seg;
            }
        } else {
            consecutive_sharp = 0;
        }
    }
    // Curvas afiadas precisam segmento ≥ 12m (cars têm tempo de reagir).
    // Sem isso, layout = curva 60° em 5m = impossível.
    if sharp_turns > 0 && min_seg_len_at_sharp < 12.0 {
        return false;
    }
    // Turn rate ≤ 8°/m (curva 60° precisa segmento ≥ 7.5m).
    if max_turn_per_meter > 8.0 {
        return false;
    }
    // Piso de raio: mesmo piso que `pista_gen.rs::gerar` já usa (`d.r_min < 6.0`,
    // pista_gen.rs:220) — consistência entre os dois geradores em vez de um piso
    // arbitrário novo. Sementes procedurais fora da tabela nunca tinham essa
    // checagem; medido no build atual o raio mínimo típico fica em 7-10m, então
    // 6.0m é conservador (abaixo do observado) sem reabrir curvas hoje aceitas.
    if min_radius_m.is_finite() && min_radius_m < 6.0 {
        return false;
    }
    // Permite hairpins até 90° + sequências de 4 curvas (chicanes desafiadoras OK).
    // EASY mode: pista REAL com curvas reais. Sem ser oval.
    // Mantém variedade (sharp turns, S-curves) mas sem extremos.
    if EASY_TRACK.load(Ordering::Relaxed) {
        if max_turn > 85.0 {
            return false;
        } // hairpins ok mas não impossíveis
        if max_consecutive > 3 {
            return false;
        } // até 3 chicanes seguidas
        if sharp_turns < 3 || sharp_turns > 8 {
            return false;
        } // 3-8 curvas afiadas
        if total_curve_deg < 360.0 {
            return false;
        } // mínimo 1 volta integrada
        let avg_seg_m = (lap_len_cells * 3.0) / (n - 1) as f32;
        if avg_seg_m > 60.0 {
            return false;
        }
        return true;
    }

    // CEILING (descarta layouts impossíveis):
    if max_turn > 95.0 {
        return false;
    }
    if max_consecutive > 5 {
        return false;
    }

    // FLOOR (descarta layouts FÁCEIS DEMAIS):
    if total_curve_deg < 540.0 {
        return false;
    }
    if sharp_turns < 4 {
        return false;
    }
    if max_turn < 55.0 {
        return false;
    }
    let avg_seg_m = (lap_len_cells * 3.0) / (n - 1) as f32;
    if avg_seg_m > 50.0 {
        return false;
    }

    true
}

/// (amp, freq) — amplitude em metros, frequência em radianos/cell.
pub fn track_elevation_for_seed(seed: u64) -> (f32, f32) {
    match seed {
        123 => (0.8, 0.02),  // Dubai: quase flat
        64 => (1.5, 0.03),   // Interlagos: ondulação suave
        256 => (2.0, 0.04),  // Monaco: street-level bumps
        512 => (2.5, 0.035), // Silverstone: ondulações
        768 => (6.0, 0.025), // Nürburgring Hills: RAMPAS de verdade
        1024 => (3.5, 0.05), // Suzuka: relevo moderado
        _ => (1.5, 0.03),
    }
}

/// Cells (x,y) onde holes (buracos kill-on-touch) ficam sobrepostos ao Road.
/// Só pistas HARD+ recebem. Layout escolhido pra punir trajetória preguiçosa sem
/// bloquear a racing line ideal — buraco sempre fora do centro da pista ou em
/// trechos onde a IA deve já ter aprendido curvar.
///
/// `waypoints`: traçado REAL desta seed em cell-space (o mesmo que
/// `track_waypoints_for_seed` produziu para o mundo — cobre inclusive o
/// currículo, `do_curriculo`, cujo traçado é geometricamente distinto do que
/// `procedural_waypoints(seed)` geraria sozinho). `largura_cells`: largura da
/// pista em CÉLULAS (metros / cell_size), usada só pelo ramo procedural.
pub fn track_holes_for_seed(seed: u64, waypoints: &[Vec2], largura_cells: f32) -> Vec<Vec2> {
    match seed {
        // Dubai / Interlagos — sem buracos (fase de aprendizado).
        123 | 64 => vec![],

        // Monaco Tight — 2 buracos em apex de 90° interno.
        // Coords shifted +40 cells (x,y) para grid 240×240.
        256 => vec![
            Vec2::new(170.0, 95.0),  // Apex curva 2 (E→S)
            Vec2::new(138.0, 168.0), // Apex curva 5 (S→W)
        ],

        // Silverstone Flow — 3 buracos em trechos rápidos.
        512 => vec![
            Vec2::new(150.0, 67.0),  // Reta 1
            Vec2::new(178.0, 125.0), // Curva longa
            Vec2::new(72.0, 163.0),  // Saída W→N
        ],

        // Nürburgring Hills — 3 buracos em descidas (rampas amplificam).
        768 => vec![
            Vec2::new(170.0, 80.0),  // Descida 1
            Vec2::new(170.0, 155.0), // Vale
            Vec2::new(80.0, 165.0),  // Contra-subida
        ],

        // Suzuka Crossing — 4 buracos (insano: trajetória quase forçada).
        1024 => vec![
            Vec2::new(115.0, 70.0),  // Reta inicial
            Vec2::new(175.0, 120.0), // Curva descendente
            Vec2::new(155.0, 178.0), // Hairpin S
            Vec2::new(65.0, 135.0),  // Subida W
        ],

        _ => procedural_holes(seed, waypoints, largura_cells),
    }
}

/// Procedural hazards: K holes near apex of waypoint curves, deterministic per seed.
/// Inspired by water finding obstacles — placed at trajectory-critical points
/// (apex of sharpest curves, ~70% from segment start, deslocado para o lado
/// INTERNO da curva por uma fração da largura real da pista).
///
/// P1.8: recebe o traçado já calculado pelo chamador em vez de rodar
/// `procedural_waypoints(seed)` de novo — achado "buracos do currículo
/// calculados sobre traçado fantasma" (para seed≥1_000_000 o traçado real vem de
/// `do_curriculo`, um algoritmo geometricamente diferente de `procedural_waypoints`;
/// rodar os dois em paralelo produzia buracos sem relação com a pista que o carro
/// de fato percorre). `largura_cells`: usada para o offset lateral (achado "buracos
/// nascem em cima do eixo, deslocamento lateral zero").
pub fn procedural_holes(seed: u64, waypoints: &[Vec2], largura_cells: f32) -> Vec<Vec2> {
    if waypoints.len() < 4 {
        return vec![];
    }
    // Score each waypoint by turn angle; place hole near sharpest few apexes.
    let mut scored: Vec<(usize, f32)> = Vec::new();
    for i in 1..waypoints.len() - 1 {
        let v_in = (waypoints[i] - waypoints[i - 1]).normalize_or_zero();
        let v_out = (waypoints[i + 1] - waypoints[i]).normalize_or_zero();
        let dot = v_in.dot(v_out).clamp(-1.0, 1.0);
        let turn = dot.acos(); // 0=straight, π=u-turn
        scored.push((i, turn));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Seed-deterministic count: 1 or 2 holes (conservative — must not break learning curve).
    let n_holes = 1 + ((seed.wrapping_mul(0xA1B2C3D4) >> 32) as usize & 1);
    scored
        .iter()
        .take(n_holes)
        .map(|(i, _)| {
            // Position at 70% from prev → current (apex zone), deslocado lateralmente
            // para o lado interno da curva por 30% da largura real — antes o buraco
            // nascia exatamente EM CIMA do eixo (deslocamento lateral zero), podendo
            // bloquear a pista inteira em curvas fechadas/estreitas.
            let prev = waypoints[i.saturating_sub(1)];
            let cur = waypoints[*i];
            let next = waypoints[(*i + 1).min(waypoints.len() - 1)];
            let dir_in = (cur - prev).normalize_or_zero();
            let dir_out = (next - cur).normalize_or_zero();
            let normal = Vec2::new(-dir_in.y, dir_in.x);
            // Lado interno da curva: sinal do produto cruzado 2D entre entrada e saída.
            let lado = if dir_in.perp_dot(dir_out) > 0.0 {
                1.0
            } else {
                -1.0
            };
            prev + (cur - prev) * 0.7 + normal * lado * (largura_cells * 0.3)
        })
        .collect()
}

pub fn track_start_for_seed(seed: u64) -> Vec2 {
    match seed {
        123 => Vec2::new(95.0, 60.0),  // Dubai: runway largo
        64 => Vec2::new(90.0, 60.0),   // Interlagos
        256 => Vec2::new(80.0, 60.0),  // Monaco: start curto (pista técnica)
        512 => Vec2::new(95.0, 62.0),  // Silverstone
        768 => Vec2::new(95.0, 68.0),  // Nürburgring
        1024 => Vec2::new(80.0, 70.0), // Suzuka
        // Achado 2026-08-15 (debug de unidades): currículo (≥1_000_000) e perfil
        // (8000-8299) caíam no braço `_` abaixo, que chama `procedural_waypoints(seed)`
        // — uma pista candidata TOTALMENTE diferente da que `track_waypoints_for_seed`
        // de fato construiu pro mundo (do_curriculo / gerar_perfil). O carro não nascia
        // fora da pista de verdade (start_progress ainda projeta no traçado real, ver
        // `VoxelWorld::new`), mas a largada saía num ponto arbitrário da volta, gastava
        // até 96 tentativas de `generate_track_candidate` à toa, e disparava o aviso
        // "96 tentativas... caindo no candidato de reserva" em TODO boot dessas sementes
        // — ruído que quase escondeu o bug real desta investigação. Roteando pro
        // traçado que o mundo de fato usa.
        s if s >= 1_000_000 || (8000..=8299).contains(&s) => {
            let wp = track_waypoints_for_seed(seed);
            if wp.len() >= 2 {
                (wp[0] + wp[1]) * 0.5
            } else {
                Vec2::new(50.0, 20.0)
            }
        }
        // Procedural: spawn at midpoint between waypoint[0] and waypoint[1] (forward into track).
        _ => {
            let wp = procedural_waypoints(seed);
            if wp.len() >= 2 {
                (wp[0] + wp[1]) * 0.5
            } else {
                Vec2::new(50.0, 20.0)
            }
        }
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
            min_dist_to_road: Vec::new(),
            cell_size: 0.0,
            start_pos: Vec3::ZERO,
            lap_distance: 0.0,
            waypoints_world: Vec::new(),
            waypoint_arcs: Vec::new(),
            start_progress: 0.0,
            racing_line: crate::simulation::racing_line::RacingLine::default(),
            proposals: Vec::new(),
            proposal_valid: true,
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
        for &(x, z) in &[
            (0.0, 0.0),
            (-500.0, -500.0),
            (1000.0, 1000.0),
            (50.0, -100.0),
        ] {
            assert!(
                w.get_track_progress(x, z) >= 0.0,
                "progress < 0 em ({}, {})",
                x,
                z
            );
        }
    }

    #[test]
    fn lap_distance_is_positive() {
        let w = ref_world();
        assert!(
            w.lap_distance > 0.0,
            "lap_distance deve ser positivo para pista real"
        );
    }

    #[test]
    fn track_progress_near_start_is_small() {
        let w = ref_world();
        // Na própria largada, progresso deve ser ~0 (até 2m de tolerância pela quantização do grid).
        let p = w.get_track_progress(w.start_pos.x, w.start_pos.z);
        assert!(p < 2.0, "progresso na largada = {}, esperado ~0", p);
    }

    /// `docs/RETINA-BEINGS.md` §3.2: "off is bit-identical" — an empty `other_beings`
    /// slice must reproduce this function's pre-`--retina-beings` behaviour exactly.
    ///
    /// Compares the empty-slice call against one with a being 1,000,000 m away: the
    /// occlusion code runs (the slice is non-empty) but every ray's intersection test
    /// must come back empty at that range, so if the two results ever differ, the
    /// occlusion block is not properly gated by `other_beings.is_empty()`.
    #[test]
    fn retina_beings_off_is_bit_identical_to_before() {
        let w = ref_world();
        let pos = Vec3::new(w.start_pos.x, 0.0, w.start_pos.z);
        let forward = Vec3::new(1.0, 0.0, 0.0);
        let right = Vec3::new(0.0, 0.0, 1.0);

        let off = w.render_binocular_retina(pos, forward, right, 0.0, &[]);
        let far_away = Vec3::new(pos.x + 1_000_000.0, 0.0, pos.z);
        let on_but_irrelevant = w.render_binocular_retina(pos, forward, right, 0.0, &[far_away]);
        assert_eq!(
            off, on_but_irrelevant,
            "a being far outside any ray's reach changed the retina — the occlusion \
             block is not properly gated"
        );
        assert!(
            off.iter().all(|&v| v != -29),
            "the beings-off retina must never encode a Being pixel (-29)"
        );
    }
}
