use bevy::prelude::*;
use crate::config::CITY_BLOCK_CELLS;
use crate::world::VoxelType;
use std::collections::VecDeque;

/// Traffic light state machine phase
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrafficPhase {
    NsGreen,
    NsYellow,
    EwGreen,
    EwYellow,
}

/// Single traffic light at an intersection
#[derive(Clone, Copy, Debug)]
pub struct TrafficLight {
    pub phase: TrafficPhase,
    pub timer: f32,
    pub cycle_ns: f32,   // seconds green for North-South roads
    pub cycle_ew: f32,   // seconds green for East-West roads
    pub cycle_yellow: f32, // yellow phase duration
}

impl TrafficLight {
    fn new() -> Self {
        Self {
            phase: TrafficPhase::NsGreen,
            timer: 0.0,
            cycle_ns: 15.0,
            cycle_ew: 15.0,
            cycle_yellow: 3.0,
        }
    }

    fn tick(&mut self, dt: f32) {
        self.timer += dt;
        let cycle_total = self.cycle_ns + self.cycle_yellow + self.cycle_ew + self.cycle_yellow;

        self.phase = if self.timer < self.cycle_ns {
            TrafficPhase::NsGreen
        } else if self.timer < self.cycle_ns + self.cycle_yellow {
            TrafficPhase::NsYellow
        } else if self.timer < self.cycle_ns + self.cycle_yellow + self.cycle_ew {
            TrafficPhase::EwGreen
        } else {
            TrafficPhase::EwYellow
        };

        if self.timer >= cycle_total {
            self.timer = 0.0;
        }
    }

    pub fn is_red_for_north_south(&self) -> bool {
        matches!(self.phase, TrafficPhase::EwGreen | TrafficPhase::EwYellow)
    }

    pub fn is_red_for_east_west(&self) -> bool {
        matches!(self.phase, TrafficPhase::NsGreen | TrafficPhase::NsYellow)
    }
}

/// Single intersection in the city
#[derive(Clone, Debug)]
pub struct Intersection {
    pub cell_x: usize,
    pub cell_y: usize,
    pub world_pos: Vec2,
    pub light: TrafficLight,
}

/// City navigation state — entire grid system
pub struct CityNavState {
    pub intersections: Vec<Intersection>,
    pub spawn_points: Vec<Vec2>,
    pub grid_size: usize,           // 320 (world cells)
    pub cell_size: f32,             // 3.0 m
    pub block_size_cells: usize,    // CITY_BLOCK_CELLS = 32
}

impl CityNavState {
    /// Generate a city grid. Modifies the VoxelWorld grid and flow field in-place.
    pub fn generate(
        grid: &mut Vec<VoxelType>,
        flow: &mut Vec<Vec2>,
        grid_size: usize,
        cell_size: f32,
        num_blocks: usize,
    ) -> Self {
        let block_size = CITY_BLOCK_CELLS;
        let street_width = 3;  // cells, = 9m roads

        // Clear grid
        for cell in grid.iter_mut() {
            *cell = VoxelType::Grass;
        }

        let mut intersections = Vec::new();
        let mut spawn_points = Vec::new();

        // Lay out streets and avenues
        for block_row in 0..num_blocks {
            for block_col in 0..num_blocks {
                let row_start = block_row * block_size;
                let col_start = block_col * block_size;

                // Streets (horizontal roads, every block_size rows)
                if row_start + street_width <= grid_size {
                    for r in row_start..row_start + street_width {
                        for c in 0..grid_size {
                            let idx = r * grid_size + c;
                            if idx < grid.len() {
                                grid[idx] = VoxelType::Road;
                            }
                        }
                    }
                }

                // Avenues (vertical roads, every block_size cols)
                if col_start + street_width <= grid_size {
                    for c in col_start..col_start + street_width {
                        for r in 0..grid_size {
                            let idx = r * grid_size + c;
                            if idx < grid.len() {
                                grid[idx] = VoxelType::Road;
                            }
                        }
                    }
                }
            }
        }

        // Spawn points — random cells on the roads (start + some buffer inward)
        for block_row in 0..num_blocks {
            for block_col in 0..num_blocks {
                let row_start = block_row * block_size;
                let col_start = block_col * block_size;

                // Pick a random cell in the street boundaries of this block
                let spawn_r = row_start + (street_width / 2);
                let spawn_c = col_start + (street_width / 2);
                if spawn_r < grid_size && spawn_c < grid_size {
                    let world_x = (spawn_c as f32 + 0.5) * cell_size;
                    let world_z = (spawn_r as f32 + 0.5) * cell_size;
                    spawn_points.push(Vec2::new(world_x, world_z));
                }
            }
        }

        // Intersections at each street×avenue crossing
        for block_row in 1..num_blocks {
            for block_col in 1..num_blocks {
                let inter_r = block_row * block_size;
                let inter_c = block_col * block_size;
                if inter_r < grid_size && inter_c < grid_size {
                    let world_x = (inter_c as f32 + 1.5) * cell_size;
                    let world_z = (inter_r as f32 + 1.5) * cell_size;
                    intersections.push(Intersection {
                        cell_x: inter_c,
                        cell_y: inter_r,
                        world_pos: Vec2::new(world_x, world_z),
                        light: TrafficLight::new(),
                    });
                }
            }
        }

        // BFS flood-fill to compute flow field (point toward nearest intersection)
        let mut visited = vec![false; grid.len()];
        let mut queue = VecDeque::new();

        // Seed queue with intersection cells
        for inter in &intersections {
            let idx = inter.cell_y * grid_size + inter.cell_x;
            if idx < visited.len() {
                visited[idx] = true;
                queue.push_back((inter.cell_x, inter.cell_y));
            }
        }

        // BFS to label all road cells with direction to nearest intersection
        while let Some((cx, cy)) = queue.pop_front() {
            let idx = cy * grid_size + cx;

            // For each unvisited neighbor
            for (dx, dy) in &[(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let nx = (cx as i32 + dx) as usize;
                let ny = (cy as i32 + dy) as usize;
                if nx < grid_size && ny < grid_size {
                    let n_idx = ny * grid_size + nx;
                    if !visited[n_idx] && grid[n_idx] == VoxelType::Road {
                        visited[n_idx] = true;
                        queue.push_back((nx, ny));
                        // Set flow from (nx,ny) back toward (cx,cy)
                        let flow_x = (cx as f32 - nx as f32).signum();
                        let flow_y = (cy as f32 - ny as f32).signum();
                        flow[n_idx] = Vec2::new(flow_x, flow_y).normalize_or_zero();
                    }
                }
            }
        }

        // Set all grass/wall cells to flow zero (only roads have flow)
        for i in 0..grid.len() {
            if grid[i] != VoxelType::Road {
                flow[i] = Vec2::ZERO;
            }
        }

        Self {
            intersections,
            spawn_points,
            grid_size,
            cell_size,
            block_size_cells: block_size,
        }
    }

    /// Advance all traffic lights by dt
    pub fn tick(&mut self, dt: f32) {
        for inter in &mut self.intersections {
            inter.light.tick(dt);
        }
    }

    /// Find nearest traffic light within 30m, return (distance, phase)
    pub fn nearest_traffic_light(&self, pos: Vec2) -> Option<(f32, TrafficPhase)> {
        const SEARCH_RADIUS: f32 = 30.0;
        let mut closest: Option<(f32, TrafficPhase)> = None;
        for inter in &self.intersections {
            let dist = pos.distance(inter.world_pos);
            if dist < SEARCH_RADIUS {
                if closest.is_none() || dist < closest.unwrap().0 {
                    closest = Some((dist, inter.light.phase));
                }
            }
        }
        closest
    }

    /// Pick a random destination at least 200m from current position
    pub fn random_destination(&self, pos: Vec2, rng: &mut impl rand::Rng) -> Vec2 {
        use rand::Rng;
        const MIN_DIST: f32 = 200.0;
        if self.spawn_points.is_empty() {
            return pos;
        }
        loop {
            let idx = rng.random_range(0..self.spawn_points.len());
            let candidate = self.spawn_points[idx];
            if pos.distance(candidate) >= MIN_DIST {
                return candidate;
            }
        }
    }
}

