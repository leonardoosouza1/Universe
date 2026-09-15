// Physical ray geometry, composed with retina.wgsl and its shared grid declarations.
// Returns (distance in metres, material). Hole (3) is ground material, never an opaque wall.

const GEOMETRY_MAX_DISTANCE: f32 = 1200.0;
const GEOMETRY_MAX_STEPS: i32 = 1024;

fn geometry_cell_index(coordinate: f32) -> i32 {
    // Match physics rounding, including negative half-cell boundary ties.
    return round_away_from_zero(coordinate / params.cell_size);
}

fn geometry_ground_voxel(wx: f32, wz: f32) -> u32 {
    let x = geometry_cell_index(wx);
    let z = geometry_cell_index(wz);
    let n = i32(params.grid_size);
    if (x >= 0 && z >= 0 && x < n && z < n) {
        return grid[u32(z * n + x)];
    }
    return 2u;
}

fn trace_surface(eye: vec3<f32>, direction: vec3<f32>) -> vec2<f32> {
    let horizontal = vec2<f32>(direction.x, direction.z);
    let horizontal_len = length(horizontal);
    let ground = select(1e30, max(0.0, -eye.y / direction.y), direction.y < -1e-6);
    let limit = min(ground, GEOMETRY_MAX_DISTANCE);
    if (geometry_ground_voxel(eye.x, eye.z) == 2u) {
        return vec2<f32>(0.0, 2.0);
    }
    if (horizontal_len <= 1e-6) {
        let material = select(1u, geometry_ground_voxel(eye.x, eye.z), ground <= GEOMETRY_MAX_DISTANCE);
        return vec2<f32>(limit, f32(material));
    }
    let ray = horizontal / horizontal_len;
    // World cell centres are at index * cell_size; +0.5 moves boundaries to integers.
    let sx = eye.x / params.cell_size + 0.5;
    let sz = eye.z / params.cell_size + 0.5;
    var x = geometry_cell_index(eye.x);
    var z = geometry_cell_index(eye.z);
    let dx = select(1e30, abs(1.0 / ray.x), abs(ray.x) > 1e-6);
    let dz = select(1e30, abs(1.0 / ray.y), abs(ray.y) > 1e-6);
    let stepx = select(1, -1, ray.x < 0.0);
    let stepz = select(1, -1, ray.y < 0.0);
    var side_x = select(f32(x) + 1.0 - sx, sx - f32(x), ray.x < 0.0) * dx;
    var side_z = select(f32(z) + 1.0 - sz, sz - f32(z), ray.y < 0.0) * dz;
    for (var i: i32 = 0; i < GEOMETRY_MAX_STEPS; i++) {
        var side_distance: f32;
        if (side_x < side_z) { side_distance = side_x; side_x += dx; x += stepx; }
        else { side_distance = side_z; side_z += dz; z += stepz; }
        let distance = side_distance * params.cell_size / horizontal_len;
        if (distance > limit) { break; }
        // Hole is intentionally skipped: it only appears at the ground intercept below.
        if (geometry_ground_voxel(f32(x) * params.cell_size, f32(z) * params.cell_size) == 2u) {
            return vec2<f32>(distance, 2.0);
        }
    }
    if (ground <= GEOMETRY_MAX_DISTANCE) {
        return vec2<f32>(ground, f32(geometry_ground_voxel(
            eye.x + direction.x * ground, eye.z + direction.z * ground)));
    }
    return vec2<f32>(GEOMETRY_MAX_DISTANCE, 1.0);
}
