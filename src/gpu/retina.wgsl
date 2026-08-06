// RETINA NA GPU — devolve FÍSICA, não julgamento.
//
// 14 linhas × 126 colunas × N seres = 1764·N raios independentes por tick. Com 60 seres são
// 105.840 marchas DDA, todas sem dependência entre si: o formato canônico de trabalho para
// GPU. Na CPU isso custava 57% de todo o tempo de simulação (2440 amostras de PERF), contra
// 14% do cérebro. Medido aqui: 40,5× mais rápido que a CPU de uma thread.
//
// # O que este kernel devolve, e por quê
//
// Duas saídas por pixel: **distância em metros** e **material**. Cru. É o que um lidar
// entrega e é o que a natureza entrega — fótons da coisa opaca mais próxima.
//
// A versão anterior devolvia um único `i8` por pixel, produzido por uma curva de salência
// escrita à mão. Medido, ela:
//
//   * normalizava por 1200 m num raio que a geometria trava em 33,3 m (a linha 0 aponta para
//     baixo com pitch −0,012 e bate no CHÃO ali; o código então descarta qualquer parede mais
//     distante). Resultado: **Road usava 4 dos 256 valores, Grass usava 3**;
//   * punha Wall no topo da escala (127..53) e Road no fundo (−125..−128), e o consumo é por
//     `max`-pool — então **a pista desaparecia de qualquer bin que tivesse uma parede**. Havia
//     um rastreador separado (canais 90-93) procurando o MÍNIMO só para recuperar o que o
//     max-pool apagava: remendo em cima de viés.
//
// Leonardo: *"a rede precisa se virar, a gente só dá as informações"*. Então a decisão de o
// que importa sai daqui e vai para a evolução, que é onde a tese diz que ela pertence.
//
// A codificação antiga continua reconstruível no host (`legacy_encoding`) para que a
// equivalência com a CPU siga demonstrável e a comparação física × codificada seja possível.

struct Params {
    grid_size:  u32,
    n_beings:    u32,
    cell_size:  f32,
    fov_half:   f32,
    pitches:    array<vec4<f32>, 4>,  // 14 pitches em 4 vec4
}

struct Pose {
    pos_gaze: vec4<f32>,   // xyz posição, w gaze_yaw
    dirs:     vec4<f32>,   // forward.xz, right.xz
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> grid: array<u32>;
@group(0) @binding(2) var<storage, read> poses: array<Pose>;
// Empacotado: bits 0-7 = material, bits 8-31 = distância em MILÍMETROS.
// Milímetro cabe folgado (alcance máximo 1200 m = 1.200.000 mm > 2^24? não: 2^24 = 16,7M, cabe)
// e evita um segundo buffer e uma segunda cópia de volta.
@group(0) @binding(3) var<storage, read_write> output: array<u32>;

const RETINA_ROWS: u32 = 14u;
const RETINA_COLS: u32 = 126u;
const RETINA_TOTAL: u32 = 1764u;
const TAN_K: f32 = 1.25;
const EYE_HEIGHT: f32 = 0.4;

fn voxel_at(wx: f32, wz: f32) -> u32 {
    // A CPU usa `.round()`, NÃO `.floor()` — trocar desloca a leitura em meia célula.
    let x = i32(round(wx / params.cell_size));
    let y = i32(round(wz / params.cell_size));
    let n = i32(params.grid_size);
    if (x >= 0 && x < n && y >= 0 && y < n) {
        return grid[u32(y * n + x)];
    }
    return 2u; // fora da grade é Wall, igual à CPU
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_beings * RETINA_TOTAL) { return; }

    let being = idx / RETINA_TOTAL;
    let pix = idx % RETINA_TOTAL;
    let row = pix / RETINA_COLS;
    let col = pix % RETINA_COLS;

    let p = poses[being];
    let eye = vec3<f32>(p.pos_gaze.x, p.pos_gaze.y + EYE_HEIGHT, p.pos_gaze.z);
    let fwd = vec2<f32>(p.dirs.x, p.dirs.y);
    let rgt = vec2<f32>(p.dirs.z, p.dirs.w);

    let pitch = params.pitches[row / 4u][row % 4u];
    let gaze = select(0.0, p.pos_gaze.w, row == 0u);
    let dir_y = sin(pitch);
    let cos_pitch = cos(pitch);

    let normalized_i = (f32(col) / (f32(RETINA_COLS) - 1.0)) * 2.0 - 1.0;
    var yaw_angle: f32;
    if (row == 0u) {
        // Fóvea: deformação por tangente, densa no centro.
        yaw_angle = (tan(normalized_i * TAN_K) / tan(TAN_K)) * params.fov_half + gaze;
    } else {
        yaw_angle = normalized_i * params.fov_half;
    }

    let dir_x = (fwd.x * cos(yaw_angle) + rgt.x * sin(yaw_angle)) * cos_pitch;
    let dir_z = (fwd.y * cos(yaw_angle) + rgt.y * sin(yaw_angle)) * cos_pitch;

    var hit_voxel: u32 = 1u;   // Road
    var hit_dist: f32 = 1200.0;
    let dist_to_ground = select(1000.0, -eye.y / dir_y, dir_y < -0.001);

    let ray_start = vec2<f32>(eye.x / params.cell_size, eye.z / params.cell_size);
    var ray_dir = vec2<f32>(dir_x, dir_z);
    let comp = length(ray_dir);
    if (comp > 1e-9) { ray_dir = ray_dir / comp; } else { ray_dir = vec2<f32>(0.0, 0.0); }

    var map_x = i32(floor(ray_start.x));
    var map_y = i32(floor(ray_start.y));
    let delta_x = select(abs(1.0 / ray_dir.x), 1e30, abs(ray_dir.x) < 1e-5);
    let delta_y = select(abs(1.0 / ray_dir.y), 1e30, abs(ray_dir.y) < 1e-5);

    var step_x: i32; var side_dist_x: f32;
    if (ray_dir.x < 0.0) {
        step_x = -1; side_dist_x = (ray_start.x - f32(map_x)) * delta_x;
    } else {
        step_x = 1; side_dist_x = ((f32(map_x) + 1.0) - ray_start.x) * delta_x;
    }
    var step_y: i32; var side_dist_y: f32;
    if (ray_dir.y < 0.0) {
        step_y = -1; side_dist_y = (ray_start.y - f32(map_y)) * delta_y;
    } else {
        step_y = 1; side_dist_y = ((f32(map_y) + 1.0) - ray_start.y) * delta_y;
    }

    var hit = false;
    var side: i32 = 0;
    for (var k: i32 = 0; k < 400; k = k + 1) {
        if (side_dist_x < side_dist_y) {
            side_dist_x = side_dist_x + delta_x; map_x = map_x + step_x; side = 0;
        } else {
            side_dist_y = side_dist_y + delta_y; map_y = map_y + step_y; side = 1;
        }
        let v = voxel_at(f32(map_x) * params.cell_size, f32(map_y) * params.cell_size);
        if (v == 2u || v == 3u) { hit_voxel = v; hit = true; break; }
    }

    if (hit) {
        var d: f32;
        if (side == 0) {
            d = ((f32(map_x) - ray_start.x + f32(1 - step_x) / 2.0) / ray_dir.x) * params.cell_size;
        } else {
            d = ((f32(map_y) - ray_start.y + f32(1 - step_y) / 2.0) / ray_dir.y) * params.cell_size;
        }
        // O chão intercepta antes da parede: é isso que trava a visão em 33,3 m na linha 0.
        // Continua valendo, porque é GEOMETRIA, não julgamento de salência.
        if (dir_y < -0.001 && d > dist_to_ground) {
            hit_dist = dist_to_ground;
            hit_voxel = voxel_at(eye.x + dir_x * hit_dist, eye.z + dir_z * hit_dist);
        } else {
            hit_dist = d;
        }
    } else if (dir_y < -0.001 && dist_to_ground <= 1200.0) {
        hit_dist = dist_to_ground;
        hit_voxel = voxel_at(eye.x + dir_x * hit_dist, eye.z + dir_z * hit_dist);
    }

    let mm = u32(clamp(hit_dist, 0.0, 1200.0) * 1000.0);
    output[idx] = (mm << 8u) | (hit_voxel & 0xFFu);
}
