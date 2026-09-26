//! Opt-in research settings. Validation precedes GPU allocation; units are explicit.
use universe::gpu::school::{Lesson, LessonGeometry};
use universe::world::{VoxelType, VoxelWorld};

#[derive(Clone, Copy, Debug)]
pub struct StartDistanceMetres(f32);
impl StartDistanceMetres {
    pub fn parse(text: &str) -> Result<Self, String> {
        let value = text
            .parse::<f32>()
            .map_err(|_| "start distance must be metres".to_owned())?;
        if !value.is_finite() || !(0.0..=120.0).contains(&value) {
            return Err("start distance must be finite and within 0..=120 metres".into());
        }
        Ok(Self(value))
    }
    pub fn get(self) -> f32 {
        self.0
    }
}

/// Prepared starts are a curriculum intervention, never an autonomous-service claim.
/// A short swept road check rejects geometry that the point-only test would accept.
pub fn prepare_starts(
    world: &VoxelWorld,
    starts: &mut [LessonGeometry],
    distance: StartDistanceMetres,
) -> Result<(), String> {
    for g in starts {
        let (goal, direction) = match g.lesson {
            Lesson::Pickup => (g.target, g.forward),
            Lesson::Delivery => (g.exit, g.exit_dir),
            _ => return Err("prepared starts support pickup and delivery only".into()),
        };
        // Preserve the canonical lane offset. At distance zero the car is beside the
        // station, inside the taxi service radius, with a legal lane heading.
        let lateral = [-direction[1], direction[0]];
        let offset = if direction[0].abs() > 0.5 { -1.0 } else { 1.0 };
        let start = [
            goal[0] - direction[0] * distance.0 + lateral[0] * world.cell_size * offset,
            goal[1] - direction[1] * distance.0 + lateral[1] * world.cell_size * offset,
        ];
        let count = (distance.0 / (world.cell_size * 0.5)).ceil() as usize;
        for step in 0..=count {
            let along = (step as f32 * world.cell_size * 0.5).min(distance.0);
            for longitudinal in [-2.25_f32, 0.0, 2.25] {
                for side in [-1.0_f32, 0.0, 1.0] {
                    let x = start[0] + direction[0] * (along + longitudinal) + lateral[0] * side;
                    let z = start[1] + direction[1] * (along + longitudinal) + lateral[1] * side;
                    let col = (x / world.cell_size).round() as i32;
                    let row = (z / world.cell_size).round() as i32;
                    if col < 0
                        || row < 0
                        || col >= world.size as i32
                        || row >= world.size as i32
                        || world.grid[row as usize * world.size + col as usize] != VoxelType::Road
                    {
                        return Err(format!(
                            "prepared {} start or swept approach is off road",
                            g.lesson.name()
                        ));
                    }
                }
            }
        }
        g.start = start;
        g.forward = direction;
        g.depot_distance_m = (start[0] - world.start_pos.x).hypot(start[1] - world.start_pos.z);
    }
    Ok(())
}

pub fn configuration(
    distance: Option<StartDistanceMetres>,
    linear: bool,
    retention: usize,
    witness: bool,
    linear_head: bool,
) -> serde_json::Value {
    serde_json::json!({"schema_version":2,"start_distance_m":distance.map(StartDistanceMetres::get),
        "prepared_start":distance.is_some(),"initial_speed_mps":0.0,
        "controller":if witness {universe::gpu::taxi_witness::TAXI_WITNESS_ID} else if linear {"linear-wave-v2"} else {"native"},
        "learning_evidence":!witness, "intent_producer":if witness {"taxi_witness"} else {"brain"},
        "linear_equation":if linear {Some("intent[j]=tanh(sum(l1[j,c]*post_noise_wave[c]/127)); j<9; other intents=0")} else {None},
        "retention_slots":retention,"inference_device":"gpu",
        "motor_head":if witness || linear_head {"linear"} else {"genome_spline"},
        "wave_source":if witness {"resident_percept_plus_privileged_body_task_routes"} else {"resident_percept_with_sensor_noise"},
        "evaluation_claim":if witness {"scripted feasibility only; not learning evidence"} else {"prepared success is not autonomous service"}})
}

/// Strict increasing distances followed by ordinary autonomous starts (training continues).
pub fn parse_curriculum(text: &str) -> Result<Vec<Option<StartDistanceMetres>>, String> {
    let parts: Vec<_> = text.split(',').collect();
    if parts.len() < 2 || parts.len() > 8 || parts.last() != Some(&"native") {
        return Err(
            "curriculum requires 1..7 increasing metre distances followed by native".into(),
        );
    }
    let mut stages = Vec::new();
    let mut previous = -1.0;
    for part in &parts[..parts.len() - 1] {
        let d = StartDistanceMetres::parse(part)?;
        if d.get() <= previous {
            return Err("curriculum distances must strictly increase".into());
        }
        previous = d.get();
        stages.push(Some(d));
    }
    stages.push(None);
    Ok(stages)
}
