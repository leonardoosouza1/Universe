//! Optional driving evidence: exact per-tick contacts and sampled sensory/state data.
//!
//! Contacts arrive in the existing physics readback. Only the selected being's
//! retina, perception and fast state are copied at the configured display cadence.
//! No metric is fed back into the controller by this observer.

use crate::native_action::MotorTrace;
use crate::statics::{
    NEUROMOD_REGIOES, continuous_wall_probes, motor_head_linear, reverse_controller_v2,
    slow_canonical_kinematics, throttle_arbitration_signed,
};
use crate::types::BeingState;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use universe::config::iara::brain::{L1_NEURONS, SENSORY_WAVE_SIZE};
use universe::config::iara::vision::{RETINA_COLS, RETINA_ROWS, RETINA_TOTAL};
use universe::gpu::brain::{BrainGpu, OBSERVATION_BYTES, decode_observation};
use universe::gpu::evolution::Genome;
use universe::gpu::genome_io::save_genome;
use universe::gpu::physics::PhysicsStateGpu;
use universe::gpu::slow_brain::SlowBrainGpu;

const WAVE_BYTES: usize = SENSORY_WAVE_SIZE;
const FAST_FLOATS: usize = L1_NEURONS;
const RETINA_BYTES: usize = RETINA_TOTAL * 4;
const FAST_OFFSET: usize = RETINA_BYTES + WAVE_BYTES;
const TRACE_OFFSET: usize = FAST_OFFSET + FAST_FLOATS * 4;
const LIFE_OFFSET: usize = TRACE_OFFSET + OBSERVATION_BYTES;
const BASE_SAMPLE_BYTES: usize = LIFE_OFFSET + universe::gpu::lifetime::LIFETIME_STRIDE * 4;
const SLOW_INPUT_OFFSET: usize = BASE_SAMPLE_BYTES;
const SLOW_INPUT_FLOATS: usize = 48;
const SLOW_STATE_OFFSET: usize = SLOW_INPUT_OFFSET + SLOW_INPUT_FLOATS * 4;
const SLOW_STATE_FLOATS: usize = 133;
const SLOW_OUTPUT_OFFSET: usize = SLOW_STATE_OFFSET + SLOW_STATE_FLOATS * 4;
const SLOW_OUTPUT_FLOATS: usize = 3;
const SAMPLE_BYTES: usize = SLOW_OUTPUT_OFFSET + SLOW_OUTPUT_FLOATS * 4;

#[derive(Default, Clone)]
struct Window {
    ticks: u64,
    alive: u64,
    contact: u64,
    impacts: u64,
    events: u64,
    brake: u64,
    overlap: u64,
    stationary: u64,
    reverse: u64,
    near_wall: u64,
    speed_sum: f64,
    distance: f64,
    energy: f64,
    impulse: f64,
    impact_peak: f32,
    slip_peak: f32,
    slip_angle_peak: f32,
    throttle_sum: f64,
    brake_sum: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct ServiceSnapshot {
    pub task_available: bool,
    pub onboard: bool,
    pub task_key: f32,
    pub passenger_id: f32,
    pub deliveries: u32,
    pub pickup: [f32; 2],
    pub destination: [f32; 2],
    pub refusal_code: u32,
}

impl ServiceSnapshot {
    fn from_body(body: &universe::gpu::organism::OrganismState) -> Self {
        let v = &body.values;
        Self {
            task_available: v[90] > 0.5,
            onboard: v[65] > 0.5,
            task_key: v[31],
            passenger_id: v[72],
            deliveries: v[68].max(0.0) as u32,
            pickup: [v[70], v[71]],
            destination: [v[66], v[67]],
            refusal_code: v[69].max(0.0) as u32,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct ObservationServiceEvents {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub service_events: Vec<ServiceEvent>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct ServiceEvent {
    pub event: String,
    pub tick: usize,
    pub being: usize,
    pub passenger_id: u32,
    pub pickup_station_m: [f32; 2],
    pub destination_station_m: [f32; 2],
    pub refusal_code: u32,
}

pub(crate) fn detect_service_events(
    previous: Option<&ServiceSnapshot>,
    current: &ServiceSnapshot,
    tick: usize,
    being: usize,
) -> Vec<ServiceEvent> {
    let passenger_id = |snapshot: &ServiceSnapshot| {
        if snapshot.passenger_id > 0.5 {
            snapshot.passenger_id as u32
        } else {
            (snapshot.task_key.max(0.0) * 0.5).floor() as u32
        }
    };
    let event = |name: &str, snapshot: &ServiceSnapshot| ServiceEvent {
        event: name.to_string(),
        tick,
        being,
        passenger_id: passenger_id(snapshot),
        pickup_station_m: snapshot.pickup,
        destination_station_m: snapshot.destination,
        refusal_code: snapshot.refusal_code,
    };
    let mut events = Vec::new();
    let issued = !current.onboard
        && current.task_available
        && previous.is_none_or(|old| {
            !old.task_available || (old.task_key - current.task_key).abs() > 0.25
        });
    if previous.is_some_and(|old| !old.onboard) && current.onboard {
        events.push(event("pickup_accepted", current));
    }
    if let Some(old) = previous
        && old.onboard
        && !current.onboard
        && current.deliveries > old.deliveries
    {
        events.push(event("dropoff_accepted", old));
        events.push(event("delivery_incremented", old));
    }
    // Production can issue the next passenger in the same GPU tick that closes the
    // previous delivery. Preserve causal order for replay consumers.
    if issued {
        events.push(event("passenger_issued", current));
    }
    events
}

pub(crate) fn taxi_generation_summary(
    generation: usize,
    alive: usize,
    beings: usize,
    waiting: usize,
    onboard: usize,
    deliveries: u64,
    refusals: usize,
    impacts: u64,
    route_progress_mean: f64,
    energy_mean_j: f64,
    water_mean_kg: f64,
) -> String {
    format!(
        "  [taxi] generation={generation} alive={alive}/{beings} phase_waiting={waiting} \
         phase_onboard={onboard} accepted_deliveries={deliveries} refusals={refusals} \
         impacts={impacts} route_progress_mean={route_progress_mean:.2} \
         energy_mean_j={energy_mean_j:.1} water_mean_kg={water_mean_kg:.3}"
    )
}

/// A persistent, bounded observer. Writers fail loudly instead of dropping evidence.
pub struct DrivingDiagnostics {
    out: PathBuf,
    staging: wgpu::Buffer,
    csv: BufWriter<std::fs::File>,
    samples: BufWriter<std::fs::File>,
    windows: Vec<Window>,
    previous_contact: Vec<bool>,
    previous_position: Vec<[f32; 3]>,
    pub selected: usize,
    pub every: usize,
    generation: usize,
    observed_genome_file: String,
    lineage_id: u64,
    first_tick: usize,
    pending: Option<serde_json::Value>,
    previous_service: Vec<Option<ServiceSnapshot>>,
    service_events: Vec<ServiceEvent>,
    slow_captured: std::cell::Cell<bool>,
}

impl DrivingDiagnostics {
    /// Create new files; never append a second experiment to an existing recording.
    pub fn new(
        device: &wgpu::Device,
        brain: &BrainGpu,
        out: &Path,
        beings: usize,
        selected: usize,
        every: usize,
        organism_scenario: bool,
        ablation: Option<&universe::gpu::brain::AblationPlan>,
    ) -> io::Result<Self> {
        if selected >= beings || every == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "invalid diagnostic being/cadence",
            ));
        }
        let create = |name: &str| {
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(out.join(name))
        };
        let mut csv = BufWriter::new(create("driving.csv")?);
        writeln!(
            csv,
            "generation,first_tick,last_tick,being,ticks,alive_ticks,contact_ticks,impact_ticks,contact_events,brake_ticks,overlap_ticks,stationary_ticks,reverse_ticks,near_wall_ticks,speed_sum,distance_m,collision_energy_j,impulse_ns,impact_peak_mps,slip_peak,slip_angle_peak_rad,throttle_sum,brake_sum,x,z,speed,throttle,steer,brake,wall_left,wall_front,wall_right,life,fuel"
        )?;
        let counterfactual = ablation.map(|plan| {
            json!({
                "being": plan.being(),
                "targets": plan.canonical_spec(),
                "requested_world_tick": plan.from_tick(),
                "effective_tick_semantics": "first neural dispatch at or after requested_world_tick"
            })
        });
        let mut metadata = json!({"schema_version":2,"observation_schema":2,"contact_energy_version":2,"tick_rate":64,"window_ticks":64,
            "eye_offset_m":universe::visao::ALTURA_OLHO,"eye_height_m":universe::visao::ALTURA_OLHO,
            "road_view_calibration_version":2,"reference_eye_height_m":universe::visao::ROAD_VIEW_REFERENCE_EYE_HEIGHT_M,"retina_road_view_m":universe::visao::road_view_range(),
            "retina_geometry":if universe::visao::road_view_enabled(){"physical-v1"}else{"legacy"},
            "fov_degrees":universe::visao::fov_deg(),"retina_pitch_radians":(0..RETINA_ROWS).map(universe::visao::pitch).collect::<Vec<_>>(),
            "selected_being":selected,"sample_every":every,"retina_rows":RETINA_ROWS,
            "retina_cols":RETINA_COLS,"retina_encoding":"u32: material low byte, distance_mm upper 24 bits",
            "wave_channels":WAVE_BYTES,"fast_state_values":FAST_FLOATS,"fast_state_units":"dimensionless",
            "sample_phase":"retina/perception pre-physics; fast state post-brain; body post-physics",
            "motor_trace_source":if organism_scenario {"resident_native_action_v1"} else {"native_action_mapper_v1"},
            "scenario":if organism_scenario {"organism-taxi-v1"} else {"race-v1"},
            "fast_state_stage":"sensory_cfc_pre_l1",
            "brain_runtime":{"output_gain":brain.out_gain,"output_fatigue_beta":brain.out_beta,
                "l2_normalisation_by_std":brain.norm_l2_by_std,
                "mechanisms_off":{"memory":brain.mem_off,"period":brain.period_off,
                    "threshold":brain.threshold_off,"heterogeneous_activation":brain.activation_off},
                "regional_gating":NEUROMOD_REGIOES.load(Ordering::Relaxed),
                "sensor_noise_sigma":brain.noise_sigma,"sensor_dropout_probability":brain.dropout_prob},
            "slow_kinematics":if slow_canonical_kinematics(){"canonical"}else{"legacy"},
            "wall_probes":if continuous_wall_probes(){"continuous"}else{"legacy"},
            "motor_trace_phase":"pre_physics_command_assembly",
            "motor_head":if motor_head_linear(){"linear"}else{"spline"},
            "throttle_arbitration":if throttle_arbitration_signed(){"signed"}else{"legacy"},
            "reverse_controller":if reverse_controller_v2(){"v2"}else{"legacy"},
            "contact_semantics":"four physical probes in wall voxels; impact requires positive inward velocity",
            "near_wall_semantics":"CPU wall distance <=3 m; proximity is not a collision",
            "extra_sample_readback_bytes":SAMPLE_BYTES,
            "trace_gpu_bytes_per_sample":OBSERVATION_BYTES,
            "counterfactual_ablation":counterfactual});
        metadata["lifetime_layout"] = json!({"version":1,"stride":512,"genes":[0,16],
            "feedback":[32,4],"plastic_weights":[36,144],"eligibility":[180,144],
            "previous_latent":[324,48],"previous_motor":[372,3],"runtime_offset":375,
            "inheritance":"genes only; acquired state reset at birth"});
        metadata["brain_graph"] = json!({"version":1,
        "nodes":[
            {"name":"wave","shape":[192],"dtype":"i8","unit":"sensor_code","scale":1,"zero_point":0,"phase":"pre_brain","source":"perception.buf_output","optional":false},
            {"name":"cfc_state","shape":[96],"dtype":"f32","unit":"dimensionless","phase":"post_brain","source":"buf_cfc","optional":true},
            {"name":"l1_matmul","shape":[96],"dtype":"f32","unit":"dimensionless","phase":"post_brain","source":"w_l1r_pre_mechanisms","optional":true},
            {"name":"l1_effective","shape":[96],"dtype":"f32","unit":"dimensionless","phase":"post_brain","source":"w_l1r_post_mechanisms","optional":true},
            {"name":"l1_to_l2","shape":[96],"dtype":"i8","unit":"sensor_code","scale":1,"zero_point":0,"phase":"post_brain","source":"w_l1a","optional":true},
            {"name":"l2_matmul","shape":[48],"dtype":"f32","unit":"dimensionless","phase":"post_brain","source":"raw_l2","optional":true},
                    {"name":"l2_effective","shape":[48],"dtype":"f32","unit":"dimensionless","phase":"post_brain","source":"w_l2r","optional":true},
                    {"name":"l2_std","shape":[],"dtype":"f32","unit":"dimensionless","phase":"post_brain","source":"l2_std","optional":true},
                    {"name":"l2_adapted","shape":[48],"dtype":"f32","unit":"dimensionless","phase":"post_brain","source":"adapted","optional":true},
            {"name":"bypass_adapted","shape":[48],"dtype":"f32","unit":"dimensionless","phase":"post_brain","source":"adapted_bp","optional":true},
            {"name":"final_intent","shape":[96],"dtype":"f32","unit":"dimensionless","phase":"pre_physics","source":"existing_intent_readback","optional":true}
        ],
        "edges":[
            {"from":"wave[0:96]","to":"cfc_state"},{"from":"cfc_state","to":"l1_matmul"},
            {"from":"wave[96:192]","to":"l1_matmul"},{"from":"l1_matmul","to":"l1_effective"},
            {"from":"l1_effective","to":"l1_to_l2"},{"from":"l1_to_l2","to":"l2_matmul"},
            {"from":"l2_matmul","to":"l2_effective"},{"from":"l2_effective","to":"l2_adapted"},
            {"from":"l2_adapted","to":"final_intent[0:48]"},
            {"from":"l1_effective[48:96]","to":"bypass_adapted"},
            {"from":"bypass_adapted","to":"final_intent[48:96]"}
        ]});
        if organism_scenario {
            metadata["service_events"] = json!({
                "version": 1,
                "optional": true,
                "field": "service_events",
                "events": [
                    "passenger_issued",
                    "pickup_accepted",
                    "dropoff_accepted",
                    "delivery_incremented"
                ],
                "fields": {
                    "tick": "simulation_tick",
                    "being": "slot_index",
                    "passenger_id": "dimensionless_integer",
                    "pickup_station_m": "world_xz_metres",
                    "destination_station_m": "world_xz_metres",
                    "refusal_code": "dimensionless_integer"
                },
                "timestamp_semantics": "first observation boundary containing the authoritative body transition"
            });
        }
        serde_json::to_writer_pretty(create("driving.meta.json")?, &metadata)?;
        Ok(Self {
            out: out.to_path_buf(),
            staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("selected-brain-observation"),
                size: SAMPLE_BYTES as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            csv,
            samples: BufWriter::new(create("observation.jsonl")?),
            windows: vec![Window::default(); beings],
            previous_contact: vec![false; beings],
            previous_position: vec![[0.0; 3]; beings],
            selected,
            every,
            generation: 0,
            observed_genome_file: String::new(),
            lineage_id: selected as u64 + 1,
            first_tick: 1,
            pending: None,
            previous_service: vec![None; beings],
            service_events: Vec::new(),
            slow_captured: std::cell::Cell::new(false),
        })
    }

    /// Start a generation and preserve the exact selected genome used by its samples.
    pub fn reset(&mut self, generation: usize, genome: &Genome) -> io::Result<()> {
        let file = format!("observed_genome_gen{generation:04}.bin");
        let path = self.out.join(&file);
        let path = path.to_str().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "diagnostic path is not UTF-8")
        })?;
        save_genome(genome, path)?;
        self.generation = generation;
        self.lineage_id = self.selected as u64 + 1;
        self.observed_genome_file = file;
        self.first_tick = 1;
        self.windows.fill(Window::default());
        self.previous_contact.fill(false);
        self.pending = None;
        self.previous_service.fill(None);
        self.service_events.clear();
        Ok(())
    }

    fn emit_pending(
        &mut self,
        _tick: usize,
        states: &[BeingState],
        raw: &[[f32; 96]],
        slow: &[[f32; 3]],
        motor_trace: Option<&MotorTrace>,
        organism: Option<&[universe::gpu::organism::OrganismState]>,
        resident: bool,
    ) -> io::Result<()> {
        if let Some(mut sample) = self.pending.take() {
            let i = self.selected;
            let p = &states[i].phys;
            let contact = p.contact_diagnostics();
            let tyre_temperature_native =
                (0..4).map(|wheel| p.wheels[wheel * 2 + 1][0]).sum::<f32>() / 4.0;
            let slip_ratio_peak = (0..4)
                .map(|wheel| p.wheels[wheel * 2][1].abs())
                .fold(0.0_f32, f32::max);
            let slip_angle_peak = (0..4)
                .map(|wheel| p.wheels[wheel * 2][2].abs())
                .fold(0.0_f32, f32::max);
            sample["body"] = json!({"x":p.pos[0],"z":p.pos[2],"forward":[p.fwd[0],p.fwd[2]],
                "speed":p.vel[0].hypot(p.vel[2]),"alive":states[i].life_left > 0.0,"life":states[i].life_left,"fuel":states[i].fuel,"fuel_capacity":states[i].fuel_capacity,
                "raw_motor":motor_trace.map(|t| t.raw_intent).unwrap_or_else(|| raw[i][..3].try_into().unwrap()),"applied_motor":[p.throttle,p.steer,p.brake_cmd],
                "slow":motor_trace.map(|t| t.slow_output).or_else(|| slow.get(i).copied()),"motor_trace":motor_trace,
                "contact_mask":contact.contact_mask,"impact_mps":contact.impact_speed_mps,
                "damage":states[i].dano,"engine_temperature_c":p.engine_temp,
                "oil_temperature_c":p.oil_temp,"rpm":p.current_rpm,"rpm_limit":p.max_rpm,
                "rpm_fraction":p.current_rpm / p.max_rpm.max(1.0),"gear":p.current_gear,
                "drive_torque_nm":p.drive_torque_out,"mass_kg":p.mass,"engine_max_torque_nm":p.max_torque,
                "brake_max_torque_nm":p.brake_torque_max,
                "fuel_load_scale":universe::gpu::body::fuel_load_scale(p),
                "tyre_temperature_native":tyre_temperature_native,"tyre_temperature_model":states[i].tire_temp,
                "terrain_grip":p.terrain_grip,"environment_grip":p.env_grip,
                "slip_ratio_peak":slip_ratio_peak,"slip_angle_peak_rad":slip_angle_peak,
                "joy":states[i].joy_tick,"joy_streak_s":states[i].joy_streak,
                "stagnation_ticks":states[i].stagnation_ticks,"hole_hits":states[i].hole_hits});
            if resident {
                sample["body"]["joy"] = serde_json::Value::Null;
                sample["body"]["joy_streak_s"] = serde_json::Value::Null;
                sample["body"]["stagnation_ticks"] = serde_json::Value::Null;
                sample["body"]["hole_hits"] = serde_json::Value::Null;
                sample["body"]["fuel_load_scale"] = serde_json::Value::Null;
                sample["body"]["tyre_temperature_model"] = serde_json::Value::Null;
            }
            if let Some(body) = organism.and_then(|rows| rows.get(i)) {
                let v = &body.values;
                sample["organism"] = json!({"schema_version":1,
                    "age_seconds":v[3],"maturity":v[4],"energy_j":v[0],"water_kg":v[1],"material_kg":v[2],
                    "engine_temp_c":v[9]-273.15,"oil_temp_c":p.oil_temp,"fatigue":v[5],"engine_wear":v[6],
                    "tire_wear":v[7],"integrity":v[8],"repair_engine":v[42],"repair_tire":v[43],
                    "repair_energy_j":v[44],"repair_material_kg":v[45],"alive":v[29],
                    "money":v[64],"passenger":v[65],"deliveries":v[68]});
                sample["taxi"] = json!({"schema_version":1,"money":v[64],"onboard":v[65],
                    "passenger_id":v[72],"deliveries":v[68],"destination_x_m":v[66],"destination_z_m":v[67],
                    "target_dx_m":v[80],"target_dz_m":v[81],"service_dx_m":v[85],"service_dz_m":v[86],
                    "refusal_code":v[69],"energy_received_j":v[76],"water_received_kg":v[77],
                    "material_received_kg":v[78],"service_cost":v[79],"task_age_s":v[74]});
                sample["service_events"] = json!(std::mem::take(&mut self.service_events));
            }
            if sample["neural_dispatch"].as_bool() == Some(true) {
                sample["stages"]["final_intent"] = json!(&raw[i][..]);
            }
            serde_json::to_writer(&mut self.samples, &sample)?;
            writeln!(self.samples)?;
        }
        Ok(())
    }

    pub fn on_birth(
        &mut self,
        being: usize,
        lineage: u64,
        genome: &Genome,
        position: [f32; 3],
    ) -> io::Result<()> {
        self.previous_contact[being] = false;
        self.previous_position[being] = position;
        if being == self.selected {
            self.lineage_id = lineage;
            self.observed_genome_file = format!(
                "observed_genome_gen{:04}_lineage{lineage}.bin",
                self.generation
            );
            let path = self.out.join(&self.observed_genome_file);
            save_genome(
                genome,
                path.to_str()
                    .ok_or_else(|| io::Error::other("non-UTF8 genome path"))?,
            )?;
        }
        Ok(())
    }

    /// Resume observation identity without emitting a birth or reusing a pre-resume window.
    /// Acquired diagnostic aggregation is a new segment starting after `tick`.
    pub fn resume_identity(
        &mut self,
        tick: usize,
        lineage_ids: &[u64],
        genomes: &[Genome],
        positions: &[[f32; 3]],
    ) -> io::Result<()> {
        let n = self.previous_position.len();
        if lineage_ids.len() != n
            || genomes.len() != n
            || positions.len() != n
            || self.selected >= n
            || lineage_ids.iter().any(|id| *id == 0)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "diagnostic resume identity shape",
            ));
        }
        let lineage = lineage_ids[self.selected];
        let file = format!(
            "observed_genome_gen{:04}_lineage{lineage}_resume{tick:012}.bin",
            self.generation
        );
        let path = self.out.join(&file);
        save_genome(
            &genomes[self.selected],
            path.to_str()
                .ok_or_else(|| io::Error::other("non-UTF8 genome path"))?,
        )?;
        self.observed_genome_file = file;
        self.lineage_id = lineage;
        self.previous_position.copy_from_slice(positions);
        self.previous_contact.fill(false);
        self.windows.fill(Window::default());
        self.pending = None;
        self.previous_service.fill(None);
        self.service_events.clear();
        self.first_tick = tick.saturating_add(1);
        Ok(())
    }

    pub fn sampling(&self, tick: usize) -> bool {
        tick > 0 && tick % self.every == 0
    }

    pub fn record_sample(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        retina: &wgpu::Buffer,
        wave: &wgpu::Buffer,
        brain: Option<&BrainGpu>,
        fresh_brain_trace: bool,
    ) {
        self.record_sample_with_slow(encoder, retina, wave, brain, None, fresh_brain_trace);
    }

    pub fn record_sample_with_slow(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        retina: &wgpu::Buffer,
        wave: &wgpu::Buffer,
        brain: Option<&BrainGpu>,
        slow: Option<&SlowBrainGpu>,
        fresh_brain_trace: bool,
    ) {
        encoder.copy_buffer_to_buffer(
            retina,
            (self.selected * RETINA_BYTES) as u64,
            &self.staging,
            0,
            RETINA_BYTES as u64,
        );
        encoder.copy_buffer_to_buffer(
            wave,
            (self.selected * WAVE_BYTES) as u64,
            &self.staging,
            RETINA_BYTES as u64,
            WAVE_BYTES as u64,
        );
        if let Some(brain) = brain {
            brain
                .copy_lifetime_state_into(encoder, &self.staging, self.selected, LIFE_OFFSET as u64)
                .expect("selected lifetime sample must fit staging");
            brain
                .copy_fast_state_into(encoder, &self.staging, self.selected, FAST_OFFSET as u64)
                .expect("selected fast-state sample must fit its staging buffer");
            if fresh_brain_trace {
                brain
                    .copy_observation_into(encoder, &self.staging, TRACE_OFFSET as u64)
                    .expect("selected brain trace must fit its staging buffer");
            }
        }
        self.slow_captured.set(slow.is_some());
        if let Some(slow) = slow {
            slow.copy_diagnostic_into(
                encoder,
                self.selected,
                &self.staging,
                SLOW_INPUT_OFFSET as u64,
                SLOW_STATE_OFFSET as u64,
                SLOW_OUTPUT_OFFSET as u64,
            )
            .expect("selected slow-brain sample must fit staging");
        }
    }

    pub fn attach_traffic(&mut self, values: &[f32; 16]) {
        if let Some(sample) = self.pending.as_mut() {
            sample["traffic_state"] = json!(values);
        }
    }

    pub fn request_sample_read(&self) -> universe::gpu::readback::ReadbackTicket {
        universe::gpu::readback::ReadbackTicket::request(&self.staging, SAMPLE_BYTES as u64)
    }

    pub fn collect_sample(
        &mut self,
        ticket: universe::gpu::readback::ReadbackTicket,
        tick: usize,
        native: bool,
        fresh_brain_trace: bool,
        pose: &universe::gpu::retina_types::PoseRaw,
    ) {
        assert!(ticket.is_ready(), "observation mapping failed");
        let slice = self.staging.slice(..);
        let bytes = slice.get_mapped_range();
        let retina: Vec<u32> = bytemuck::cast_slice(&bytes[..RETINA_BYTES]).to_vec();
        let wave: Vec<i8> = bytes[RETINA_BYTES..RETINA_BYTES + WAVE_BYTES]
            .iter()
            .map(|v| *v as i8)
            .collect();
        let fast: Option<Vec<f32>> =
            native.then(|| bytemuck::cast_slice(&bytes[FAST_OFFSET..TRACE_OFFSET]).to_vec());
        if let Some(values) = &fast {
            assert!(values.iter().all(|v| v.is_finite()), "nonfinite fast state");
        }
        let stages = (native && fresh_brain_trace).then(|| {
            let trace = decode_observation(&bytes[TRACE_OFFSET..LIFE_OFFSET])
                .expect("GPU brain trace must match its declared layout");
            json!({"cfc_state":fast,"l1_matmul":trace.l1_matmul,
                "l1_effective":trace.l1_effective,"l1_to_l2":trace.l1_to_l2,
                "l2_matmul":trace.l2_matmul,"l2_effective":trace.l2_effective,
                "l2_adapted":trace.l2_adapted,"bypass_adapted":trace.bypass_adapted,
                "l2_std":trace.l2_std})
        });
        let lifetime: Option<Vec<f32>> = (native && fresh_brain_trace).then(|| {
            let values: Vec<f32> =
                bytemuck::cast_slice(&bytes[LIFE_OFFSET..BASE_SAMPLE_BYTES]).to_vec();
            assert!(
                values.iter().all(|v| v.is_finite()),
                "nonfinite resident lifetime state"
            );
            values
        });
        let slow_brain = (native && self.slow_captured.get()).then(|| {
            let slow_input: Vec<f32> = bytes[SLOW_INPUT_OFFSET..SLOW_STATE_OFFSET]
                .chunks_exact(4)
                .map(|chunk| i32::from_ne_bytes(chunk.try_into().unwrap()) as f32)
                .collect();
            let slow_state: Vec<f32> =
                bytemuck::cast_slice(&bytes[SLOW_STATE_OFFSET..SLOW_OUTPUT_OFFSET]).to_vec();
            let slow_output: Vec<f32> =
                bytemuck::cast_slice(&bytes[SLOW_OUTPUT_OFFSET..SAMPLE_BYTES]).to_vec();
            assert!(slow_state.iter().all(|value| value.is_finite()));
            assert!(slow_output.iter().all(|value| value.is_finite()));
            json!({
            "input": slow_input,
            "state": {
                "cfc": &slow_state[..64],
                "fatigue": &slow_state[64..128],
                "counter_bits": slow_state[128].to_bits(),
                "tick_counter_bits": slow_state[129].to_bits(),
            },
            "output": slow_output,
            "intermediate_mlp": null,
            "layout": {"input": [48], "cfc": [64], "fatigue": [64], "output": [3]},
            "phases": ["gather", "run_cfc", "run_dense"],
            })
        });
        self.pending = Some(
            json!({"generation":self.generation,"tick":tick,"being":self.selected,
            "genome_file":self.observed_genome_file,"lineage_id":self.lineage_id,
            "retina":retina,"wave":wave,"fast_state":fast,
            "neural_dispatch":fresh_brain_trace,"stages":stages,
            "lifetime_state":lifetime,
            "slow_brain":slow_brain,
            "sensor_pose":{"position":&pose.pos_gaze[..3],"forward":&pose.dirs[..2],"right":&pose.dirs[2..],"gaze_yaw":pose.pos_gaze[3]}}),
        );
        drop(bytes);
        self.staging.unmap();
    }

    pub fn after_physics(
        &mut self,
        tick: usize,
        states: &[BeingState],
        alive: &[bool],
        raw: &[[f32; 96]],
        slow: &[[f32; 3]],
        motor_trace: Option<&MotorTrace>,
        organism: Option<&[universe::gpu::organism::OrganismState]>,
    ) -> io::Result<()> {
        if tick == 0 {
            for (previous, state) in self.previous_position.iter_mut().zip(states) {
                *previous = state.phys.pos;
            }
            return Ok(());
        }
        for (i, state) in states.iter().enumerate() {
            let p = &state.phys;
            let contact = p.contact_diagnostics();
            let w = &mut self.windows[i];
            w.ticks += 1;
            if alive[i] {
                w.alive += 1;
                let touching = contact.contact_mask != 0;
                w.contact += u64::from(touching);
                w.impacts += u64::from(contact.impulse_ns > 0.0);
                w.events += u64::from(touching && !self.previous_contact[i]);
                w.brake += u64::from(p.brake_cmd > 0.05);
                w.overlap += u64::from(p.brake_cmd > 0.05 && p.throttle > 0.1);
                let speed = p.vel[0].hypot(p.vel[2]);
                w.speed_sum += speed as f64;
                w.stationary += u64::from(speed < 0.5);
                w.reverse += u64::from(p.current_gear == 0);
                w.near_wall += u64::from(
                    state
                        .dist_wall_f
                        .min(state.dist_wall_l.min(state.dist_wall_r))
                        <= 3.0,
                );
                w.distance += (p.pos[0] - self.previous_position[i][0])
                    .hypot(p.pos[2] - self.previous_position[i][2])
                    as f64;
                w.energy += contact.energy_loss_j as f64;
                w.impulse += contact.impulse_ns as f64;
                w.impact_peak = w.impact_peak.max(contact.impact_speed_mps);
                for wheel in p.wheels.iter().step_by(2) {
                    w.slip_peak = w.slip_peak.max(wheel[1].abs());
                    w.slip_angle_peak = w.slip_angle_peak.max(wheel[2].abs());
                }
                w.throttle_sum += p.throttle as f64;
                w.brake_sum += p.brake_cmd as f64;
                self.previous_contact[i] = touching;
            }
            self.previous_position[i] = p.pos;
        }
        if let Some(rows) = organism {
            for (being, body) in rows.iter().enumerate() {
                let current = ServiceSnapshot::from_body(body);
                if being == self.selected {
                    self.service_events.extend(detect_service_events(
                        self.previous_service[being].as_ref(),
                        &current,
                        tick,
                        being,
                    ));
                }
                self.previous_service[being] = Some(current);
            }
        }
        self.emit_pending(
            tick,
            states,
            raw,
            slow,
            motor_trace,
            organism,
            organism.is_some(),
        )?;
        if tick % 64 == 0 {
            self.flush_window(tick, states)?;
        }
        Ok(())
    }

    /// Consume the resident telemetry window produced by the GPU adapter.
    ///
    /// The adapter owns the per-tick accumulation, including contact edges and
    /// distances. This method therefore copies slots 104..122 directly into
    /// the report window and never re-counts host samples. Call it once at the
    /// same 64-tick boundary used by `flush_window`.
    pub fn after_physics_resident(
        &mut self,
        tick: usize,
        states: &[BeingState],
        alive: &[bool],
        raw: &[[f32; 96]],
        slow: &[[f32; 3]],
        motor_trace: Option<&MotorTrace>,
        organism: &[universe::gpu::organism::OrganismState],
    ) -> io::Result<()> {
        if states.len() != self.windows.len()
            || alive.len() != states.len()
            || organism.len() != states.len()
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "resident telemetry shape",
            ));
        }
        for (window, body) in self.windows.iter_mut().zip(organism) {
            let v = &body.values;
            if v.len() < 123 || v[104..123].iter().any(|value| !value.is_finite()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "resident telemetry values",
                ));
            }
            window.ticks = v[104].max(0.0) as u64;
            window.alive = v[105].max(0.0) as u64;
            window.contact = v[106].max(0.0) as u64;
            window.impacts = v[107].max(0.0) as u64;
            window.events = v[108].max(0.0) as u64;
            window.brake = v[109].max(0.0) as u64;
            window.overlap = v[110].max(0.0) as u64;
            window.stationary = v[111].max(0.0) as u64;
            window.reverse = v[112].max(0.0) as u64;
            window.near_wall = v[113].max(0.0) as u64;
            window.speed_sum = v[114] as f64;
            window.distance = v[115] as f64;
            window.energy = v[116] as f64;
            window.impulse = v[117] as f64;
            window.impact_peak = v[118];
            window.slip_peak = v[119];
            window.slip_angle_peak = v[120];
            window.throttle_sum = v[121] as f64;
            window.brake_sum = v[122] as f64;
        }
        for (being, body) in organism.iter().enumerate() {
            let current = ServiceSnapshot::from_body(body);
            if being == self.selected {
                self.service_events.extend(detect_service_events(
                    self.previous_service[being].as_ref(),
                    &current,
                    tick,
                    being,
                ));
            }
            self.previous_service[being] = Some(current);
        }
        self.emit_pending(tick, states, raw, slow, motor_trace, Some(organism), true)?;
        if tick % 64 == 0 {
            self.flush_window(tick, states)?;
        }
        Ok(())
    }

    pub fn flush_window(&mut self, tick: usize, states: &[BeingState]) -> io::Result<()> {
        for (i, (w, s)) in self.windows.iter().zip(states).enumerate() {
            if w.ticks == 0 {
                continue;
            }
            let p: &PhysicsStateGpu = &s.phys;
            writeln!(
                self.csv,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                self.generation,
                tick.saturating_add(1).saturating_sub(w.ticks as usize),
                tick,
                i,
                w.ticks,
                w.alive,
                w.contact,
                w.impacts,
                w.events,
                w.brake,
                w.overlap,
                w.stationary,
                w.reverse,
                w.near_wall,
                w.speed_sum,
                w.distance,
                w.energy,
                w.impulse,
                w.impact_peak,
                w.slip_peak,
                w.slip_angle_peak,
                w.throttle_sum,
                w.brake_sum,
                p.pos[0],
                p.pos[2],
                p.vel[0].hypot(p.vel[2]),
                p.throttle,
                p.steer,
                p.brake_cmd,
                s.dist_wall_l,
                s.dist_wall_f,
                s.dist_wall_r,
                s.life_left,
                s.fuel
            )?;
        }
        self.windows.fill(Window::default());
        self.first_tick = tick + 1;
        self.csv.flush()?;
        self.samples.flush()?;
        Ok(())
    }
}
