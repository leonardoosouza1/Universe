//! Optional driving evidence: exact per-tick contacts and sampled sensory/state data.
//!
//! Contacts arrive in the existing physics readback. Only the selected being's
//! retina, perception and fast state are copied at the configured display cadence.
//! No metric is fed back into the controller by this observer.

use crate::native_action::MotorTrace;
use crate::statics::{
    continuous_wall_probes, motor_head_linear, reverse_controller_v2, slow_canonical_kinematics,
    throttle_arbitration_signed,
};
use crate::types::BeingState;
use serde_json::json;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use universe::config::iara::brain::{L1_NEURONS, SENSORY_WAVE_SIZE};
use universe::config::iara::vision::{RETINA_COLS, RETINA_ROWS, RETINA_TOTAL};
use universe::gpu::brain::BrainGpu;
use universe::gpu::physics::PhysicsStateGpu;

const WAVE_BYTES: usize = SENSORY_WAVE_SIZE;
const FAST_FLOATS: usize = L1_NEURONS;
const RETINA_BYTES: usize = RETINA_TOTAL * 4;
const SAMPLE_BYTES: usize = RETINA_BYTES + WAVE_BYTES + FAST_FLOATS * 4;

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

/// A persistent, bounded observer. Writers fail loudly instead of dropping evidence.
pub struct DrivingDiagnostics {
    staging: wgpu::Buffer,
    csv: BufWriter<std::fs::File>,
    samples: BufWriter<std::fs::File>,
    windows: Vec<Window>,
    previous_contact: Vec<bool>,
    previous_position: Vec<[f32; 3]>,
    pub selected: usize,
    pub every: usize,
    generation: usize,
    first_tick: usize,
    pending: Option<serde_json::Value>,
}

impl DrivingDiagnostics {
    /// Create new files; never append a second experiment to an existing recording.
    pub fn new(
        device: &wgpu::Device,
        out: &Path,
        beings: usize,
        selected: usize,
        every: usize,
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
        let metadata = json!({"schema_version":1,"contact_energy_version":2,"tick_rate":64,"window_ticks":64,
            "eye_offset_m":universe::visao::ALTURA_OLHO,"eye_height_m":universe::visao::ALTURA_OLHO,
            "road_view_calibration_version":2,"reference_eye_height_m":universe::visao::ROAD_VIEW_REFERENCE_EYE_HEIGHT_M,"retina_road_view_m":universe::visao::road_view_range(),
            "retina_geometry":if universe::visao::road_view_enabled(){"physical-v1"}else{"legacy"},
            "fov_degrees":universe::visao::fov_deg(),"retina_pitch_radians":(0..RETINA_ROWS).map(universe::visao::pitch).collect::<Vec<_>>(),
            "selected_being":selected,"sample_every":every,"retina_rows":RETINA_ROWS,
            "retina_cols":RETINA_COLS,"retina_encoding":"u32: material low byte, distance_mm upper 24 bits",
            "wave_channels":WAVE_BYTES,"fast_state_values":FAST_FLOATS,"fast_state_units":"dimensionless",
            "sample_phase":"retina/perception pre-physics; fast state post-brain; body post-physics",
            "motor_trace_source":"native_action_mapper_v1",
            "fast_state_stage":"sensory_cfc_pre_l1",
            "slow_kinematics":if slow_canonical_kinematics(){"canonical"}else{"legacy"},
            "wall_probes":if continuous_wall_probes(){"continuous"}else{"legacy"},
            "motor_trace_phase":"pre_physics_command_assembly",
            "motor_head":if motor_head_linear(){"linear"}else{"spline"},
            "throttle_arbitration":if throttle_arbitration_signed(){"signed"}else{"legacy"},
            "reverse_controller":if reverse_controller_v2(){"v2"}else{"legacy"},
            "contact_semantics":"four physical probes in wall voxels; impact requires positive inward velocity",
            "near_wall_semantics":"CPU wall distance <=3 m; proximity is not a collision",
            "extra_sample_readback_bytes":SAMPLE_BYTES});
        serde_json::to_writer_pretty(create("driving.meta.json")?, &metadata)?;
        Ok(Self {
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
            first_tick: 1,
            pending: None,
        })
    }

    pub fn reset(&mut self, generation: usize) {
        self.generation = generation;
        self.first_tick = 1;
        self.windows.fill(Window::default());
        self.previous_contact.fill(false);
        self.pending = None;
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
                .copy_fast_state_into(
                    encoder,
                    &self.staging,
                    self.selected,
                    (RETINA_BYTES + WAVE_BYTES) as u64,
                )
                .expect("selected fast-state sample must fit its staging buffer");
        }
    }

    pub fn read_sample(
        &mut self,
        device: &wgpu::Device,
        tick: usize,
        native: bool,
        pose: &universe::gpu::retina_types::PoseRaw,
    ) {
        let slice = self.staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .expect("observation callback")
            .expect("observation mapping");
        let bytes = slice.get_mapped_range();
        let retina: Vec<u32> = bytemuck::cast_slice(&bytes[..RETINA_BYTES]).to_vec();
        let wave: Vec<i8> = bytes[RETINA_BYTES..RETINA_BYTES + WAVE_BYTES]
            .iter()
            .map(|v| *v as i8)
            .collect();
        let fast: Option<Vec<f32>> =
            native.then(|| bytemuck::cast_slice(&bytes[RETINA_BYTES + WAVE_BYTES..]).to_vec());
        if let Some(values) = &fast {
            assert!(values.iter().all(|v| v.is_finite()), "nonfinite fast state");
        }
        self.pending = Some(
            json!({"generation":self.generation,"tick":tick,"being":self.selected,
            "retina":retina,"wave":wave,"fast_state":fast,
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
        if let Some(mut sample) = self.pending.take() {
            let i = self.selected;
            let p = &states[i].phys;
            let contact = p.contact_diagnostics();
            let tyre_temperature_native =
                (0..4).map(|wheel| p.wheels[wheel * 2 + 1][0]).sum::<f32>() / 4.0;
            let slip_ratio_peak = (0..4)
                .map(|wheel| p.wheels[wheel * 2 + 1][1].abs())
                .fold(0.0_f32, f32::max);
            let slip_angle_peak = (0..4)
                .map(|wheel| p.wheels[wheel * 2 + 1][2].abs())
                .fold(0.0_f32, f32::max);
            sample["body"] = json!({"x":p.pos[0],"z":p.pos[2],"forward":[p.fwd[0],p.fwd[2]],
                "speed":p.vel[0].hypot(p.vel[2]),"alive":alive[i],"life":states[i].life_left,"fuel":states[i].fuel,
                "raw_motor":&raw[i][..3],"applied_motor":[p.throttle,p.steer,p.brake_cmd],
                "slow":slow.get(i),"motor_trace":motor_trace,
                "contact_mask":contact.contact_mask,"impact_mps":contact.impact_speed_mps,
                "damage":states[i].dano,"engine_temperature_c":p.engine_temp,
                "oil_temperature_c":p.oil_temp,"rpm":p.current_rpm,"rpm_limit":p.max_rpm,
                "rpm_fraction":p.current_rpm / p.max_rpm.max(1.0),"gear":p.current_gear,
                "drive_torque_nm":p.drive_torque_out,
                "tyre_temperature_native":tyre_temperature_native,
                "tyre_temperature_model":states[i].tire_temp,
                "terrain_grip":p.terrain_grip,"environment_grip":p.env_grip,
                "slip_ratio_peak":slip_ratio_peak,"slip_angle_peak_rad":slip_angle_peak,
                "joy":states[i].joy_tick,"joy_streak_s":states[i].joy_streak,
                "stagnation_ticks":states[i].stagnation_ticks,"hole_hits":states[i].hole_hits});
            serde_json::to_writer(&mut self.samples, &sample)?;
            writeln!(self.samples)?;
        }
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
                self.first_tick,
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
