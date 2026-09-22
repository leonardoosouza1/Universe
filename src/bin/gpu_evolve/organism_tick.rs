//! Resident scenario scheduling inside GpuCtx: existing retina, brain and physics.
//! Host work records commands and samples observations; it never advances a body.
use super::*;
use bytemuck::Zeroable;
use universe::gpu::organism_action::*;

impl GpuCtx {
    pub(super) fn tick_organism(
        &mut self,
        states: &mut [BeingState],
        n: usize,
        dt: f32,
        tick: usize,
    ) -> Result<(), &'static str> {
        if !self.organism_resident_ready {
            for state in states.iter_mut() {
                state.phys.dt = dt;
                state.phys.engine_temp = 20.0;
                state.phys.oil_temp = 20.0;
                if self.taxi_witness.is_some() {
                    state.phys.config_flags |= universe::gpu::physics::STATIC_REST_FLAG;
                }
            }
            self.phys_gpu
                .upload_resident_state(&states.iter().map(|s| s.phys).collect::<Vec<_>>())?;
            let action = self
                .organism_action
                .as_ref()
                .ok_or("missing resident action")?;
            action.upload_gear_state(
                &self.queue,
                &states
                    .iter()
                    .map(|s| GearState {
                        gear: s.phys.current_gear,
                        ..GearState::zeroed()
                    })
                    .collect::<Vec<_>>(),
            )?;
            self.organism_resident_ready = true;
        }
        let started = std::time::Instant::now();
        let diagnostic = self
            .driving_diagnostics
            .as_ref()
            .is_some_and(|d| d.sampling(tick));
        let profile = perfil_kernel() > 0 && tick % perfil_kernel() as usize == 0;
        let sample = self.force_organism_sample
            || diagnostic
            || tick % 64 == 0
            || (self.replay_capture && tick % 4 == 0);
        self.brain_tick = self.brain_tick.wrapping_add(1);
        let fires =
            self.brain_tick == 1 || self.brain_skip <= 1 || self.brain_tick % self.brain_skip == 0;
        self.brain.set_observed_being(
            (diagnostic && fires).then(|| self.driving_diagnostics.as_ref().unwrap().selected),
        );
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("organism-resident-tick"),
            });
        let body = self.organism.as_ref().ok_or("missing resident body")?;
        let action = self
            .organism_action
            .as_ref()
            .ok_or("missing resident action")?;
        let taxi = self.taxi.as_ref().ok_or("missing resident taxi")?;
        let orientation = self
            .organism_orientation
            .as_ref()
            .ok_or("missing resident orientation")?;
        self.organism_sensors
            .as_ref()
            .ok_or("missing resident sensors")?
            .encode(
                &self.device,
                &mut enc,
                self.phys_gpu.input_buffer(),
                &body.buf_state,
                self.retina_digest.extra_buffer(),
                self.percept.temporal_buffer(),
                &self.buf_poses_retina,
                self.percept.poses_buffer(),
                self.brain.buf_intent(),
                n,
                dt,
            )?;
        self.percept
            .copy_retina_prev(&mut enc, &self.buf_retina_out);
        let mut pitches = [[0.0; 4]; 4];
        for row in 0..RETINA_ROWS {
            pitches[row / 4][row % 4] = pitch(row);
        }
        let params = RetinaParamsGpu {
            grid_size: self.grid_size,
            n_beings: n as u32,
            cell_size: self.cell_size,
            fov_half: (universe::visao::fov_deg() / 2.0).to_radians(),
            pitches,
            beings_on: retina_beings_on() as u32,
            _pad0: universe::visao::road_view_enabled() as u32,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("organism-retina-params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("organism-retina"),
            layout: &self.layout_retina,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.buf_grid.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.buf_poses_retina.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.buf_retina_out.as_entire_binding(),
                },
            ],
        });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("organism-retina"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_retina);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((n as u32 * RETINA_TOTAL as u32).div_ceil(64), 1, 1);
        }
        self.percept
            .encode_resident(&mut enc, &self.device, n, &self.buf_retina_out);
        self.retina_digest.bind_and_dispatch(
            &mut enc,
            &self.device,
            &self.queue,
            n,
            &self.buf_retina_out,
            self.percept.buf_output(),
        );
        taxi.encode_senses(
            &self.device,
            &mut enc,
            &body.buf_state,
            self.phys_gpu.input_buffer(),
            self.percept.buf_output(),
            n,
        );
        self.traffic
            .as_ref()
            .ok_or("missing traffic runtime")?
            .encode_senses(
                &self.device,
                &mut enc,
                self.phys_gpu.input_buffer(),
                self.percept.buf_output(),
                n,
                tick as u32,
            )?;
        if fires {
            if self.erase_fast_memory {
                self.brain.clear_fast_memory(&mut enc, n);
            }
            self.organism_feedback.as_ref().unwrap().encode(
                &self.device,
                &mut enc,
                &body.buf_state,
                self.brain.lifetime_buffer(),
                n,
            );
            self.brain.set_world_tick(tick as u32);
            self.brain
                .bind_and_dispatch(&mut enc, n, &[], dt, Some(self.percept.buf_output()));
        }
        self.slow
            .dispatch_gather(&mut enc, self.percept.buf_output(), n);
        self.slow.dispatch(&mut enc, n, dt);
        enc.copy_buffer_to_buffer(
            self.brain.buf_intent(),
            0,
            &action.buf_intent,
            0,
            (n * 96 * 4) as u64,
        );
        if slow_ablated() {
            enc.clear_buffer(&action.buf_slow, 0, Some((n * 3 * 4) as u64));
        } else {
            let buffers = self.slow.snapshot_buffers();
            let output = buffers
                .iter()
                .find(|(name, _)| *name == "slow.output")
                .ok_or("slow output contract")?
                .1;
            enc.copy_buffer_to_buffer(output, 0, &action.buf_slow, 0, (n * 3 * 4) as u64);
        }
        // The reference witness is a separate, explicit controller arm.  It
        // owns the intent producer but still goes through the ordinary action
        // mapper, so gearbox, steering smoothing, brake gates and physics laws
        // remain the production path.  Slow/hippocampal signals must not leak
        // into this arm's command.
        if let Some(witness) = self.taxi_witness.as_ref() {
            let taxi = self.taxi.as_ref().ok_or("missing taxi for witness")?;
            enc.clear_buffer(&action.buf_slow, 0, Some((n * 3 * 4) as u64));
            enc.clear_buffer(&action.buf_hip_bias, 0, Some((n * 3 * 4) as u64));
            witness.encode(
                &self.device,
                &self.queue,
                &mut enc,
                self.percept.buf_output(),
                self.phys_gpu.input_buffer(),
                &body.buf_state,
                &taxi.buf_routes,
                &taxi.buf_world,
                &action.buf_intent,
                n,
            )?;
        }
        let flags = FLAG_ALLOW_LOW_SPEED_BRAKE
            | if motor_head_linear() || self.taxi_witness.is_some() {
                FLAG_LINEAR_HEAD
            } else {
                0
            }
            | if throttle_arbitration_signed() {
                FLAG_SIGNED_THROTTLE
            } else {
                0
            };
        action.encode(
            &self.device,
            &self.queue,
            &mut enc,
            self.phys_gpu.input_buffer(),
            n,
            ActionParams {
                n: n as u32,
                flags,
                steering_alpha: STEER_ALPHA.load(Ordering::Relaxed),
                slow_steering_authority: SLOW_STEER_AUTH.load(Ordering::Relaxed),
                gearbox_mode: CAMBIO_MODO.load(Ordering::Relaxed),
                gear_cap: CAMBIO_TETO.load(Ordering::Relaxed),
                dt,
                _pad: 0.0,
            },
        )?;
        let pose_staging = diagnostic.then(|| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("observed-sensor-pose"),
                size: 96,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        });
        if diagnostic {
            let diagnostics = self.driving_diagnostics.as_ref().unwrap();
            enc.copy_buffer_to_buffer(
                &self.buf_poses_retina,
                (diagnostics.selected * 32) as u64,
                pose_staging.as_ref().unwrap(),
                0,
                32,
            );
            enc.copy_buffer_to_buffer(
                &action.buf_trace,
                (diagnostics.selected * 64) as u64,
                pose_staging.as_ref().unwrap(),
                32,
                64,
            );
            diagnostics.record_sample_with_slow(
                &mut enc,
                &self.buf_retina_out,
                self.percept.buf_output(),
                Some(&self.brain),
                Some(&self.slow),
                fires,
            );
        }
        body.encode_pre(&self.device, &mut enc, self.phys_gpu.input_buffer(), n, dt)?;
        self.phys_gpu
            .encode_step(&mut enc, self.phys_gpu.input_buffer(), n)?;
        self.traffic.as_ref().unwrap().encode_contacts(
            &self.device,
            &mut enc,
            self.phys_gpu.input_buffer(),
            self.phys_gpu.output_buffer(),
            &body.buf_state,
            n,
        )?;
        orientation.encode_rotation_gpu(
            &self.device,
            &mut enc,
            self.phys_gpu.output_buffer(),
            n,
            dt,
        )?;
        self.organism_telemetry.as_ref().unwrap().encode(
            &self.device,
            &mut enc,
            self.phys_gpu.input_buffer(),
            self.phys_gpu.output_buffer(),
            &body.buf_state,
            n,
            tick,
        )?;
        self.traffic.as_ref().unwrap().encode_post(
            &self.device,
            &mut enc,
            self.phys_gpu.output_buffer(),
            &body.buf_state,
            n,
            tick as u32,
        )?;
        body.encode_post(&self.device, &mut enc, self.phys_gpu.output_buffer(), n, dt)?;
        let taxi_intent = if self.taxi_witness.is_some() {
            &action.buf_intent
        } else {
            self.brain.buf_intent()
        };
        taxi.encode_post(
            &self.device,
            &mut enc,
            self.phys_gpu.output_buffer(),
            &body.buf_state,
            taxi_intent,
            n,
            dt,
            tick as u32,
        );
        orientation.encode_feedback_to_input(
            &self.device,
            &mut enc,
            self.phys_gpu.output_buffer(),
            self.phys_gpu.input_buffer(),
            n,
        )?;
        self.queue.submit(Some(enc.finish()));
        if sample {
            let intent_read = self
                .brain
                .request_intent_read(n)
                .ok_or("intent sample request")?;
            let slow_read = self
                .slow
                .request_output_read(n)
                .ok_or("slow sample request")?;
            let diagnostic_read = diagnostic.then(|| {
                self.driving_diagnostics
                    .as_ref()
                    .unwrap()
                    .request_sample_read()
            });
            self.device.poll(wgpu::Maintain::Wait);
            let intents = self.brain.collect_intent_read(n, intent_read);
            let slow = self.slow.collect_output_read(n, slow_read);
            if intents.len() != n || slow.len() != n {
                return Err("incomplete sampled brain state");
            }
            self.last_intents = intents.clone();
            self.replay_raw_motor = intents.iter().map(|i| [i[0], i[1], i[2]]).collect();
            let physical = self.phys_gpu.read_resident_state(n)?;
            for (host, resident) in states.iter_mut().zip(physical) {
                host.phys = resident;
            }
            self.sample_organism(states)?;
            self.traffic_samples =
                self.traffic
                    .as_ref()
                    .unwrap()
                    .download(&self.device, &self.queue, n);
            for (host, resident) in states.iter_mut().zip(&self.organism_samples) {
                host.ticks_alive = (resident.values[3] * 64.0).round() as u32;
                host.dist_wall_l = resident.values[124];
                host.dist_wall_r = resident.values[125];
                host.dist_wall_f = resident.values[126];
            }
            let mut motor_trace = None;
            if diagnostic {
                let pose_buffer = pose_staging.as_ref().unwrap();
                let (sender, receiver) = std::sync::mpsc::channel();
                pose_buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| {
                        let _ = sender.send(r);
                    });
                self.device.poll(wgpu::Maintain::Wait);
                receiver
                    .recv()
                    .map_err(|_| "pose callback")?
                    .map_err(|_| "pose mapping")?;
                let data = pose_buffer.slice(..).get_mapped_range();
                let pose: PoseRaw = bytemuck::pod_read_unaligned(&data[..32]);
                let trace: &[f32] = bytemuck::cast_slice(&data[32..96]);
                let selected = self.driving_diagnostics.as_ref().unwrap().selected;
                let state = &states[selected].phys;
                motor_trace = Some(crate::native_action::MotorTrace {
                    source: if self.taxi_witness.is_some() { "resident_reference_action_v1" } else { "resident_native_action_v1" },
                    phase: "pre_physics_command_assembly",
                    pre_physics_speed_mps: trace[6],
                    raw_intent: if self.taxi_witness.is_some() {
                        trace[..3].try_into().unwrap()
                    } else {
                        intents[selected][..3].try_into().unwrap()
                    },
                    motor_head: if motor_head_linear() || self.taxi_witness.is_some() {
                        "linear"
                    } else {
                        "spline"
                    },
                    throttle_arbitration: if throttle_arbitration_signed() {
                        "signed"
                    } else {
                        "legacy"
                    },
                    after_head: trace[..3].try_into().unwrap(),
                    slow_output: if self.taxi_witness.is_some() { [0.0;3] } else { slow[selected] },
                    hip_bias: [0.0; 3],
                    after_modulation: trace[3..6].try_into().unwrap(),
                    steering_filtered: trace[7],
                    brake_suppression: if trace[8] > 0.5 {
                        crate::native_action::BrakeSuppression::LowSpeedGate
                    } else if trace[5] <= 0.0 {
                        crate::native_action::BrakeSuppression::NonpositiveAfterModulation
                    } else {
                        crate::native_action::BrakeSuppression::None
                    },
                    override_cause: if self.organism_samples[selected].values[29] < 0.5 {
                        crate::native_action::OverrideCause::DeadBody
                    } else {
                        crate::native_action::OverrideCause::None
                    },
                    final_command: [state.throttle, state.steer, state.brake_cmd],
                });
                if self.taxi_witness.is_some() {
                    self.replay_raw_motor[selected] = trace[..3].try_into().unwrap();
                }
                drop(data);
                pose_buffer.unmap();
                self.driving_diagnostics.as_mut().unwrap().collect_sample(
                    diagnostic_read.unwrap(),
                    tick,
                    true,
                    fires,
                    &pose,
                );
            }
            if let Some(diagnostics) = self.driving_diagnostics.as_mut() {
                if let Some(traffic) = self.traffic_samples.get(diagnostics.selected) { diagnostics.attach_traffic(traffic); }
                let alive = states.iter().map(|s| s.life_left > 0.0).collect::<Vec<_>>();
                diagnostics
                    .after_physics_resident(
                        tick,
                        states,
                        &alive,
                        &intents,
                        &slow,
                        motor_trace.as_ref(),
                        &self.organism_samples,
                    )
                    .map_err(|_| "resident diagnostics write")?;
            }
        }
        if profile {
            println!("[resident-profile] tick={tick} simulation_submit=1 sampled={sample} host_wall_ms={:.3} numerical_host_steps=0",started.elapsed().as_secs_f64()*1000.0);
        }
        Ok(())
    }
}
