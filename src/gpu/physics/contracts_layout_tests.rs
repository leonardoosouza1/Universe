//! Static host/WGSL ABI checks for the production A → B → C physics pipeline.
//! These tests parse layouts only; they create no device and execute no physics.
use super::{PhysicsOutputGpu, PhysicsStateGpu, INPUT_WGSL_STRIDE};
use std::mem::{offset_of, size_of};

macro_rules! offsets {
    ($ty:ty; $($field:ident),+ $(,)?) => {
        &[$((stringify!($field), offset_of!($ty, $field),
            std::mem::size_of_val(&<$ty as bytemuck::Zeroable>::zeroed().$field))),+]
    };
}

fn check_struct(
    shader: &str,
    module: &naga::Module,
    layout: &naga::proc::Layouter,
    name: &str,
    host_size: usize,
    host_stride: usize,
    host_fields: &[(&str, usize, usize)],
) {
    let (handle, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("{shader}: missing {name}"));
    let naga::TypeInner::Struct { members, span } = &ty.inner else {
        panic!("{shader}: {name} must be a struct");
    };
    assert_eq!(
        members.len(),
        host_fields.len(),
        "{shader}: {name} field count"
    );
    for (member, &(field, offset, size)) in members.iter().zip(host_fields) {
        assert_eq!(
            member.name.as_deref(),
            Some(field),
            "{shader}: {name} field order"
        );
        assert_eq!(
            member.offset as usize, offset,
            "{shader}: {name}.{field} offset"
        );
        assert_eq!(
            layout[member.ty].size as usize, size,
            "{shader}: {name}.{field} size"
        );
    }
    let (_, last_offset, last_size) = *host_fields.last().unwrap();
    assert_eq!(
        last_offset + last_size,
        host_size,
        "{shader}: {name} host extent"
    );
    assert!(
        host_size <= host_stride,
        "{shader}: {name} host state exceeds serialized stride"
    );
    assert_eq!(
        *span as usize, host_stride,
        "{shader}: {name} span / serialized stride"
    );
    assert_eq!(
        layout[handle].size as usize, host_stride,
        "{shader}: {name} array element size"
    );
    assert_eq!(
        layout[handle].to_stride() as usize,
        host_stride,
        "{shader}: {name} array stride"
    );
}

#[test]
fn production_physics_buffers_match_host_abi() {
    let input = offsets!(PhysicsStateGpu;
        vel,
        av_y,
        wheels,
        pos,
        _pad_pos,
        fwd,
        _pad_fwd,
        lat,
        _pad_lat,
        max_torque,
        current_rpm,
        max_rpm,
        throttle,
        brake_cmd,
        steer,
        drive_torque_out,
        current_gear,
        shift_timer,
        is_stalled,
        starter_timer,
        flywheel_inertia,
        engine_temp,
        oil_temp,
        mass,
        drag_coeff,
        wheel_base,
        cg_height,
        track_width,
        lsd_lock_ratio,
        brake_torque_max,
        drivetrain,
        mu_peak,
        cornering_stiffness,
        wheel_radius,
        wheel_inertia,
        optimal_temp,
        susp_stiffness,
        susp_damping,
        susp_rest,
        susp_max_travel,
        _pad_before_wind,
        wind,
        _pad_wind,
        terrain_grip,
        env_grip,
        aquaplaning,
        config_flags,
        dt,
        _pad_dt,
    );
    let output = offsets!(PhysicsOutputGpu;
        vel,
        av_y,
        wheels,
        pos,
        _pad_pos,
        fwd,
        _pad_fwd,
        lat,
        _pad_lat,
        current_rpm,
        drive_torque_out,
        shift_timer,
        is_stalled,
        starter_timer,
        engine_temp,
        oil_temp,
        _pad_eng,
    );
    for (name, source, has_output) in [
        ("physics_a.wgsl", include_str!("../physics_a.wgsl"), false),
        ("physics_b.wgsl", include_str!("../physics_b.wgsl"), false),
        ("physics_c.wgsl", include_str!("../physics_c.wgsl"), true),
    ] {
        let module = naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|error| panic!("{name}: {}", error.emit_to_string(source)));
        let mut layout = naga::proc::Layouter::default();
        layout
            .update(module.to_ctx())
            .unwrap_or_else(|error| panic!("{name}: {error}"));
        check_struct(
            name,
            &module,
            &layout,
            "CarInput",
            size_of::<PhysicsStateGpu>(),
            INPUT_WGSL_STRIDE,
            input,
        );
        if has_output {
            check_struct(
                name,
                &module,
                &layout,
                "CarOutput",
                size_of::<PhysicsOutputGpu>(),
                size_of::<PhysicsOutputGpu>(),
                output,
            );
        }
    }
}

/// Two distinct records prove the upload adapter honors the padded WGSL stride.
#[test]
fn input_serializer_preserves_each_record_and_zeroes_padding() {
    use bytemuck::Zeroable;
    let mut first = PhysicsStateGpu::zeroed();
    let mut second = PhysicsStateGpu::zeroed();
    first.current_gear = 3;
    first.dt = 0.125;
    second.current_gear = 6;
    second.dt = 0.25;
    let input = [first, second];
    let bytes = super::serialize_input(&input);
    assert_eq!(bytes.len(), input.len() * INPUT_WGSL_STRIDE);
    for (record, chunk) in input.iter().zip(bytes.chunks_exact(INPUT_WGSL_STRIDE)) {
        assert_eq!(
            &chunk[..size_of::<PhysicsStateGpu>()],
            bytemuck::bytes_of(record)
        );
        assert!(chunk[size_of::<PhysicsStateGpu>()..]
            .iter()
            .all(|&value| value == 0));
    }
}
