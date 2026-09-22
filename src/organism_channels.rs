//! Executed resident-city wave manifest, not the legacy host/motor channel registry.
//! Sources are embedded from production shaders so formula excerpts retain provenance.
//! Formula excerpts containing local variables are deliberately not presented as a
//! complete physical conversion: inspect the linked producer for its definitions.
use serde_json::{json, Value};
const SPATIAL: &str = include_str!("gpu/perception.wgsl");
const TEMPORAL: &str = include_str!("gpu/perception_temporal.wgsl");
const MOTION: &str = include_str!("gpu/perception_motion.wgsl");

/// Complete ordered 192-channel description for organism-city-observation-v2.
/// `normalization` describes the stored signed byte, except body/task/traffic rows
/// which provide the explicitly stated pre-quantization feature and quantization.
pub fn channel_manifest() -> Value {
    let mut channels = Vec::with_capacity(192);
    for ch in 0..160 {
        let motion = matches!(ch,21..=27|83|84|98|114..=117);
        let temporal = matches!(ch,20|64..=75|78..=81|85..=89|99..=113|118..=127);
        let (producer, source) = if motion {
            ("perception_motion.wgsl", MOTION)
        } else if temporal {
            ("perception_temporal.wgsl", TEMPORAL)
        } else {
            ("perception.wgsl", SPATIAL)
        };
        let selector = match ch {
            0..=7 => "being, bin,",
            8..=15 => "being, 8u + bin,",
            16..=19 => "being, 16u + ai,",
            21..=27 => "being, 21u + row,",
            28..=31 => "being, 24u + ai,",
            32..=39 => "being, 32u + ai,",
            40..=47 => "being, 40u + di,",
            48..=55 => "being, 48u + si,",
            128..=143 => "being, 128u + bin,",
            144..=159 => "being, 144u + bin,",
            _ => "",
        }
        .to_string();
        let selector = if selector.is_empty() {
            format!("being, {ch}u,")
        } else {
            selector
        };
        let writer = source
            .lines()
            .enumerate()
            .find(|(_, line)| !line.trim_start().starts_with("//") && line.contains(&selector));
        let (line, expression) = writer
            .map(|(line, text)| (line + 1, text.trim()))
            .unwrap_or((0, "writer not resolved; do not infer a physical scale"));
        let name = match ch {
            0..=7 => format!("RetinaFarPool[{}]", ch),
            8..=15 => format!("RetinaNearPool[{}]", ch - 8),
            16..=19 => format!("WallDistanceRay[{}]", ch - 16),
            20 => "EnergyFractionDuplicate".into(),
            28..=31 => format!("WallWidthRay[{}]", ch - 24),
            32..=39 => format!("FlowCrossRay[{}]", ch - 32),
            40..=47 => format!("RoadNear[{}m]", 5 + (ch - 40) * 5),
            48..=55 => format!("RoadFar[{}m]", 45 + (ch - 48) * 15),
            56..=59 => format!("SurfaceMaterial[{}m]", [0, 3, 8, 20][ch - 56]),
            60..=63 => format!("EscapeDirectionDuplicate[{}]", ch - 60),
            67 => "IntegrityTimesAlive".into(),
            75 => "StationaryDuration".into(),
            78..=80 => format!("HeritableArcCurvature[{}]", ch - 78),
            89 => "ActualReverseGear".into(),
            90..=93 => format!("DarkCenterDuplicate[{}]", ch - 90),
            94..=97 => format!("DiagonalWhisker[{}deg]", [30, -30, 60, -60][ch - 94]),
            120 => "PreviousBrainThrottleIntent".into(),
            126 => "PreviousDecisionConsequence".into(),
            127 => "PositiveConsequenceStreak".into(),
            128..=143 => format!("RetinaFoveaHD[{}]", ch - 128),
            144..=159 => format!("RetinaPeripheralHD[{}]", ch - 144),
            _ => crate::perception::channels::name_for(ch).to_string(),
        };
        let normalization = match ch {
            0..=15|128..=159 => "max pool of legacy_encoding(material,distance); spatial retina bins; perif_real changes HD angular coverage",
            16..=19 => "0 if no wall; otherwise i32(clamp(127*(wall_range-distance)/wall_range,1,127)); hereditary wall_range",
            20 => "i32(clamp(energy_J/max(1,capacity_J)*127,0,127)); explicitly replaces earlier spatial byte",
            21..=27|98 => "i32(clamp(sum_columns(abs(current_distance-previous_distance))/MOTION_NORM*127,0,127))",
            28..=31 => "i32(clamp(sum(abs(lateral_probe_offset) where wall_or_hole)/6*127,0,127))",
            32..=39 => "i32(clamp(cross(forward,local_flow)*127,-127,127)); resident path has no host curve overwrite",
            40..=47 => "i32(clamp(surface_code*(0.5+length(flow)*0.5),-127,127)); Road60 Grass-60 Wall127 Hole-127",
            56..=59 => "Road0 Grass64 Wall100 Hole-127",
            60..=63 => "i32(clamp(escape_norm*127,-127,127)); exact run center calculation in producer",
            65 => "i32(clamp(speed_m_s*2,0,127))",
            66 => "i32(clamp(lateral_velocity_m_s*10,-127,127))",
            67 => "i32(clamp(integrity*alive*127,0,127))",
            68 => "i32(clamp(current_rpm/max_rpm,0,1)*127)",
            69 => "i32(clamp(actual_gear*20,0,127))",
            71 => "i32(clamp((engine_temperature_C-20)/130,0,1)*127)",
            73 => "grass=-128; otherwise0",
            75 => "i32(min(stationary_seconds,1)*127); reset when speed>=0.5m/s",
            89 => "actual gear0 ->127, otherwise0; no host reversing controller",
            99|100 => "previous-tick retina_digest extra edge6/edge7 through resident bridge; i32(clamp(edge*127,0,127))",
            102..=104 => "i32(clamp((previous_probe_m-current_probe_m)/(1/64)/30*127,-127,127)); temporal kernel uses fixed1/64s",
            106|107 => "i32(clamp((previous_probe_m-current_probe_m)/(1/64)/15*64,-127,127))",
            120 => "i32(clamp(previous brain intent[0],0,1)*127); not CfC hidden state",
            125 => "social enabled: nearby<40m=>0.5, ahead<20m=>1; otherwise0; byte=feature*127",
            126 => "clamp(delivery_delta-2*integrity_loss+0.2*positive_energy_delta/capacity,-1,1)*127 since feedback baseline",
            127 => "positive-consequence consecutive_seconds/5 clamped0..1, then*127",
            _ => "literal writer expression below; local-variable physical conversion not expanded here",
        };
        channels.push(json!({"index":ch,"name":name,"normalization":normalization,
            "unit":"encoded i8; physical feature described in normalization/source",
            "producer":format!("src/gpu/{producer}"),"source_line":line,"writer_expression":expression,
            "active":if line==0 {serde_json::Value::Null}else{json!(true)},"status":if line==0 {"unresolved_static_mapping"}else if ch==125{"conditional_social_flag"}else{"written"},
            "upstream":if temporal{"organism_sensors.wgsl resident bridge + persistent temporal state"}else{"resident retina/current world geometry"}}));
    }
    let body_task = [
        ("EnergyFraction", "energy_J/max(capacity_J,1e-6)", "J/J"),
        ("WaterFraction", "water_kg/max(capacity_kg,1e-6)", "kg/kg"),
        (
            "MaterialFraction",
            "material_kg/max(capacity_kg,1e-6)",
            "kg/kg",
        ),
        ("BodyTemperature", "(temperature_K-293.15)/130", "K"),
        ("Fatigue", "fatigue", "fraction"),
        ("EngineWear", "engine_wear", "fraction"),
        ("TyreWear", "tyre_wear", "fraction"),
        ("Integrity", "integrity", "fraction"),
        ("Maturity", "maturity", "fraction"),
        ("Age", "age_seconds/7200", "s"),
        (
            "EngineCapacityRatio",
            "effective_engine/max(birth_engine,1e-6)",
            "ratio",
        ),
        (
            "BrakeCapacityRatio",
            "effective_brake/max(birth_brake,1e-6)",
            "ratio",
        ),
        (
            "TrafficLightPresent",
            "nearest relevant directional light within100m:1 else0",
            "boolean",
        ),
        (
            "TrafficLightDistance",
            "distance_to_stop_line_m/100; absent=1",
            "m",
        ),
        ("RepairResourceScale", "repair_resource_scale", "fraction"),
        ("Alive", "alive", "boolean"),
        ("Money", "money/100", "currency"),
        ("PassengerOnboard", "onboard", "boolean"),
        ("TaskDeltaX", "target_x-position_x, divided100", "m"),
        ("TaskDeltaZ", "target_z-position_z, divided100", "m"),
        ("PassengerWaitingAvailable", "waitingavailable", "boolean"),
        ("TaskAge", "task_age_seconds/300", "s"),
        (
            "TrafficLightPhase",
            "red=-1 yellow=0 green=1; absent=0 (usepresence)",
            "enum",
        ),
        (
            "TaskRefusal",
            "refusal_code/6; code7unfundedbirth saturates",
            "enum",
        ),
        (
            "ServiceDeltaX",
            "nearest_service_x-position_x, divided100",
            "m",
        ),
        (
            "ServiceDeltaZ",
            "nearest_service_z-position_z, divided100",
            "m",
        ),
        (
            "StationEnergyFraction",
            "nearest_station_energy_stock/capacity",
            "fraction",
        ),
        (
            "StationWaterFraction",
            "nearest_station_water_stock/capacity",
            "fraction",
        ),
        (
            "StationMaterialFraction",
            "nearest_station_material_stock/capacity",
            "fraction",
        ),
        ("ServiceAvailable", "serviceavailable", "boolean"),
        ("PassengerFare", "fare/100", "currency"),
        (
            "TrafficTimeToChange",
            "ticks_until_next_phase/full_cycle_ticks; absent0",
            "fraction",
        ),
    ];
    for (offset, (name, normalization, unit)) in body_task.iter().enumerate() {
        let ch = 160 + offset;
        let traffic = matches!(ch, 172 | 173 | 182 | 191);
        channels.push(json!({"index":ch,"name":name,"normalization":normalization,"unit":unit,
            "quantization":"round(clamp(feature,-1,1)*127), signed i8",
            "producer":if traffic{"src/gpu/traffic.wgsl::senses"}else{"src/gpu/taxi.wgsl::senses"},
            "active":true,"status":"written","upstream":if traffic{"resident physics input + immutable stop lines + absolute tick"}else{"previous completed organism/taxi post state"}}));
    }
    json!({"schema":crate::gpu::traffic::TRAFFIC_OBSERVATION_ID,"channel_count":192,
        "encoding":"48 u32 words per being; four two's-complement i8 bytes per word, least-significant byte first",
        "execution_order":["resident sensors -> temporal extras","retina","perception clear -> spatial -> temporal -> motion","retina_digest -> separate extra buffer only","taxi senses overwrites160..191","traffic senses overwrites172,173,182,191","brain"],
        "scope":"organism resident city only; legacy BODY remap disabled; static producer map; unresolved rows explicitly marked; no channels192..207",
        "retina_digest":"CfC hidden exposures live in separate19-float extra buffer, not extra brain input dimensions; edge extras enter99/100 next tick",
        "channels":channels})
}

#[cfg(test)]
mod tests {
    #[test]
    fn resident_channel_manifest_covers_executed_wave() {
        let manifest = super::channel_manifest();
        let channels = manifest["channels"].as_array().unwrap();
        assert_eq!(channels.len(), 192);
        for (index, channel) in channels.iter().enumerate() {
            assert_eq!(channel["index"], index);
            if index < 160 {
                assert!(
                    channel["source_line"].as_u64().unwrap() > 0,
                    "unresolved writer: {index}"
                );
            }
        }
        for index in [172, 173, 182, 191] {
            assert_eq!(channels[index]["producer"], "src/gpu/traffic.wgsl::senses");
        }
        assert_eq!(channels[20]["name"], "EnergyFractionDuplicate");
        assert_eq!(channels[28]["name"], "WallWidthRay[4]");
        assert_eq!(channels[32]["name"], "FlowCrossRay[0]");
    }
}
