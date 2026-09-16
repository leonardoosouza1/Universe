//! Stable binary layout for one selected being's fast-brain trace.

use std::fmt;

use crate::config::iara::brain::{L1_NEURONS, L2_NEURONS};

pub const OBS_L1_MATMUL_OFFSET: usize = 0;
pub const OBS_L1_EFFECTIVE_OFFSET: usize = OBS_L1_MATMUL_OFFSET + L1_NEURONS;
pub const OBS_L1_TO_L2_OFFSET: usize = OBS_L1_EFFECTIVE_OFFSET + L1_NEURONS;
pub const OBS_L2_MATMUL_OFFSET: usize = OBS_L1_TO_L2_OFFSET + L1_NEURONS;
pub const OBS_L2_EFFECTIVE_OFFSET: usize = OBS_L2_MATMUL_OFFSET + L2_NEURONS;
pub const OBS_L2_ADAPTED_OFFSET: usize = OBS_L2_EFFECTIVE_OFFSET + L2_NEURONS;
pub const OBS_BYPASS_ADAPTED_OFFSET: usize = OBS_L2_ADAPTED_OFFSET + L2_NEURONS;
pub const OBS_L2_STD_OFFSET: usize = OBS_BYPASS_ADAPTED_OFFSET + L2_NEURONS;
pub const OBSERVATION_WORDS: usize = OBS_L2_STD_OFFSET + 1;
pub const OBSERVATION_BYTES: usize = OBSERVATION_WORDS * std::mem::size_of::<u32>();

#[derive(Clone, Debug, PartialEq)]
pub struct BrainObservation {
    pub l1_matmul: Vec<f32>,
    pub l1_effective: Vec<f32>,
    pub l1_to_l2: Vec<i8>,
    pub l2_matmul: Vec<f32>,
    pub l2_effective: Vec<f32>,
    pub l2_adapted: Vec<f32>,
    pub bypass_adapted: Vec<f32>,
    pub l2_std: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObservationDecodeError(String);

impl fmt::Display for ObservationDecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ObservationDecodeError {}

pub fn decode_observation(bytes: &[u8]) -> Result<BrainObservation, ObservationDecodeError> {
    if bytes.len() != OBSERVATION_BYTES {
        return Err(ObservationDecodeError(format!(
            "brain observation has {} bytes; expected {OBSERVATION_BYTES}",
            bytes.len()
        )));
    }
    let words: &[u32] = bytemuck::cast_slice(bytes);
    let floats = |start: usize, len: usize| -> Vec<f32> {
        words[start..start + len]
            .iter()
            .map(|word| f32::from_bits(*word))
            .collect()
    };
    let l1_to_l2: Vec<i8> = words[OBS_L1_TO_L2_OFFSET..OBS_L1_TO_L2_OFFSET + L1_NEURONS]
        .iter()
        .map(|word| i32::from_ne_bytes(word.to_ne_bytes()))
        .map(|value| {
            i8::try_from(value).map_err(|_| {
                ObservationDecodeError(format!(
                    "l1_to_l2 contains {value}, outside the declared i8 range"
                ))
            })
        })
        .collect::<Result<_, _>>()?;
    let observation = BrainObservation {
        l1_matmul: floats(OBS_L1_MATMUL_OFFSET, L1_NEURONS),
        l1_effective: floats(OBS_L1_EFFECTIVE_OFFSET, L1_NEURONS),
        l1_to_l2,
        l2_matmul: floats(OBS_L2_MATMUL_OFFSET, L2_NEURONS),
        l2_effective: floats(OBS_L2_EFFECTIVE_OFFSET, L2_NEURONS),
        l2_adapted: floats(OBS_L2_ADAPTED_OFFSET, L2_NEURONS),
        bypass_adapted: floats(OBS_BYPASS_ADAPTED_OFFSET, L2_NEURONS),
        l2_std: f32::from_bits(words[OBS_L2_STD_OFFSET]),
    };
    let all_finite = observation
        .l1_matmul
        .iter()
        .chain(&observation.l1_effective)
        .chain(&observation.l2_matmul)
        .chain(&observation.l2_effective)
        .chain(&observation.l2_adapted)
        .chain(&observation.bypass_adapted)
        .chain(std::iter::once(&observation.l2_std))
        .all(|value| value.is_finite());
    if !all_finite {
        return Err(ObservationDecodeError(
            "brain observation contains a non-finite value".into(),
        ));
    }
    Ok(observation)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_the_documented_481_words() {
        assert_eq!(OBS_L1_EFFECTIVE_OFFSET, 96);
        assert_eq!(OBS_L1_TO_L2_OFFSET, 192);
        assert_eq!(OBS_L2_MATMUL_OFFSET, 288);
        assert_eq!(OBS_L2_EFFECTIVE_OFFSET, 336);
        assert_eq!(OBS_L2_ADAPTED_OFFSET, 384);
        assert_eq!(OBS_BYPASS_ADAPTED_OFFSET, 432);
        assert_eq!(OBS_L2_STD_OFFSET, 480);
        assert_eq!(OBSERVATION_WORDS, 481);
        assert_eq!(OBSERVATION_BYTES, 1_924);
    }

    #[test]
    fn decoder_preserves_float_bits_and_signed_quantised_values() {
        let mut words = vec![0u32; OBSERVATION_WORDS];
        words[OBS_L1_MATMUL_OFFSET] = 1.25f32.to_bits();
        words[OBS_L1_EFFECTIVE_OFFSET + 95] = (-2.5f32).to_bits();
        words[OBS_L1_TO_L2_OFFSET] = (-127i32) as u32;
        words[OBS_L1_TO_L2_OFFSET + 95] = 126i32 as u32;
        words[OBS_L2_MATMUL_OFFSET + 47] = 3.5f32.to_bits();
        words[OBS_L2_STD_OFFSET] = 0.75f32.to_bits();

        let decoded = decode_observation(bytemuck::cast_slice(&words)).unwrap();
        assert_eq!(decoded.l1_matmul[0], 1.25);
        assert_eq!(decoded.l1_effective[95], -2.5);
        assert_eq!(decoded.l1_to_l2[0], -127);
        assert_eq!(decoded.l1_to_l2[95], 126);
        assert_eq!(decoded.l2_matmul[47], 3.5);
        assert_eq!(decoded.l2_std, 0.75);
    }

    #[test]
    fn decoder_rejects_wrong_size_invalid_i8_and_nonfinite_values() {
        assert!(decode_observation(&[]).is_err());
        let mut words = vec![0u32; OBSERVATION_WORDS];
        words[OBS_L1_TO_L2_OFFSET] = 128;
        assert!(decode_observation(bytemuck::cast_slice(&words)).is_err());
        words[OBS_L1_TO_L2_OFFSET] = 0;
        words[OBS_L2_STD_OFFSET] = f32::NAN.to_bits();
        assert!(decode_observation(bytemuck::cast_slice(&words)).is_err());
    }
}
